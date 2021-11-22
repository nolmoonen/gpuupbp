// Copyright (C) 2021, Nol Moonen
// Copyright (C) 2014, Petr Vevoda, Martin Sik (http://cgg.mff.cuni.cz/~sik/),
// Tomas Davidovic (http://www.davidovic.cz),
// Iliyan Georgiev (http://www.iliyan.com/),
// Jaroslav Krivanek (http://cgg.mff.cuni.cz/~jaroslav/)
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom
// the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
// OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// (The above is MIT License: http://en.wikipedia.origin/wiki/MIT_License)

#ifndef KERNEL_MAP_FUNCTS_BP2D_CUH
#define KERNEL_MAP_FUNCTS_BP2D_CUH

#include "../../kernel/bsdf.cuh"
#include "../../kernel/defs.cuh"
#include "../../kernel/intersection.cuh"
#include "../../kernel/medium.cuh"
#include "../../kernel/optix_util.cuh"
#include "../../kernel/params_def.cuh"
#include "../../kernel/path_weight.cuh"
#include "../../kernel/phase_function.cuh"
#include "../../kernel/types.cuh"
#include "../../shared/intersect_defs.h"

#include <cuda_runtime.h>
#include <optix.h>

struct ParamsBP2D {
    float3 contrib;
    SubpathState* camera_state;
    BSDF* camera_bsdf;
    float3 hit_point;
};

__forceinline__ __device__ void eval_bp2d_contrib(ParamsBP2D* p, const LightBeam* light_beam)
{
    // the intersection distance along the ray
    float isect_dist;
    // square of the distance from the intersection point to the point location
    float isect_rad_sqr;
    // test whether photon beam intersects the camera disc of camera point
    bool intersected = test_intersection_bre(
        isect_dist, isect_rad_sqr, light_beam->ray, 0.f, light_beam->beam_length, p->hit_point, params.radius_bp2d_2);

    if (!intersected) return;

    GPU_ASSERT(isect_dist != 0.f);

    // reject if full path length below/above min/max path length
    // photon path length equals all segments up to the photon vertex
    // camera path length equals all segments up to the camera vertex
    // we create an additional segment connecting photon and camera vertex
    unsigned int path_length = light_beam->path_length + p->camera_state->path_length;
    if (path_length > params.max_path_length || path_length < params.min_path_length) {
        return;
    }

    // compute attenuation in current segment and overall PDFs. only valid for
    // homogeneous media

    // attenuation along the beam from origin to intersection
    float3 attenuation = eval_attenuation(light_beam->medium, isect_dist);
    const float pdf = get_min_positive_attenuation_coeff_comp(light_beam->medium, attenuation);
    if (params.photon_beam_type == SHORT_BEAM) attenuation /= pdf;

    // ray sampling pdfs in the segment
    // no need to test ray sampling flags, we know end is in medium
    float ray_sample_for_pdf = light_beam->medium->min_positive_attenuation_coeff_comp * pdf;
    float ray_sample_rev_pdf = (light_beam->ray_sampling_flags & ORIGIN_IN_MEDIUM) ? ray_sample_for_pdf : pdf;
    // ratio of a probability that the beam is sampled long enough for
    // intersection and a pdf of sampling a scattering point at intersection
    // todo at some point restore heterogeneous media sampling
    float ray_sample_for_pdfs_ratio = 1.f / light_beam->medium->min_positive_attenuation_coeff_comp;
    float ray_sample_rev_pdfs_ratio = ray_sample_for_pdfs_ratio;

    if (!is_positive(attenuation)) return;

    // multiply pdfs of current segment with pdfs of previous segments to
    // get overall pdfs
    ray_sample_for_pdf *= light_beam->ray_sample_for_pdf;
    ray_sample_rev_pdf *= light_beam->ray_sample_rev_pdf;
    GPU_ASSERT(ray_sample_for_pdf > 0.f);
    GPU_ASSERT(ray_sample_rev_pdf > 0.f);

    // update the weights (s-1)
    VertexWeights weights_l = light_beam->weights.weights; // copy
    if (light_beam->path_length > 1u) {
        float f_i = get_pde_factor(weights_l.prev_in_medium, // i - 1 in medium
                                   weights_l.prev_specular,  // i - 1 specular
                                   weights_l.ray_sample_for_pdf_inv,
                                   weights_l.ray_sample_for_pdfs_ratio,
                                   1.f / ray_sample_rev_pdf,
                                   weights_l.ray_sample_rev_pdfs_ratio,
                                   light_beam->weights.last_sin_theta); // sin(i-2,i-1,i)

        weights_l.d_pde = light_beam->weights.d_pde_a * f_i + light_beam->weights.d_pde_b;
        weights_l.d_bpt = light_beam->weights.d_bpt_a * f_i + light_beam->weights.d_bpt_b;
    }

    // retrieve camera incoming direction in world coordinates
    // (points away from scattering location)
    const float3 camera_direction = world_dir_fix(*p->camera_bsdf);

    // evaluate the scattering function (from camera to light)
    float bsdf_dir_pdf_w, bsdf_rev_pdf_w, sinTheta;
    const float3 bsdf_factor = PhaseFunction::evaluate(camera_direction,
                                                       -light_beam->ray.direction,
                                                       light_beam->medium->mean_cosine,
                                                       &bsdf_dir_pdf_w,
                                                       &bsdf_rev_pdf_w,
                                                       &sinTheta);
    if (is_black_or_negative(bsdf_factor)) return;

    bsdf_dir_pdf_w *= p->camera_bsdf->continuation_prob;
    GPU_ASSERT(bsdf_dir_pdf_w > 0.f);

    bsdf_rev_pdf_w *= light_beam->medium->continuation_prob;
    GPU_ASSERT(bsdf_rev_pdf_w > 0.f);

    // update MIS weights light
    {
        if (!(light_beam->path_length == 1 && light_beam->is_infinite_light)) {
            weights_l.d_shared *= isect_dist * isect_dist;
        }
        weights_l.d_shared /= ray_sample_for_pdf;
        weights_l.d_bpt /= ray_sample_for_pdf;
        weights_l.d_pde /= ray_sample_for_pdf;
        weights_l.ray_sample_for_pdf_inv = 1.f / ray_sample_for_pdf;
        weights_l.ray_sample_rev_pdf_inv = 1.f / ray_sample_rev_pdf;
        weights_l.ray_sample_for_pdfs_ratio = ray_sample_for_pdfs_ratio;
        weights_l.ray_sample_rev_pdfs_ratio = ray_sample_rev_pdfs_ratio;
    }

    // Epanechnikov kernel
    const float kernel = (1.f - isect_rad_sqr / params.radius_bp2d_2) / (params.radius_bp2d_2 * PI_F * .5f);
    if (!is_positive(kernel)) return;

    // scattering coefficient, is evaluated at query location
    float3 scattering_coeff = light_beam->medium->scattering_coeff;

    // unweighted result
    const float3 unweighted_result =
        light_beam->throughput_at_origin * attenuation * scattering_coeff * bsdf_factor * kernel;

    if (is_black_or_negative(unweighted_result)) return;

    // index t - 1
    const VertexWeights& weights_c = p->camera_state->weights.weights;

    float for_pdf_factor =
        params.photon_beam_type == LONG_BEAM ? weights_l.ray_sample_for_pdf_inv : weights_l.ray_sample_for_pdfs_ratio;
    float rev_pdf_factor =
        params.camera_beam_type == LONG_BEAM ? weights_c.ray_sample_for_pdf_inv : weights_c.ray_sample_for_pdfs_ratio;

    // scaling factor: 1 / (n_v * e_{v,s}) = 1 / (n_v * e_{v,t})
    float val = 1.f / (params.mis_factor_bp2d * for_pdf_factor);

    float bpt_s_sub_1 = !weights_l.prev_specular;

    /** Eq 4.52 */
    const float w_light = val * (weights_l.d_shared * bpt_s_sub_1 * params.subpath_count_bpt +
                                 bsdf_dir_pdf_w * weights_l.d_pde / weights_l.ray_sample_rev_pdf_inv);

    /** Eq. 4.17 */
    const float w_local =
        // evaluate on t - 1 == s - 1
        1.f + // val / val
        val * (params.mis_factor_bb1d * sinTheta * for_pdf_factor * rev_pdf_factor +
               params.mis_factor_pb2d * rev_pdf_factor + params.mis_factor_pp3d);

    float bpt_t_sub_1 = !weights_c.prev_specular;

    /** Eq 4.53 */
    const float w_camera = val * (weights_c.d_shared * bpt_t_sub_1 * params.subpath_count_bpt +
                                  bsdf_rev_pdf_w * weights_c.d_pde / weights_c.ray_sample_rev_pdf_inv);

    /** Eq 4.18 */
    const float mis_weight = 1.f / (w_light + w_local + w_camera);

    // weight and accumulate contribution.
    p->contrib += mis_weight * unweighted_result;
}

__forceinline__ __device__ float3 eval_bp2d(float3 hit_point, SubpathState* camera_state, BSDF* camera_bsdf)
{
    ParamsBP2D p;
    p.contrib = make_float3(0.f);
    p.camera_state = camera_state;
    p.camera_bsdf = camera_bsdf;
    p.hit_point = hit_point;

    unsigned int u0, u1;
    pack_pointer(&p, u0, u1);
#ifdef DBG_CORE_UTIL
    // this trace call does not make recursive trace calls
    params.trace_count[optixGetLaunchIndex().x]++;
#endif
    optixTrace(params.handle_beams,
               hit_point,
               make_float3(EPS_LAUNCH),
               0.f,
               EPS_LAUNCH,
               0.f,
               GEOM_MASK_BP2D,
               OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               RAY_TYPE2_BP2D,
               RAY_TYPE2_COUNT,
               RAY_TYPE_COUNT + RAY_TYPE2_BP2D,
               u0,
               u1);

    return p.contrib;
}

#endif // KERNEL_MAP_FUNCTS_BP2D_CUH
