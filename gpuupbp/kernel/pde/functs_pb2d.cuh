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

#ifndef KERNEL_MAP_FUNCTS_PB2D_CUH
#define KERNEL_MAP_FUNCTS_PB2D_CUH

#include "../../kernel/bsdf.cuh"
#include "../../kernel/intersection.cuh"
#include "../../kernel/medium.cuh"
#include "../../kernel/optix_util.cuh"
#include "../../kernel/params_def.cuh"
#include "../../kernel/path_weight.cuh"
#include "../../kernel/phase_function.cuh"
#include "../../kernel/scene.cuh"
#include "../../kernel/types.cuh"
#include "../../shared/intersect_defs.h"
#include "../../shared/light_vertex.h"

#include <optix.h>

struct ParamsPB2D {
    float3 contrib;
    const Ray* query_ray;
    float dist_min;
    float dist_max;
    const Medium* med;
    float ray_sample_for_pdf;
    float ray_sample_rev_pdf;
    unsigned int ray_sampling_flags;
    unsigned int camera_path_length;
    const StateWeights* camera_weights;
};

__forceinline__ __device__ void eval_pb2d(ParamsPB2D* p, const LightVertex* light_vertex)
{
    GPU_ASSERT(light_vertex != nullptr);
    GPU_ASSERT(light_vertex->is_in_medium());

    // the intersection distance along the ray
    float isect_dist;
    // square of the distance from the intersection point to the point location
    float isect_rad_sqr;
    // test whether query beam intersects the photon disc of photon point
    bool intersected = test_intersection_bre(isect_dist,
                                             isect_rad_sqr,
                                             *p->query_ray,
                                             p->dist_min,
                                             p->dist_max,
                                             light_vertex->hitpoint,
                                             params.radius_pb2d_2);

    if (!intersected) return;

    GPU_ASSERT(isect_dist > 0.f);

    // reject if full path length below/above min/max path length
    // photon path length equals all segments up to the photon vertex
    // camera path length equals all segments up to the camera vertex
    // we create an additional segment connecting photon and camera vertex
    const unsigned int path_length = light_vertex->path_length + p->camera_path_length;
    if (path_length > params.max_path_length || path_length < params.min_path_length) {
        return;
    }

    // compute attenuation in current segment and overall PDFs. only valid for
    // homogeneous media

    float3 attenuation = eval_attenuation(p->med, isect_dist - p->dist_min);
    const float pdf = get_min_positive_attenuation_coeff_comp(p->med, attenuation);
    if (params.camera_beam_type == SHORT_BEAM) attenuation /= pdf;

    float ray_sample_for_pdf = p->med->min_positive_attenuation_coeff_comp * pdf;
    float ray_sample_rev_pdf = (p->ray_sampling_flags & ORIGIN_IN_MEDIUM) ? ray_sample_for_pdf : pdf;
    // todo at some point restore heterogeneous media sampling
    float ray_sample_for_pdfs_ratio = 1.f / p->med->min_positive_attenuation_coeff_comp;
    float ray_sample_rev_pdfs_ratio = ray_sample_for_pdfs_ratio;

    if (!is_positive(attenuation)) return;

    ray_sample_for_pdf *= p->ray_sample_for_pdf;
    ray_sample_rev_pdf *= p->ray_sample_rev_pdf;
    GPU_ASSERT(ray_sample_for_pdf > 0.f);
    GPU_ASSERT(ray_sample_rev_pdf > 0.f);

    // update the weights (t-1)
    VertexWeights weights_c = p->camera_weights->weights; // copy
    if (p->camera_path_length > 1u) {
        float f_i = get_pde_factor(weights_c.prev_in_medium, // i - 1 in medium
                                   weights_c.prev_specular,  // i - 1 specular
                                   1.f / ray_sample_rev_pdf, // 1 / pr_{T,i-1}(x)
                                   weights_c.ray_sample_rev_pdfs_ratio,
                                   weights_c.ray_sample_for_pdf_inv, // 1 / pl_{T,i-1}(x)
                                   weights_c.ray_sample_for_pdfs_ratio,
                                   p->camera_weights->last_sin_theta); // sin(i-2,i-1,i)

        weights_c.d_pde = p->camera_weights->d_pde_a * f_i + p->camera_weights->d_pde_b;
        weights_c.d_bpt = p->camera_weights->d_bpt_a * f_i + p->camera_weights->d_bpt_b;
    }

    // retrieve light incoming direction in world coordinates
    // (points away from scattering location)
    const float3 photon_direction = world_dir_fix(light_vertex->bsdf);

    // evaluate the scattering function (from camera to light)
    float bsdf_dir_pdf_w, bsdf_rev_pdf_w, sinTheta;
    const float3 bsdf_factor = PhaseFunction::evaluate(
        -p->query_ray->direction, photon_direction, p->med->mean_cosine, &bsdf_dir_pdf_w, &bsdf_rev_pdf_w, &sinTheta);
    if (is_black_or_negative(bsdf_factor)) return;

    bsdf_dir_pdf_w *= p->med->continuation_prob;
    GPU_ASSERT(bsdf_dir_pdf_w > 0.f);

    // Even though this is PDF from camera BSDF, the continuation probability
    // must come from light BSDF, because that would govern it if light path
    // actually continued.
    bsdf_rev_pdf_w *= light_vertex->bsdf.continuation_prob;
    GPU_ASSERT(bsdf_rev_pdf_w > 0.f);

    // update MIS weights
    {
        weights_c.d_shared *= isect_dist * isect_dist;
        weights_c.d_shared /= ray_sample_for_pdf;
        weights_c.d_bpt /= ray_sample_for_pdf;
        weights_c.d_pde /= ray_sample_for_pdf;
        weights_c.ray_sample_for_pdf_inv = 1.f / ray_sample_for_pdf;
        weights_c.ray_sample_rev_pdf_inv = 1.f / ray_sample_rev_pdf;
        weights_c.ray_sample_for_pdfs_ratio = ray_sample_for_pdfs_ratio;
        weights_c.ray_sample_rev_pdfs_ratio = ray_sample_rev_pdfs_ratio;
    }

    // Epanechnikov kernel
    const float kernel = (1.f - isect_rad_sqr / params.radius_pb2d_2) / (params.radius_pb2d_2 * PI_F * .5f);
    if (!is_positive(kernel)) return;

    // scattering coefficient
    const float3& scattering_coeff = p->med->scattering_coeff;

    // unweighted result
    const float3 unweighted_result = light_vertex->throughput * attenuation * scattering_coeff * bsdf_factor * kernel;

    if (is_black_or_negative(unweighted_result)) return;

    // index s - 1
    const VertexWeights& weights_l = light_vertex->weights;

    float for_pdf_factor =
        params.photon_beam_type == LONG_BEAM ? weights_l.ray_sample_for_pdf_inv : weights_l.ray_sample_for_pdfs_ratio;
    float rev_pdf_factor =
        params.camera_beam_type == LONG_BEAM ? weights_c.ray_sample_for_pdf_inv : weights_c.ray_sample_for_pdfs_ratio;

    // scaling factor: 1 / (n_v * e_{v,s}) = 1 / (n_v * e_{v,t})
    float val = 1.f / (params.mis_factor_pb2d * rev_pdf_factor);

    float bpt_s_sub_1 = !weights_l.prev_specular;

    /** Eq 4.52 */
    const float w_light = val * (weights_l.d_shared * bpt_s_sub_1 * params.subpath_count_bpt +
                                 bsdf_dir_pdf_w * weights_l.d_pde / weights_l.ray_sample_rev_pdf_inv);

    /** Eq. 4.17 */
    const float w_local =
        // evaluate on t - 1 == s - 1
        1.f + // val / val
        val * (params.mis_factor_bb1d * sinTheta * for_pdf_factor * rev_pdf_factor +
               params.mis_factor_bp2d * for_pdf_factor + params.mis_factor_pp3d);

    float bpt_t_sub_1 = !weights_c.prev_specular;

    /** Eq 4.53 */
    const float w_camera = val * (weights_c.d_shared * bpt_t_sub_1 * params.subpath_count_bpt +
                                  bsdf_rev_pdf_w * weights_c.d_pde / weights_c.ray_sample_rev_pdf_inv);

    /** Eq 4.18 */
    const float mis_weight = 1.f / (w_light + w_local + w_camera);

    // weight and accumulate contribution
    p->contrib += mis_weight * unweighted_result;
}

__forceinline__ __device__ void launch_pb2d(float3 origin, float3 direction, float t_min, float t_max, ParamsPB2D* p)
{
    unsigned int u0, u1;
    pack_pointer(p, u0, u1);
#ifdef DBG_CORE_UTIL
    // this trace call does not make recursive trace calls
    params.trace_count[optixGetLaunchIndex().x]++;
#endif
    optixTrace(params.handle_points,
               origin,
               direction,
               t_min,
               t_max,
               0.f,
               GEOM_MASK_POINT_MEDIUM,
               OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               RAY_TYPE2_PB2D,
               RAY_TYPE2_COUNT,
               RAY_TYPE_COUNT + RAY_TYPE2_PB2D,
               u0,
               u1);
}

__forceinline__ __device__ float3 eval_pb2d_segments(const Ray& query_ray,
                                                     const VolSegmentArray& volume_segments,
                                                     const unsigned int ray_sampling_flags,
                                                     unsigned int camera_path_length,
                                                     const StateWeights& camera_weights)
{
    float3 result = make_float3(0.f);
    float3 attenuation = make_float3(1.f);
    float ray_sample_for_pdf = 1.f;
    float ray_sample_rev_pdf = 1.f;

    // accumulate for each segment
    for (unsigned int i = 0; i < volume_segments.size(); i++) {
        // get segment medium
        const Medium* medium = &params.scene.media[volume_segments[i].med_id];

        // accumulate
        float3 segment_result = make_float3(0.f);
        if (medium->has_scattering) {
            ParamsPB2D p;
            p.contrib = make_float3(0.f);
            p.query_ray = &query_ray;
            p.dist_min = volume_segments[i].dist_min;
            p.dist_max = volume_segments[i].dist_max;
            p.med = medium;
            p.ray_sample_for_pdf = ray_sample_for_pdf;
            p.ray_sample_rev_pdf = ray_sample_rev_pdf;
            p.ray_sampling_flags = END_IN_MEDIUM;
            if (i == 0) p.ray_sampling_flags |= ray_sampling_flags;
            p.camera_path_length = camera_path_length;
            p.camera_weights = &camera_weights;

            launch_pb2d(
                query_ray.origin, query_ray.direction, volume_segments[i].dist_min, volume_segments[i].dist_max, &p);
            segment_result = p.contrib;
        }

        // Add to total result.
        result += attenuation * segment_result;

        // Update attenuation.
        if (params.camera_beam_type == SHORT_BEAM) {
            // Short beams - no attenuation
            attenuation *= volume_segments[i].attenuation / volume_segments[i].ray_sample_pdf_for;
        } else {
            attenuation *= volume_segments[i].attenuation;
        }
        if (!is_positive(attenuation)) return result;

        // Update PDFs.
        ray_sample_for_pdf *= volume_segments[i].ray_sample_pdf_for;
        ray_sample_rev_pdf *= volume_segments[i].ray_sample_pdf_rev;
    }

    return result;
}

__forceinline__ __device__ float3 eval_pb2d_segments(const Ray& query_ray,
                                                     const VolLiteSegmentArray& lite_volume_segments,
                                                     const unsigned int ray_sampling_flags,
                                                     unsigned int camera_path_length,
                                                     const StateWeights& camera_weights)
{
    float3 result = make_float3(0.f);
    float3 attenuation = make_float3(1.f);
    float ray_sample_for_pdf = 1.f;
    float ray_sample_rev_pdf = 1.f;

    // accumulate for each segment
    for (unsigned int i = 0; i < lite_volume_segments.size(); i++) {
        // get segment medium
        const Medium* medium = &params.scene.media[lite_volume_segments[i].med_id];

        // accumulate
        float3 segment_result = make_float3(0.f);
        if (medium->has_scattering) {
            ParamsPB2D p;
            p.contrib = make_float3(0.f);
            p.query_ray = &query_ray;
            p.dist_min = lite_volume_segments[i].dist_min;
            p.dist_max = lite_volume_segments[i].dist_max;
            p.med = medium;
            p.ray_sample_for_pdf = ray_sample_for_pdf;
            p.ray_sample_rev_pdf = ray_sample_rev_pdf;
            p.ray_sampling_flags = END_IN_MEDIUM;
            if (i == 0) p.ray_sampling_flags |= ray_sampling_flags;
            p.camera_path_length = camera_path_length;
            p.camera_weights = &camera_weights;

            launch_pb2d(query_ray.origin,
                        query_ray.direction,
                        lite_volume_segments[i].dist_min,
                        lite_volume_segments[i].dist_max,
                        &p);
            segment_result = p.contrib;
        }

        // Add to total result.
        result += attenuation * segment_result;

        // Update attenuation.
        attenuation *= eval_attenuation(medium, lite_volume_segments[i].dist_min, lite_volume_segments[i].dist_max);
        if (!is_positive(attenuation)) return result;

        // Update PDFs.
        float segment_ray_sample_rev_pdf;
        float segment_ray_sample_for_pdf = ray_sample_pdf(medium,
                                                          lite_volume_segments[i].dist_min,
                                                          lite_volume_segments[i].dist_max,
                                                          i == 0 ? ray_sampling_flags : 0,
                                                          &segment_ray_sample_rev_pdf);
        ray_sample_for_pdf *= segment_ray_sample_for_pdf;
        ray_sample_rev_pdf *= segment_ray_sample_rev_pdf;
    }

    return result;
}

#endif // KERNEL_MAP_FUNCTS_PB2D_CUH
