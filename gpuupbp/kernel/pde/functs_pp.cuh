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

#ifndef KERNEL_MAP_FUNCTS_PP_CUH
#define KERNEL_MAP_FUNCTS_PP_CUH

#include "../../shared/light_vertex.h"
#include "../bsdf.cuh"
#include "../defs.cuh"
#include "../params_def.cuh"
#include "../path_weight.cuh"
#include "../scene.cuh"
#include "../types.cuh"

struct ParamsPP {
    float3 contrib;
    SubpathState* camera_state;
    BSDF* camera_bsdf;
};

__forceinline__ __device__ void eval_pp(ParamsPP* p, const LightVertex* light_vertex)
{
    // use only vertices with same location (on surface/in medium)
    GPU_ASSERT(light_vertex->is_in_medium() == is_in_medium(*p->camera_bsdf));
    bool medium = light_vertex->is_in_medium();

    // reject if full path length below/above min/max path length
    // photon path length equals all segments up to the photon vertex
    // camera path length equals all segments up to the camera vertex
    // we merge photon and camera vertex, no segment is created
    const unsigned int path_length = light_vertex->path_length + p->camera_state->path_length;
    if (path_length > params.max_path_length || path_length < params.min_path_length) {
        return;
    }

    // retrieve light incoming direction in world coordinates
    // (points away from scattering location)
    const float3 photon_direction = world_dir_fix(light_vertex->bsdf);

    // evaluate the scattering function (from camera to light)
    float cos_camera, bsdf_dir_pdf_w, bsdf_rev_pdf_w, sin_theta;
    const float3 bsdf_factor = evaluate(*p->camera_bsdf,
                                        photon_direction,
                                        cos_camera,
                                        params.scene.materials,
                                        &bsdf_dir_pdf_w,
                                        &bsdf_rev_pdf_w,
                                        &sin_theta);

    if (is_black_or_negative(bsdf_factor)) return;

    bsdf_dir_pdf_w *= p->camera_bsdf->continuation_prob;
    GPU_ASSERT(bsdf_dir_pdf_w > 0.f);

    // even though this is PDF from camera BSDF, the continuation probability
    // must come from light BSDF, because that would govern it if light path
    // actually continued
    bsdf_rev_pdf_w *= light_vertex->bsdf.continuation_prob;
    GPU_ASSERT(bsdf_rev_pdf_w > 0.f);

    // MIS weight

    // note: do not worry about specularity here, we know the light vertex is
    // not specular (otherwise it would not be stored), and the camera vertex
    // is not specular (otherwise the merging would not be executed)

    // index s - 1
    const VertexWeights& weights_l = light_vertex->weights;
    // index t - 1
    const VertexWeights& weights_c = p->camera_state->weights.weights;

    float for_pdf_factor =
        params.photon_beam_type == LONG_BEAM ? weights_l.ray_sample_for_pdf_inv : weights_l.ray_sample_for_pdfs_ratio;
    float rev_pdf_factor =
        params.camera_beam_type == LONG_BEAM ? weights_c.ray_sample_for_pdf_inv : weights_c.ray_sample_for_pdfs_ratio;

    // scaling factor: 1 / (n_v * e_{v,s}) = 1 / (n_v * e_{v,t})
    float val;
    if (medium) {
        // we are evaluating PP3D
        val = 1.f / params.mis_factor_pp3d;
    } else {
        // we are evaluating SURF
        val = 1.f / params.mis_factor_surf;
    }

    const float b_s_sub_1 = !weights_l.prev_specular; // s - 2 is specular
    // s - 1 is not delta (otherwise it would not have been stored)

    /** Eq 4.52 */
    const float w_light = val * (weights_l.d_shared * b_s_sub_1 * params.subpath_count_bpt +
                                 bsdf_dir_pdf_w * weights_l.d_pde / weights_l.ray_sample_rev_pdf_inv);

    /** Eq. 4.17 */
    const float w_local =
        // evaluate on t - 1 == s - 1
        1.f + // val / val
        float(medium) * val *
            (params.mis_factor_bb1d * sin_theta * for_pdf_factor * rev_pdf_factor +
             params.mis_factor_bp2d * for_pdf_factor + params.mis_factor_pb2d * rev_pdf_factor);

    const float b_t_sub_1 = !weights_c.prev_specular; // t - 2 is specular
    // t - 1 is not delta (otherwise this function is not called)

    /** Eq 4.53 */
    const float w_camera = val * (weights_c.d_shared * b_t_sub_1 * params.subpath_count_bpt +
                                  bsdf_rev_pdf_w * weights_c.d_pde / weights_c.ray_sample_rev_pdf_inv);

    /** Eq 4.18 */
    const float mis_weight = 1.f / (w_light + w_local + w_camera);

    p->contrib += mis_weight * bsdf_factor * light_vertex->throughput;
}

#endif // KERNEL_MAP_FUNCTS_PP_CUH
