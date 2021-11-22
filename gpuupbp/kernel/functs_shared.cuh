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

#ifndef KERNEL_FUNCTS_SHARED_CUH
#define KERNEL_FUNCTS_SHARED_CUH

#include "../shared/light_vertex.h"
#include "bsdf.cuh"
#include "defs.cuh"
#include "rng.cuh"

/// Samples a scattering direction camera/light sample according to BSDF.
/// Returns false for termination.
__forceinline__ __device__ gpuupbp_status sample_scattering(
    bool& cont, const BSDF& bsdf, const float3& hitpoint, const Intersection& isect, SubpathState& state, RNGState& rng)
{
    // aoState.weights now stores i-1, meaning we have incremented i
    const float4 rnd = get_rnd4(rng);

    // sample scattering function
    // x,y for direction, z for component. no rescaling happens
    float3 rnd3 = make_float3(rnd.x, rnd.y, rnd.z);
    float bsdf_for_pdf_w, cos_theta_out, sin_theta;
    unsigned int sampled_event;
    float3 bsdfFactor =
        sample(bsdf, rnd3, state.direction, bsdf_for_pdf_w, cos_theta_out, params.scene, &sampled_event, &sin_theta);

    if (is_black_or_negative(bsdfFactor)) {
        cont = false;
        return RET_SUCCESS;
    }

    GPU_ASSERT(bsdf_for_pdf_w > 0.f);

    bool specular = (sampled_event & BSDF::SPECULAR) != 0;

    // if we sampled specular event, then the reverse probability
    // cannot be evaluated, but we know it is exactly the same as
    // forward probability, so just set it. If non-specular event happened,
    // we evaluate the PDF
    float bsdf_rev_pdf_w = bsdf_for_pdf_w;
    if (!specular) {
        bsdf_rev_pdf_w = pdf(bsdf, state.direction, params.scene, BSDF::REVERSE);
        GPU_ASSERT(bsdf_rev_pdf_w > 0.f);
    }

    // Russian roulette
    const float prob = rnd.w;
    const float cont_prob = bsdf.continuation_prob;
    if (cont_prob == 0 || (cont_prob < 1.f && prob > cont_prob)) {
        cont = false;
        return RET_SUCCESS;
    }

    bsdf_for_pdf_w *= cont_prob;
    bsdf_rev_pdf_w *= cont_prob;

    // ind^bpt_{i-1} considers a subpath where the last vertex is i - 2
    const float bpt_i_sub_1 =
        // whether i - 2 was specular
        !state.weights.weights.prev_specular &&
        // whether i - 1 was specular
        !specular;

    if (specular) {
        // we can simplify the calculation below based on these two facts
        GPU_ASSERT(bsdf_for_pdf_w == bsdf_rev_pdf_w);
        GPU_ASSERT(bpt_i_sub_1 == 0.f);
        /** Eq 4.48 (partly) */
        // dBPTa still needs multiplication by f_i
        // note: actually needs division by params.subpath_count_bpt, but this
        // is only different from 1 when BPT is disabled (in which case it is
        // zero), and dBPT is not used
        state.weights.d_bpt_a = cos_theta_out / bsdf_for_pdf_w;

        state.weights.d_bpt_b =
            cos_theta_out * state.weights.weights.d_bpt / state.weights.weights.ray_sample_rev_pdf_inv;

        /** Eq 4.49 (partly) */
        // dPDEa still needs multiplication by f_i
        state.weights.d_pde_a = cos_theta_out / bsdf_for_pdf_w;

        state.weights.d_pde_b =
            cos_theta_out * state.weights.weights.d_pde / state.weights.weights.ray_sample_rev_pdf_inv;
    } else {
        /** Eq 4.48 (partly) */
        // dBPTa still needs multiplication by f_i
        // note: actually needs division by params.subpath_count_bpt, but this
        // is only different from one when BPT is disabled (in which case it is
        // zero), and dBPT is not used
        state.weights.d_bpt_a = cos_theta_out / bsdf_for_pdf_w;

        state.weights.d_bpt_b = (cos_theta_out / bsdf_for_pdf_w) * (state.weights.weights.d_shared * bpt_i_sub_1 +
                                                                    bsdf_rev_pdf_w * state.weights.weights.d_bpt /
                                                                        state.weights.weights.ray_sample_rev_pdf_inv);

        /** Eq 4.49 (partly) */
        // dPDEa still needs multiplication by f_i
        state.weights.d_pde_a = cos_theta_out / bsdf_for_pdf_w;

        state.weights.d_pde_b =
            (cos_theta_out / bsdf_for_pdf_w) *
            (state.weights.weights.d_shared * bpt_i_sub_1 * params.subpath_count_bpt +
             bsdf_rev_pdf_w * state.weights.weights.d_pde / state.weights.weights.ray_sample_rev_pdf_inv);
    }

    /** Eq 4.47 (partly) */
    state.weights.weights.d_shared = 1.f / bsdf_for_pdf_w;

    state.is_specular_path = state.is_specular_path && specular;

    state.origin = hitpoint;
    state.throughput *= bsdfFactor * (cos_theta_out / bsdf_for_pdf_w);

    state.weights.weights.prev_in_medium = is_in_medium(bsdf);
    state.weights.weights.prev_specular = specular;

    state.weights.last_sin_theta = sin_theta;

    // aoState.weights now stores i

    // switch medium on refraction
    if ((sampled_event & BSDF::REFRACT) != 0) {
        unsigned int ret = update_boundary_stack_on_refract(params.scene, isect, state.stack);
        if (ret != RET_SUCCESS) return ERR_BOUNDARY_STACK;
    }

    cont = true;

    return RET_SUCCESS;
}

/// Accumulates PDFs from all the given segments
/// (just multiplies them together).
__forceinline__ __device__ float accumulate_for_pdf(const VolSegmentArray& segments)
{
    float pdf_for = 1.f;
    for (unsigned int i = 0; i < segments.size(); i++) {
        pdf_for *= segments[i].ray_sample_pdf_for;
    }
    return pdf_for;
}

/// Accumulates reverse PDFs from all the given segments
/// (just multiplies them together).
__forceinline__ __device__ float accumulate_rev_pdf(const VolSegmentArray& segments)
{
    float pdf_rev = 1.f;
    for (unsigned int i = 0; i < segments.size(); i++) {
        pdf_rev *= segments[i].ray_sample_pdf_rev;
    }
    return pdf_rev;
}

/// Accumulates attenuation from all the given segments
/// (just multiplies them together).
__forceinline__ __device__ float3 accumulate_attenuation_without_pdf(const VolSegmentArray& segments)
{
    float3 att = make_float3(1.f);
    for (unsigned int i = 0; i < segments.size(); i++) {
        att *= segments[i].attenuation;
    }
    return att;
}

#endif // KERNEL_FUNCTS_SHARED_CUH
