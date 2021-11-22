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

#ifndef KERNEL_MED_CUH
#define KERNEL_MED_CUH

#include "../shared/frame.h"
#include "defs.cuh"

__forceinline__ __device__ float3 eval_attenuation(const Medium* med, const float dist)
{
    GPU_ASSERT(dist >= 0.f);
    return make_float3(expf(-med->attenuation_coeff.x * dist),
                       expf(-med->attenuation_coeff.y * dist),
                       expf(-med->attenuation_coeff.z * dist));
}

__forceinline__ __device__ float3 eval_attenuation(const Medium* med, const float dist_min, const float dist_max)
{
    float dist = dist_max - dist_min;
    return eval_attenuation(med, dist);
}

__forceinline__ __device__ float3 eval_emission(Medium* med, const float dist_min, const float dist_max)
{
    float dist = dist_max - dist_min;
    return make_float3(med->emission_coeff.x * dist, med->emission_coeff.y * dist, med->emission_coeff.z * dist);
}

__forceinline__ __device__ float eval_attenuation_in_one_dim(const float attenuation_coef_comp,
                                                             const float distance_along_ray)
{
    return expf(-attenuation_coef_comp * distance_along_ray);
}

/// Samples the medium along the given ray starting at its origin.
/// Returns distance along ray to sampled point in media or distance to
/// boundary if sample fell behind.
/// Random should be in (0,1).
__forceinline__ __device__ float sample_ray(Medium* med,
                                            const float dist_to_boundary,
                                            const float rnd,
                                            float* for_pdf,
                                            const unsigned int ray_sampling_flags,
                                            float* rev_pdf)
{
    GPU_ASSERT(dist_to_boundary >= 0.f);

    if (med->min_positive_attenuation_coeff_comp != 0.f && med->has_scattering) {
        // we can sample along the ray
        float s = -logf(rnd) / med->min_positive_attenuation_coeff_comp;

        if (s < dist_to_boundary) {
            // sample is before the boundary intersection
            float att = eval_attenuation_in_one_dim(med->min_positive_attenuation_coeff_comp, s);

            if (for_pdf) {
                *for_pdf = med->min_positive_attenuation_coeff_comp * att;
            }

            if (rev_pdf) {
                if (ray_sampling_flags & ORIGIN_IN_MEDIUM) {
                    *rev_pdf = *for_pdf;
                } else {
                    *rev_pdf = att;
                }
            }

            return s;
        } else {
            // sample is behind the boundary intersection
            float att = eval_attenuation_in_one_dim(med->min_positive_attenuation_coeff_comp, dist_to_boundary);

            if (for_pdf) *for_pdf = att;

            if (rev_pdf) {
                if (ray_sampling_flags & ORIGIN_IN_MEDIUM) {
                    *rev_pdf = med->min_positive_attenuation_coeff_comp * att;
                } else {
                    *rev_pdf = att;
                }
            }

            return dist_to_boundary;
        }
    } else {
        // we cannot sample along the ray
        if (for_pdf) *for_pdf = 1.f;
        if (rev_pdf) *rev_pdf = 1.f;
        return dist_to_boundary;
    }
}

/// Get PDF (and optionally reverse PDF) of sampling in the medium along the
/// given ray. Sampling starts at the given min distance and ends at the
/// max distance. If end is said to be inside the medium, PDF for sampling in
/// medium is returned, otherwise PDF for sampling behind the medium is
/// returned.
__forceinline__ __device__ float ray_sample_pdf(const Medium* med,
                                                const float dist_min,
                                                const float dist_max,
                                                const unsigned int ray_sampling_flags,
                                                float* rev_pdf)
{
    float for_pdf = 1.f;
    if (rev_pdf) *rev_pdf = 1.f;

    if (med->min_positive_attenuation_coeff_comp != 0.f && med->has_scattering) {
        // we can sample along the ray
        float att =
            fmaxf(eval_attenuation_in_one_dim(med->min_positive_attenuation_coeff_comp, dist_max - dist_min), 1e-35f);
        // prevent returning zero
        float minatt = fmaxf(med->min_positive_attenuation_coeff_comp * att, 1e-35f);

        if (ray_sampling_flags & END_IN_MEDIUM) {
            for_pdf = minatt;
        } else {
            for_pdf = att;
        }

        if (rev_pdf) {
            if (ray_sampling_flags & ORIGIN_IN_MEDIUM) {
                *rev_pdf = minatt;
            } else {
                *rev_pdf = att;
            }
        }
    }

    return for_pdf;
}

__forceinline__ __device__ static int is_isotropic(const float mean_cosine) { return fabsf(mean_cosine) < 1e-3f; }

__forceinline__ __device__ float get_min_positive_attenuation_coeff_comp(const Medium* med, float3 that)
{
    switch (med->min_positive_attenuation_coeff_comp_idx) {
    case 0:
        return that.x;
    case 1:
        return that.y;
    case 2:
        return that.z;
    default:
        GPU_ASSERT(false);
        return -1.f;
    }
}

#endif // KERNEL_MED_CUH
