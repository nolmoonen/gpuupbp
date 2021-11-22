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

#ifndef KERNEL_PHASE_FUNCTION_CUH
#define KERNEL_PHASE_FUNCTION_CUH

#include "../shared/frame.h"
#include "medium.cuh"
#include "sample.cuh"

struct PhaseFunction {
    __forceinline__ __device__ static float3 evaluate(
        /// Points away from the scattering location.
        const float3& world_dir_fix,
        /// Points away from the scattering location.
        const float3& world_dir_gen,
        const float mean_cosine,
        float* for_pdf_w = NULL,
        float* rev_pdf_w = NULL,
        float* sin_theta = NULL)
    {
        float pdf_ = pdf(world_dir_fix, world_dir_gen, mean_cosine, sin_theta);

        if (for_pdf_w) *for_pdf_w = pdf_;
        if (rev_pdf_w) *rev_pdf_w = pdf_;

        return make_float3(pdf_);
    }

    __forceinline__ __device__ static float pdf(
        /// Points away from the scattering location.
        const float3& world_dir_fix,
        /// Points away from the scattering location.
        const float3& world_dir_gen,
        const float mean_cosine,
        float* sin_theta = NULL)
    {
        GPU_ASSERT(mean_cosine >= -1.f && mean_cosine <= 1.f);

        if (is_isotropic(mean_cosine)) {
            if (sin_theta) {
                const float cos_theta = -dot(world_dir_gen, world_dir_fix);
                *sin_theta = sqrtf(fmaxf(0.f, 1.f - cos_theta * cos_theta));
            }
            return uniform_sphere_pdf_w();
        } else {
            const float cos_theta = -dot(world_dir_gen, world_dir_fix);
            const float square_mean_cosine = mean_cosine * mean_cosine;
            const float d = 1.f + square_mean_cosine - (mean_cosine + mean_cosine) * cos_theta;
            if (sin_theta) {
                *sin_theta = sqrtf(fmaxf(0.f, 1.f - cos_theta * cos_theta));
            }

            if (d > 0.f) {
                return uniform_sphere_pdf_w() * (1.f - square_mean_cosine) / (d * sqrtf(d));
            } else {
                return 0.f;
            }
        }
    }

    __forceinline__ __device__ static float3 sample(
        /// Points away from the scattering location.
        const float3& world_dir_fix,
        const float mean_cosine,
        const float3& rnd,
        /// Points away from the scattering location.
        float3& world_dir_gen,
        float& pdf_w,
        float* sin_theta = NULL)
    {
        Frame frame;
        frame.set_from_z(-world_dir_fix);
        return sample(world_dir_fix, mean_cosine, rnd, frame, world_dir_gen, pdf_w, sin_theta);
    }

    __forceinline__ __device__ static float3 sample(
        /// Points away from the scattering location.
        const float3& world_dir_fix,
        const float mean_cosine,
        const float3& rnd,
        const Frame& frame,
        /// Points away from the scattering location.
        float3& world_dir_gen,
        float& pdf_w,
        float* sin_theta = NULL)
    {
        GPU_ASSERT(mean_cosine >= -1 && mean_cosine <= 1);

        if (is_isotropic(mean_cosine)) {
            world_dir_gen = sample_uniform_sphere_w(make_float2(rnd.x, rnd.y), &pdf_w);
            if (sin_theta) {
                const float cos_theta = -dot(world_dir_gen, world_dir_fix);
                *sin_theta = sqrtf(fmaxf(0.f, 1.f - cos_theta * cos_theta));
            }
        } else {
            const float square_mean_cosine = mean_cosine * mean_cosine;
            const float two_cosine = mean_cosine + mean_cosine;
            const float sqrtt = (1.f - square_mean_cosine) / (1.f - mean_cosine + two_cosine * rnd.x);
            const float cos_theta = (1.f + square_mean_cosine - sqrtt * sqrtt) / two_cosine;
            const float sin_theta_ = sqrtf(fmaxf(0.f, 1.f - cos_theta * cos_theta));
            const float phi = 2.f * PI_F * rnd.y;
            const float sin_phi = sinf(phi);
            const float cos_phi = cosf(phi);
            const float d = 1.f + square_mean_cosine - two_cosine * cos_theta;

            world_dir_gen = frame.to_world(make_float3(cos_phi * sin_theta_, sin_phi * sin_theta_, cos_theta));
            pdf_w = d > 0.f ? (uniform_sphere_pdf_w() * (1.f - square_mean_cosine) / (d * sqrtf(d))) : 0.f;
            if (sin_theta) *sin_theta = sin_theta_;
        }

        return make_float3(pdf_w);
    }
};

#endif // KERNEL_PHASE_FUNCTION_CUH
