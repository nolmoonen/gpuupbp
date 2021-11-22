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

#ifndef KERNEL_DISTRIBUTION_CUH
#define KERNEL_DISTRIBUTION_CUH

#include "../shared/env_map.h"
#include "../shared/frame.h"
#include "defs.cuh"

/** Functionality of this file comes from https://github.com/mmp/pbrt-v3/. */

template <typename T, typename U, typename V>
__forceinline__ __device__ static T clamp(T val, U low, V high)
{
    if (val < low) {
        return low;
    } else if (val > high) {
        return high;
    } else {
        return val;
    }
}

template <typename predicate>
__forceinline__ __device__ static int find_interval(int size, const predicate& pred)
{
    int first = 0, len = size;
    while (len > 0) {
        int half = len >> 1, middle = first + half;
        // Bisect range based on value of _pred_ at _middle_
        if (pred(middle)) {
            first = middle + 1;
            len -= half + 1;
        } else
            len = half;
    }
    return clamp(first - 1, 0, size - 2);
}

/// Takes pointers to func and cdf, rather than these being struct members.
__forceinline__ __device__ float
sample_continuous(const Distribution1D& dist, float u, float* pdf, const float* func, const float* cdf, int* off = NULL)
{
    // Find surrounding CDF segments and _offset_
    int offset = find_interval(dist.count + 1, [&](int index) { return cdf[index] <= u; });
    if (off) *off = offset;
    // Compute offset along CDF segment
    float du = u - cdf[offset];
    if ((cdf[offset + 1] - cdf[offset]) > 0) {
        GPU_ASSERT(cdf[offset + 1] > cdf[offset]);
        du /= (cdf[offset + 1] - cdf[offset]);
    }
    GPU_ASSERT(!isnan(du));

    // Compute PDF for sampled offset
    if (pdf) *pdf = (dist.func_int > 0) ? func[offset] / dist.func_int : 0;

    // Return $x\in{}[0,1)$ corresponding to sample
    return (offset + du) / dist.count;
}

/// Obtain the pointer to the func array associated with 1d distribution {idx}.
__forceinline__ __device__ float* get_func(const Distribution2D& dist, unsigned int idx)
{
    return &dist.data[(2u * dist.nu + 1u) * idx];
}

/// Obtain the pointer to the cdf array associated with 1d distribution {idx}.
__forceinline__ __device__ float* get_cdf(const Distribution2D& dist, unsigned int idx)
{
    return get_func(dist, idx) + dist.nu;
}

__forceinline__ __device__ void
sample_continuous(const Distribution2D& dist, float u0, float u1, float uv[2], float* pdf)
{
    float pdfs[2];
    int v;
    // amount of floats needed for one 1d distribution
    uv[1] = sample_continuous(dist.marginal,
                              u1,
                              &pdfs[1],
                              // marginal has index nv in the data array
                              get_func(dist, dist.nv),
                              get_cdf(dist, dist.nv),
                              &v);
    uv[0] = sample_continuous(dist.conditional_v[v], u0, &pdfs[0], get_func(dist, v), get_cdf(dist, v));
    *pdf = pdfs[0] * pdfs[1];
}

__forceinline__ __device__ float pdf(const Distribution2D& dist, float u, float v)
{
    int iu = clamp((int)(u * dist.conditional_v[0].count), 0, dist.conditional_v[0].count - 1);
    int iv = clamp((int)(v * dist.marginal.count), 0, dist.marginal.count - 1);
    if (dist.conditional_v[iv].func_int * dist.marginal.func_int == 0.f) {
        return 0.f;
    }

    // marginal has index nv in the data array
    return (get_func(dist, iv)[iu] * get_func(dist, dist.nv)[iv]) /
           (dist.conditional_v[iv].func_int * dist.marginal.func_int);
}

#endif // KERNEL_DISTRIBUTION_CUH
