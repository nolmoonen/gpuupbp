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

#ifndef KERNEL_ENV_MAP_CUH
#define KERNEL_ENV_MAP_CUH

#include "../shared/frame.h"
#include "../shared/shared_defs.h"
#include "distribution.cuh"

/// Returns direction on unit sphere such that its longitude equals 2*PI*u and
/// its latitude equals PI*v.
__forceinline__ __device__ float3 lat_long_to_dir(const EnvMap& map, float u, float v)
{
    float phi = u * 2.f * PI_F;
    float theta = v * PI_F;

    float sinTheta = sin(theta);

    return make_float3(-sinTheta * cos(phi), sinTheta * sin(phi), cos(theta));
}

/// Returns vector [u,v] such that the longitude of the given direction equals
/// 2*PI*u and its latitude equals PI*v. The direction must be non-zero and
/// normalized.
__forceinline__ __device__ float2 dir_to_lat_long(const EnvMap& map, const float3& direction)
{
    float phi = (direction.x != 0 || direction.y != 0) ? atan2f(direction.y, direction.x) : 0;
    float theta = acosf(direction.z);

    float u = clampf(.5f - phi * .5f * INV_PI_F, 0.f, 1.f);
    float v = clampf(theta * INV_PI_F, 0.f, 1.f);

    return make_float2(u, v);
}

/// Returns radiance for the given lat long coordinates.
/// Does bilinear filtering.
__forceinline__ __device__ float3 lookup_radiance(const EnvMap& map, float u, float v, const Image& image)
{
    int width = image.width;
    int height = image.height;

    float xf = u * width;
    float yf = v * height;

    int xi1 = clamp((int)xf, 0, width - 1);
    int yi1 = clamp((int)yf, 0, height - 1);

    int xi2 = xi1 == width - 1 ? xi1 : xi1 + 1;
    int yi2 = yi1 == height - 1 ? yi1 : yi1 + 1;

    float tx = xf - (float)xi1;
    float ty = yf - (float)yi1;

    return (1 - ty) * ((1 - tx) * image.element_at(xi1, yi1) + tx * image.element_at(xi2, yi1)) +
           ty * ((1 - tx) * image.element_at(xi1, yi2) + tx * image.element_at(xi2, yi2));
}

/// Returns sine of latitude for a midpoint of a pixel in a map of the
/// given height corresponding to v. Never returns zero.
__forceinline__ __device__ float sin_theta(const EnvMap& map, const float v, const float height)
{
    float result;

    if (v < 1.f) {
        result = sinf(PI_F * (float)((int)(v * height) + .5f) / (float)height);
    } else {
        result = sinf(PI_F * (float)((height - 1.f) + .5f) / (float)height);
    }

    GPU_ASSERT(result > 0.f && result <= 1.f);

    return result;
}

/// Samples direction on unit sphere proportionally to the luminance of the map.
/// Returns its PDF and optionally radiance.
__forceinline__ __device__ float3 sample(const EnvMap& map,
                                         const float2& samples,
                                         float& pdf_w,
                                         float3* radiance = NULL)
{
    float uv[2];
    float pdf;
    sample_continuous(map.distribution, samples.x, samples.y, uv, &pdf);

    GPU_ASSERT(pdf > 0);

    pdf_w = map.norm * pdf / sin_theta(map, uv[1], map.img.height);

    float3 direction = lat_long_to_dir(map, uv[0], uv[1]);

    if (radiance) *radiance = lookup_radiance(map, uv[0], uv[1], map.img);

    return direction;
}

/// Gets radiance stored for the given direction and optionally its PDF.
/// The direction must be non-zero but not necessarily normalized.
__forceinline__ __device__ float3 lookup(const EnvMap& map, const float3& direction, float* pdf_w = NULL)
{
    GPU_ASSERT(direction.x != 0 || direction.y != 0 || direction.z != 0);
    float3 norm_dir = normalize(direction);
    float2 uv = dir_to_lat_long(map, norm_dir);
    float3 radiance = lookup_radiance(map, uv.x, uv.y, map.img);

    if (pdf_w) {
        float2 uv = dir_to_lat_long(map, norm_dir);
        *pdf_w = map.norm * pdf(map.distribution, uv.x, uv.y) / sin_theta(map, uv.y, map.img.height);
        if (*pdf_w == 0.f) radiance = make_float3(0.f);
    }
    return radiance;
}

#endif // KERNEL_ENV_MAP_CUH
