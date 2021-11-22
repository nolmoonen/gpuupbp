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

#ifndef HOST_ENV_MAP_HPP
#define HOST_ENV_MAP_HPP

#include "../misc/tinyexr_wrapper.hpp"
#include "../shared/frame.h"
#include "../shared/shared_defs.h"
#include "../shared/vec_math.h"

#include <cstring>
#include <iostream>

inline float3& element_at(const Image& image, int x, int y) { return image.data[image.width * y + x]; }

// todo is it necessary to allocate func? since this is never modified.
/// Takes pointers to func and cdf, rather than these being struct members.
inline void init_distribution_1d(Distribution1D& dist,
                                 const float* f,
                                 int n,
                                 /// memory allocated of size n
                                 float* func,
                                 /// memory allocated of size n + 1
                                 float* cdf)
{
    dist.count = n;
    memcpy(func, f, n * sizeof(float));

    // Compute integral of step function at $x_i$
    cdf[0] = 0.;
    for (int i = 1; i < dist.count + 1; ++i) {
        cdf[i] = cdf[i - 1] + func[i - 1] / n;
    }

    // Transform step function integral into CDF
    dist.func_int = cdf[dist.count];
    if (dist.func_int == 0.f) {
        for (int i = 1; i < n + 1; ++i) {
            cdf[i] = float(i) / float(n);
        }
    } else {
        for (int i = 1; i < n + 1; ++i) {
            cdf[i] /= dist.func_int;
        }
    }
}

inline void init_distibution_2d(Distribution2D& dist, const float* func, int nu, int nv)
{
    dist.nu = nu;
    dist.nv = nv;
    // amount of floats needed for one 1d distribution
    uint32_t size = (2 * nu + 1);
    // last one is for marginal
    dist.data = static_cast<float*>(malloc(sizeof(float) * (nv + 1) * size));
    dist.conditional_v = static_cast<Distribution1D*>(malloc(nv * sizeof(Distribution1D)));
    for (int v = 0; v < nv; ++v) {
        // Compute conditional sampling distribution for $\tilde{v}$
        init_distribution_1d(dist.conditional_v[v], &func[v * nu], nu, &dist.data[size * v], &dist.data[size * v + nu]);
    }

    // Compute marginal sampling distribution $p[\tilde{v}]$
    auto* marginal_func = static_cast<float*>(malloc(nv * sizeof(float)));
    for (int v = 0; v < nv; ++v) {
        marginal_func[v] = dist.conditional_v[v].func_int;
    }
    init_distribution_1d(dist.marginal, &marginal_func[0], nv, &dist.data[size * nv], &dist.data[size * nv + nu]);
    free(marginal_func);
}

/// Computes sRGB luminance.
inline float luminance(const float3& rgb) { return 0.212671f * rgb.x + 0.715160f * rgb.y + 0.072169f * rgb.z; }

/// Converts luminance of the given environment map to 2D distribution with
/// latitude-longitude mapping.
inline Distribution2D convert_image_to_pdf(const Image& image)
{
    int height = image.height;
    int width = height + height; // height maps to PI, width maps to 2PI

    float* data = new float[width * height];

    for (int r = 0; r < height; ++r) {
        float v = (float)(r + 0.5f) / (float)height;
        float sin_theta = sinf(PI_F * v);
        int col_offset = r * width;

        for (int c = 0; c < width; ++c) {
            data[c + col_offset] = sin_theta * luminance(element_at(image, c, r));
        }
    }

    Distribution2D dist{};
    init_distibution_2d(dist, data, width, height);

    return dist;
}

inline void init_image(Image& image, uint32_t width, uint32_t height)
{
    image.width = width;
    image.height = height;
    image.data = static_cast<float3*>(malloc(sizeof(float3) * width * height));
}

/// Expects absolute path of an OpenEXR file with an environment map with
/// latitude-longitude mapping.
inline void init_env_map(EnvMap& env_map, const std::string filename, float rotate, float scale)
{
    env_map.norm = 0.5f * INV_PI_F * INV_PI_F;

    try {
        env_map.img = tinyexr_wrapper::load_image(filename.c_str(), rotate, scale);
        env_map.distribution = convert_image_to_pdf(env_map.img);
        std::cout << "Loading : " << filename << std::endl;
    } catch (...) {
        std::cerr << "Error: environment map loading failed" << std::endl;
        exit(2);
    }
}

#endif // HOST_ENV_MAP_HPP
