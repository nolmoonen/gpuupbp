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

#ifndef KERNEL_PATH_WEIGHT_CUH
#define KERNEL_PATH_WEIGHT_CUH

#include "params_def.cuh"

/// Evaluates f_i, which considers a subpath of which the last vertex
/// has index i-1. Note that this factor is not symmetric for light and camera,
/// hence we use x-notation here.
__forceinline__ __device__ float get_pde_factor(bool i_sub_1_in_medium,             /// i-1 is in medium
                                                bool i_sub_1_specular,              /// i-1 is specular
                                                float i_sub_1_ray_sample_for_inv,   /// 1           / pr_{T,i-1}(x)
                                                float i_sub_1_ray_sample_for_ratio, /// prob_r{i-1} / pr_{T,i-1}(x)
                                                float i_sub_1_ray_sample_rev_inv,   /// 1           / pl_{T,i-1}(x)
                                                float i_sub_1_ray_sample_rev_ratio, /// prob_l{i-1} / pl_{T,i-1}(x)
                                                float sin_theta)                    /// sine of angle i-2 -> i-1 -> i
{
    float for_pdf = params.photon_beam_type == LONG_BEAM ? i_sub_1_ray_sample_for_inv : i_sub_1_ray_sample_for_ratio;
    float rev_pdf = params.camera_beam_type == LONG_BEAM ? i_sub_1_ray_sample_rev_inv : i_sub_1_ray_sample_rev_ratio;

    /** Eq 4.23 */
    return float(!i_sub_1_in_medium && !i_sub_1_specular) * (params.mis_factor_surf) +
           float(i_sub_1_in_medium) *
               (params.mis_factor_pp3d + params.mis_factor_pb2d * rev_pdf + params.mis_factor_bp2d * for_pdf +
                params.mis_factor_bb1d * sin_theta * for_pdf * rev_pdf);
}

#endif // KERNEL_PATH_WEIGHT_CUH
