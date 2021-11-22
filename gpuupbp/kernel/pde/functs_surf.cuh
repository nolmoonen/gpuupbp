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

#ifndef KERNEL_MAP_FUNCTS_SURF_CUH
#define KERNEL_MAP_FUNCTS_SURF_CUH

#include "../../kernel/bsdf.cuh"
#include "../../kernel/optix_util.cuh"
#include "../../shared/intersect_defs.h"
#include "functs_pp.cuh"

__forceinline__ __device__ void launch_pp2d(float3 hit_point, ParamsPP* p)
{
    unsigned int u0, u1;
    pack_pointer(p, u0, u1);
#ifdef DBG_CORE_UTIL
    // this trace call does not make recursive trace calls
    params.trace_count[optixGetLaunchIndex().x]++;
#endif
    optixTrace(params.handle_points,
               hit_point,
               make_float3(EPS_LAUNCH),
               0.f,
               EPS_LAUNCH,
               0.f,
               GEOM_MASK_PP2D,
               OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               RAY_TYPE2_PP2D,
               RAY_TYPE2_COUNT,
               RAY_TYPE_COUNT + RAY_TYPE2_PP2D,
               u0,
               u1);
}

__forceinline__ __device__ float3 eval_pp2d(float3 hit_point, SubpathState* camera_state, BSDF* camera_bsdf)
{
    ParamsPP p;
    p.contrib = make_float3(0.f);
    p.camera_state = camera_state;
    p.camera_bsdf = camera_bsdf;

    launch_pp2d(hit_point, &p);
    return p.contrib;
}

#endif // KERNEL_MAP_FUNCTS_SURF_CUH
