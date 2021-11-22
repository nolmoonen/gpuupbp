// Copyright (C) 2021, Nol Moonen
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

#include "../../shared/launch_params.h"
#include "../../shared/vec_math.h"
#include "../intersection.cuh"
#include "../optix_util.cuh"
#include "functs_bb1d.cuh"

#include <optix.h>

extern "C" __global__ void __intersection__bb1d()
{
    const unsigned int prim_idx = optixGetInstanceIndex();
    const LightBeam* light_beam = &params.light_beams[prim_idx];

    const float3 o1 = optixGetWorldRayOrigin();
    const float3 d1 = optixGetWorldRayDirection();
    const float len_t1 = optixGetRayTmax();
    const float3 o2 = light_beam->ray.origin;
    const float3 d2 = light_beam->ray.direction;
    const float len_t2 = light_beam->beam_length;
    const float r2 = params.radius_bb1d_2;

    if (!beam_beam(o1, d1, len_t1, o2, d2, len_t2, r2)) return;

    ParamsBB1D* prd = get_inst<ParamsBB1D>();
    eval_bb1d(prd, light_beam);
}
