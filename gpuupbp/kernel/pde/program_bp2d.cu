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
#include "functs_bp2d.cuh"

#include <optix.h>

extern "C" __global__ void __intersection__bp2d()
{
    const unsigned int prim_idx = optixGetInstanceIndex();
    const LightBeam* light_beam = &params.light_beams[prim_idx];

    const float3 ro = light_beam->ray.origin;
    const float3 rd = light_beam->ray.direction;
    const float3 ce = optixGetWorldRayOrigin();
    const float rt = light_beam->beam_length;

    if (!point_beam(ro, rd, rt, ce, params.radius_bp2d_2)) return;

    ParamsBP2D* prd = get_inst<ParamsBP2D>();
    eval_bp2d_contrib(prd, light_beam);
}
