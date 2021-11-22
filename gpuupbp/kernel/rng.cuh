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

#ifndef KERNEL_RNG_CUH
#define KERNEL_RNG_CUH

#include "curand_kernel.h"
#include "params_def.cuh"

#include <cuda_runtime.h>

/// Random number generator state.
struct RNGState {
    curandStatePhilox4_32_10_t state;
};

/// Initializes the random number generator state.
/// Must be called before calls to {get_rnd} and {get_rnd4}.
__forceinline__ __device__ void rng_init(unsigned int iteration, unsigned int path_idx, RNGState& state)
{
    curand_init(iteration, path_idx, params.rng_offset, &state.state);
}

/// Returns a uniformly distributed pseudorandom number in [0.f, 1.f).
__forceinline__ __device__ float get_rnd(RNGState& state)
{
    // returns (0.f, 1.f]
    float rnd = curand_uniform(&state.state);
    return rnd == 1.f ? 0.f : rnd;
}

/// Returns a uniformly distributed pseudorandom number in [0.f, 1.f).
__forceinline__ __device__ float4 get_rnd4(RNGState& state)
{
    // returns (0.f, 1.f]
    float4 rnd = curand_uniform4(&state.state);
    rnd.x = rnd.x == 1.f ? 0.f : rnd.x;
    rnd.y = rnd.y == 1.f ? 0.f : rnd.y;
    rnd.z = rnd.z == 1.f ? 0.f : rnd.z;
    rnd.w = rnd.w == 1.f ? 0.f : rnd.w;
    return rnd;
}

#endif // KERNEL_RNG_CUH
