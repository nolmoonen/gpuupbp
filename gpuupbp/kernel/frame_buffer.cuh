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

#ifndef KERNEL_FRAME_BUFFER_CUH
#define KERNEL_FRAME_BUFFER_CUH

#include "../shared/framebuffer.h"

#include <cuda_runtime.h>

/// Adds the given color to pixel containing the given position.
__forceinline__ __device__ void add_color(const Framebuffer& frame_buffer, const float2& sample, const float3& color)
{
    if (sample.x < 0.f || sample.x >= frame_buffer.resolution.x) return;
    if (sample.y < 0.f || sample.y >= frame_buffer.resolution.y) return;

    int x = int(sample.x);
    int y = int(sample.y);

    frame_buffer.color[x + y * frame_buffer.resolution.x] += color;
}

/// Atomically adds the given color to pixel containing the given position.
__forceinline__ __device__ void
atomic_add_color(const Framebuffer& frame_buffer, const float2& sample, const float3& color)
{
    if (sample.x < 0.f || sample.x >= frame_buffer.resolution.x) return;
    if (sample.y < 0.f || sample.y >= frame_buffer.resolution.y) return;

    int x = int(sample.x);
    int y = int(sample.y);

    unsigned int pixel = x + y * frame_buffer.resolution.x;
    atomicAdd(&frame_buffer.color[pixel].x, color.x);
    atomicAdd(&frame_buffer.color[pixel].y, color.y);
    atomicAdd(&frame_buffer.color[pixel].z, color.z);
}

#endif // KERNEL_FRAME_BUFFER_CUH
