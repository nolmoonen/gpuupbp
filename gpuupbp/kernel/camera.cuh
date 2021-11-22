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

#ifndef KERNEL_CAMERA_CUH
#define KERNEL_CAMERA_CUH

#include "../shared/camera.h"
#include "../shared/frame.h"
#include "../shared/vec_math.h"
#include "../shared/vec_math_ext.h"

#include <vector_types.h>

__forceinline__ __device__ int raster_to_index(const Camera& camera, const float2& pixel_coords)
{
    return int(floorf(pixel_coords.x) + floorf(pixel_coords.y) * camera.resolution.x);
}

__forceinline__ __device__ float2 index_to_raster(const Camera& camera, const int& pixel_index)
{
    const float y = floorf(pixel_index / camera.resolution.x);
    const float x = float(pixel_index) - y * camera.resolution.x;
    return make_float2(x, y);
}

__forceinline__ __device__ float3 raster_to_world(const Camera& camera, const float2& raster_xy)
{
    return sutil::transform_point(camera.raster_to_world, make_float3(raster_xy.x, raster_xy.y, 0));
}

__forceinline__ __device__ float2 world_to_raster(const Camera& camera, const float3& world_pos)
{
    float3 temp = sutil::transform_point(camera.world_to_raster, world_pos);
    return make_float2(temp.x, temp.y);
}

/// Returns false when raster position is outside screen space.
__forceinline__ __device__ bool check_raster(const Camera& camera, const float2& raster_pos)
{
    return raster_pos.x >= 0 && raster_pos.y >= 0 && raster_pos.x < camera.resolution.x &&
           raster_pos.y < camera.resolution.y;
}

__forceinline__ __device__ Ray generate_ray(const Camera& camera, const float2& raster_xy)
{
    const float3 world_raster = raster_to_world(camera, raster_xy);

    Ray ray{};
    ray.origin = camera.origin;
    ray.direction = normalize(world_raster - camera.origin);

    return ray;
}

#endif // KERNEL_CAMERA_CUH
