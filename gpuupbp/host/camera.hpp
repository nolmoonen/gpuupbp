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

#ifndef HOST_CAMERA_HPP
#define HOST_CAMERA_HPP

#include "../shared/camera.h"
#include "../shared/vec_math_ext.h"

inline void init_camera(Camera& camera,
                        const float3& origin,
                        const float3& target,
                        const float3& roll,
                        const float2& resolution,
                        float horizontal_fov,
                        float focal_dist,
                        int mat_id = -1,
                        int med_id = -1)
{
    camera.origin = origin;
    camera.direction = normalize(target - origin);

    camera.resolution = resolution;

    const float3 forward = camera.direction;
    const float3 up = normalize(cross(roll, -forward));
    const float3 left = cross(-forward, up);

    const float3 pos = make_float3(dot(up, origin), dot(left, origin), dot(-forward, origin));

    sutil::Matrix4x4 world_to_camera = sutil::Matrix4x4::identity();
    world_to_camera.setRow(0, make_float4(up, -pos.x));
    world_to_camera.setRow(1, make_float4(left, -pos.y));
    world_to_camera.setRow(2, make_float4(-forward, -pos.z));

    const sutil::Matrix4x4 perspective = sutil::perspective(horizontal_fov, 0.1f, 10000.f);
    const sutil::Matrix4x4 world_to_n_screen = perspective * world_to_camera;
    const sutil::Matrix4x4 n_screen_to_world = world_to_n_screen.inverse();

    float aspect = resolution.x / resolution.y;

    camera.world_to_raster = sutil::Matrix4x4::scale(make_float3(resolution.x * .5f, resolution.x * .5f, 0.f)) *
                             sutil::Matrix4x4::translate(make_float3(1.f, 1.f / aspect, 0.f)) * world_to_n_screen;

    camera.raster_to_world = n_screen_to_world * sutil::Matrix4x4::translate(make_float3(-1.f, -1.f / aspect, 0.f)) *
                             sutil::Matrix4x4::scale(make_float3(2.f / resolution.x, 2.f / resolution.x, 0.f));

    const float tan_half_angle = tanf(horizontal_fov * PI_F / 360.f);
    camera.image_plane_dist = resolution.x / (2.f * tan_half_angle);

    camera.mat_id = mat_id;
    camera.med_id = med_id;
}

#endif // HOST_CAMERA_HPP
