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

#ifndef HOST_LIGHT_HPP
#define HOST_LIGHT_HPP

#include "../shared/light.h"

/// Area light constructor.
inline void init_area_light(AbstractLight& light,
                            const float3& p0,
                            const float3& p1,
                            const float3& p2,
                            const float3& intensity,
                            int mat_id,
                            int med_id)
{
    light.area.p0 = p0;
    light.area.e1 = p1 - p0;
    light.area.e2 = p2 - p0;
    float3 normal = cross(light.area.e1, light.area.e2);
    float len = length(normal);
    light.area.inv_area = 2.f / len;
    light.area.frame.set_from_z(normal);
    light.area.intensity = intensity;
    light.mat_id = mat_id;
    light.med_id = med_id;
    light.type = AREA;
}

/// Background light constructor.
inline void init_background_light(AbstractLight& light, const float3& color, float scale)
{
    light.background.background_color = color;
    light.background.uses_env_map = false;
    light.background.scale = scale;
    light.mat_id = -1;
    light.med_id = -1;
    light.type = BACKGROUND;
}

/// Directional light constructor
inline void init_directional_light(AbstractLight& light, const float3& direction, const float3& intensity)
{
    light.directional.frame.set_from_z(direction);
    light.directional.intensity = intensity;
    light.mat_id = -1;
    light.med_id = -1;
    light.type = DIRECTIONAL;
}

/// Point light constructor,
inline void init_point_light(AbstractLight& light, const float3& position, const float3& intensity)
{
    light.point.position = position;
    light.point.intensity = intensity;
    light.mat_id = -1;
    light.med_id = -1;
    light.type = POINT_LIGHT;
}

#endif // HOST_LIGHT_HPP
