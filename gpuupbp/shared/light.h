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

#ifndef SHARED_LIGHT_H
#define SHARED_LIGHT_H

#include "frame.h"

#include <vector_types.h>

/** Implements an abstract light type using a tagged union. */

enum LightType { AREA, BACKGROUND, DIRECTIONAL, POINT_LIGHT };

struct AreaLight {
    /// One corner of the triangle
    float3 p0;
    /// From p0 to p1.
    float3 e1;
    /// From p0 to p2.
    float3 e2;
    Frame frame;
    float3 intensity;
    float inv_area;
};

struct BackgroundLight {
    float3 background_color;
    /// Whether the background light makes use of the environment map,
    /// and the environment map is set in the scene.
    bool uses_env_map;
    float scale;
};

struct DirectionalLight {
    Frame frame;
    float3 intensity;
};

struct PointLight {
    float3 position;
    float3 intensity;
};

struct AbstractLight {
    /// Id of material enclosing the light (-1 for light in global medium).
    int mat_id;
    /// Id of medium enclosing the light (-1 for light in global medium).
    int med_id;
    LightType type;

    union {
        AreaLight area;
        BackgroundLight background;
        DirectionalLight directional;
        PointLight point;
    };
};

#endif // SHARED_LIGHT_H
