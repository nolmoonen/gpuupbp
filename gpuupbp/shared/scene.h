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

#ifndef SHARED_SCENE_H
#define SHARED_SCENE_H

#include "camera.h"
#include "env_map.h"
#include "frame.h"
#include "light.h"
#include "material.h"
#include "medium.h"

#include <optix.h>
#include <vector_types.h>

/// Array of triangles.
struct Triangles {
    /// Number of triangles.
    unsigned int count;

    /// Array of vertices, every three consecutive vertices form a triangle.
    /// The vertex consist of an x, y, and z component and includes one
    /// float for padding.
    /// Size is 3 * {count}.
    float4* verts;

    /// Array of normals, every three consecutive vertices form a triangle.
    /// The normal consist of an x, y, and z component and a face normal.
    /// Size is 4 * {count}.
    float3* norms;

    /// Array of material indices, every element representing a triangle.
    /// Size is {count}.
    /// Note: not copied to device (but used to build SBT).
    unsigned int* records;
};

/// Array of spheres.
struct Spheres {
    /// Number of spheres.
    unsigned int count;

    /// Array of bounding boxes, each representing a sphere.
    /// Size is {count}.
    OptixAabb* aabbs;

    /// Array of material indices, every element representing a sphere.
    /// Size is {count}.
    /// Note: not copied to device (but used to build SBT).
    unsigned int* records;
};

/// Bounding sphere of the scene.
struct SceneSphere {
    /// Center of the scene's bounding sphere.
    float3 center;
    /// Radius of the scene's bounding sphere.
    float radius;
    /// 1.f / (scene_radius * scene_radius).
    float inv_radius_sqr;
};

/// Structure to count unique material and light pairs.
struct Record {
    /// May not be negative.
    // todo make unsigned
    int id_material;
    /// May be -1 if no light is associated.
    int id_light;
};

/// Scene representation.
struct Scene {
    /// Array of materials.
    Material* materials;
    /// Number of materials.
    unsigned int mat_count;
    /// Array of media.
    Medium* media;
    /// Number of media.
    unsigned int med_count;
    /// All scene lights.
    /// Only one can be a background light.
    AbstractLight* lights;
    /// Number of lights.
    unsigned int light_count;
    /// Index into {media}, which is the global medium.
    unsigned int global_medium_idx;
    /// Camera.
    Camera camera;

    /// Index into {lights}, which is the background light. UINT32_MAX if none.
    unsigned int background_light_idx;
    SceneSphere scene_sphere;
    EnvMap env_map;
    bool has_env_map;

    /** Scene geometry. */
    Triangles triangles_real;
    Triangles triangles_imag;
    Spheres spheres_real;
    Spheres spheres_imag;
};

#endif // SHARED_SCENE_H
