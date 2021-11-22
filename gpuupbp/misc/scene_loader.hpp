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

#ifndef SCENE_SCENE_HPP
#define SCENE_SCENE_HPP

#include "../host/camera.hpp"
#include "../host/env_map.hpp"
#include "../host/scene.hpp"
#include "../shared/scene.h"
#include "config.hpp"

#include <optix_types.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

struct SceneLoader {
    void load_scene(const Config& config)
    {
        float3 bbox_min = make_float3(1e36f);
        float3 bbox_max = make_float3(-1e36f);

        scene.background_light_idx = UINT32_MAX;
        scene.has_env_map = false;

        if (config.scene_id > -1) {
            load_predefined(
                make_float2(static_cast<float>(config.resolution.x), static_cast<float>(config.resolution.y)),
                g_scene_configs[config.scene_id],
                bbox_min,
                bbox_max);
        } else {
            load_from_obj(config.scene_obj_file.c_str(),
                          make_float2(static_cast<float>(config.resolution.x), static_cast<float>(config.resolution.y)),
                          bbox_min,
                          bbox_max);
        }

        // build scene sphere
        float radius2 = square(bbox_max - bbox_min);
        if (radius2 == 0 || radius2 > INFTY) radius2 = 1.f;

        scene.scene_sphere.center = (bbox_max + bbox_min) * .5f;
        scene.scene_sphere.radius = std::sqrt(radius2) * .5f;
        scene.scene_sphere.inv_radius_sqr = 1.f / (scene.scene_sphere.radius * scene.scene_sphere.radius);

        // some simple checks
        assert(scene.global_medium_idx < scene.mat_count);
        assert(scene.background_light_idx < scene.light_count || scene.background_light_idx == UINT32_MAX);
    }

    void delete_scene_loader() const
    {
        delete_scene_host(scene);
        free(records);
    }

    /// Loads a Cornell Box scene.
    void load_predefined(const float2& resolution, const SceneConfig& scene_config, float3& bbox_min, float3& bbox_max);

    /// Loads scene from a selected OBJ file.
    /// Note that OBJ files cannot specify spheres.
    void load_from_obj(const char* file, const float2& resolution, float3& bbox_min, float3& bbox_max);

    /// Increment scene bounding box for a single point.
    static void grow_bb(float3& bbox_min, float3& bbox_max, const float3& p)
    {
        bbox_min.x = fminf(bbox_min.x, p.x);
        bbox_max.x = fmaxf(bbox_max.x, p.x);
        bbox_min.y = fminf(bbox_min.y, p.y);
        bbox_max.y = fmaxf(bbox_max.y, p.y);
        bbox_min.z = fminf(bbox_min.z, p.z);
        bbox_max.z = fmaxf(bbox_max.z, p.z);
    }

    /// The scene that is loaded by either one of two loading methods.
    Scene scene;
    /// Number of records.
    uint32_t record_count;
    /// (material, light) pairs.
    Record* records;
};

#endif // SCENE_SCENE_HPP
