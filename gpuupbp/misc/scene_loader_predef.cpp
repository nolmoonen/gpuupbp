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

#include "../host/light.hpp"
#include "../host/material.hpp"
#include "../host/medium.hpp"
#include "optix_helper.hpp"
#include "scene_loader.hpp"

struct PredefTriangle {
    PredefTriangle(const float3& p0, const float3& p1, const float3& p2, int ma_id, int me_id = -1, int li_id = -1)
    {
        p[0] = p0;
        p[1] = p1;
        p[2] = p2;
        mat_id = ma_id;
        med_id = me_id;
        light_id = li_id;
        normal = normalize(cross(p[1] - p[0], p[2] - p[0]));
        n[0] = normal;
        n[1] = normal;
        n[2] = normal;
    }

  public:
    float3 p[3];
    float3 n[3];
    float2 t[3];
    int mat_id;
    int med_id;
    int light_id;
    float3 normal;
};

struct PredefSphere {
    PredefSphere(const float3& aCenter, float aRadius, int ma_id, int me_id = -1, int li_id = -1)
    {
        center = aCenter;
        radius = aRadius;
        mat_id = ma_id;
        med_id = me_id;
        light_id = li_id;
    }

    float3 center;
    float radius;
    int mat_id;
    int med_id;
    int light_id;
};

void set_triangles(const std::vector<PredefTriangle>& tr_list,
                   Triangles& triangles,
                   uint32_t record_start,
                   float3& bbox_min,
                   float3& bbox_max)
{
    triangles.count = static_cast<uint32_t>(tr_list.size());
    // allocate mem
    triangles.verts = static_cast<float4*>(malloc(3 * sizeof(float4) * triangles.count));
    triangles.norms = static_cast<float3*>(malloc(4 * sizeof(float3) * triangles.count));
    triangles.records = static_cast<unsigned int*>(malloc(sizeof(unsigned int) * triangles.count));
    // initialize, simply create a new record for every triangle
    for (uint32_t i = 0; i < tr_list.size(); i++) {
        triangles.verts[3 * i + 0] = make_float4(tr_list[i].p[0], 0.f);
        triangles.verts[3 * i + 1] = make_float4(tr_list[i].p[1], 0.f);
        triangles.verts[3 * i + 2] = make_float4(tr_list[i].p[2], 0.f);
        triangles.norms[4 * i + 0] = tr_list[i].n[0];
        triangles.norms[4 * i + 1] = tr_list[i].n[1];
        triangles.norms[4 * i + 2] = tr_list[i].n[2];
        triangles.norms[4 * i + 3] = tr_list[i].normal;
        triangles.records[i] = record_start + i;

        // grow bounding box
        SceneLoader::grow_bb(bbox_min, bbox_max, tr_list[i].p[0]);
        SceneLoader::grow_bb(bbox_min, bbox_max, tr_list[i].p[1]);
        SceneLoader::grow_bb(bbox_min, bbox_max, tr_list[i].p[2]);
    }
}

void set_spheres(const std::vector<PredefSphere>& sp_list,
                 Spheres& spheres,
                 uint32_t record_start,
                 float3& bbox_min,
                 float3& bbox_max)
{
    spheres.count = static_cast<uint32_t>(sp_list.size());
    // allocate mem
    spheres.aabbs = static_cast<OptixAabb*>(malloc(sizeof(OptixAabb) * spheres.count));
    spheres.records = static_cast<unsigned int*>(malloc(sizeof(unsigned int) * spheres.count));
    // initialize, simply create a new record for every sphere
    for (uint32_t i = 0; i < sp_list.size(); i++) {
        optix::sphere_bound(
            sp_list[i].center.x, sp_list[i].center.y, sp_list[i].center.z, sp_list[i].radius, &spheres.aabbs[i].minX);
        spheres.records[i] = record_start + i;

        // grow bounding box
        SceneLoader::grow_bb(bbox_min, bbox_max, sp_list[i].center - make_float3(sp_list[i].radius));
        SceneLoader::grow_bb(bbox_min, bbox_max, sp_list[i].center + make_float3(sp_list[i].radius));
    }
}

void SceneLoader::load_predefined(const float2& resolution,
                                  const SceneConfig& scene_config,
                                  float3& bbox_min,
                                  float3& bbox_max)
{
    /** camera */

    init_camera(scene.camera,
                make_float3(-0.0439815f, -4.12529f, 0.222539f),
                make_float3(-0.03709525f, -3.126785f, 0.1683229f),
                make_float3(3.73896e-4f, 0.0542148f, 0.998529f),
                resolution,
                45.0f,
                1.0f);

    /** materials */

    std::vector<Material> materials;
    materials.resize(SceneConfig::MaterialType::kMaterialsCount);
    Material mat{};

    // diffuse red
    reset_material(mat);
    mat.diffuse_reflectance = make_float3(0.803922f, 0.152941f, 0.152941f);
    materials[SceneConfig::MaterialType::kDiffuseRed] = mat;

    // diffuse green
    reset_material(mat);
    mat.diffuse_reflectance = make_float3(0.156863f, 0.803922f, 0.172549f);
    materials[SceneConfig::MaterialType::kDiffuseGreen] = mat;

    // diffuse blue
    reset_material(mat);
    mat.diffuse_reflectance = make_float3(0.156863f, 0.172549f, 0.803922f);
    materials[SceneConfig::MaterialType::kDiffuseBlue] = mat;

    // diffuse white
    reset_material(mat);
    mat.diffuse_reflectance = make_float3(0.803922f, 0.803922f, 0.803922f);
    materials[SceneConfig::MaterialType::kDiffuseWhite] = mat;

    // glossy white
    reset_material(mat);
    mat.diffuse_reflectance = make_float3(0.1f);
    mat.phong_reflectance = make_float3(0.7f);
    mat.phong_exponent = 90.f;
    materials[SceneConfig::MaterialType::kGlossyWhite] = mat;

    // mirror
    reset_material(mat);
    mat.mirror_reflectance = make_float3(1.f);
    materials[SceneConfig::MaterialType::kMirror] = mat;

    // water
    reset_material(mat);
    mat.mirror_reflectance = make_float3(0.7f);
    mat.ior = 1.33f;
    materials[SceneConfig::MaterialType::kWaterMaterial] = mat;

    // ice
    reset_material(mat);
    mat.mirror_reflectance = make_float3(0.5f);
    mat.ior = 1.31f;
    materials[SceneConfig::MaterialType::kIce] = mat;

    // glass
    reset_material(mat);
    mat.mirror_reflectance = make_float3(1.f);
    mat.ior = 1.6f;
    materials[SceneConfig::MaterialType::kGlass] = mat;

    // for area lights
    reset_material(mat);
    materials[SceneConfig::MaterialType::kLight] = mat;

    /** media */

    scene.med_count = SceneConfig::MediumType::kMediaCount;
    scene.media = static_cast<Medium*>(malloc(sizeof(Medium) * scene.med_count));

    // clear
    init_medium_clear(scene.media[SceneConfig::MediumType::kClear]);

    // water
    init_homogeneous_medium(scene.media[SceneConfig::MediumType::kWaterMedium],
                            make_float3(0.7f, 0.6f, 0.0f),
                            make_float3(0.0f),
                            make_float3(0.0f),
                            MEDIUM_SURVIVAL_PROB);

    // red absorbing
    init_homogeneous_medium(scene.media[SceneConfig::MediumType::kRedAbsorbing],
                            make_float3(0.0f, 1.0f, 1.0f),
                            make_float3(0.0f),
                            make_float3(0.0f),
                            MEDIUM_SURVIVAL_PROB);

    // yellow emitting
    init_homogeneous_medium(scene.media[SceneConfig::MediumType::kYellowEmitting],
                            make_float3(0.0f),
                            make_float3(0.7f, 0.7f, 0.0f),
                            make_float3(0.0f),
                            MEDIUM_SURVIVAL_PROB);

    // yellow-green absorbing, emitting and scattering
    init_homogeneous_medium(scene.media[SceneConfig::MediumType::kYellowGreenAbsorbingEmittingAndScattering],
                            make_float3(0.5f, 0.0f, 0.5f),
                            make_float3(1.0f, 1.0f, 0.0f),
                            make_float3(0.1f, 0.1f, 0.0f),
                            MEDIUM_SURVIVAL_PROB);

    // blue absorbing and emitting
    init_homogeneous_medium(scene.media[SceneConfig::MediumType::kBlueAbsorbingAndEmitting],
                            make_float3(0.1f, 0.1f, 0.0f),
                            make_float3(0.0f, 0.0f, 1.0f),
                            make_float3(0.0f),
                            MEDIUM_SURVIVAL_PROB);

    // white isoscattering
    init_homogeneous_medium(scene.media[SceneConfig::MediumType::kWhiteIsoScattering],
                            make_float3(0.0f),
                            make_float3(0.0f),
                            make_float3(0.9f),
                            MEDIUM_SURVIVAL_PROB);

    // white anisoscattering
    init_homogeneous_medium(scene.media[SceneConfig::MediumType::kWhiteAnisoScattering],
                            make_float3(0.0f),
                            make_float3(0.0f),
                            make_float3(0.9f),
                            MEDIUM_SURVIVAL_PROB,
                            0.6f);

    // weak white isoscattering
    init_homogeneous_medium(scene.media[SceneConfig::MediumType::kWeakWhiteIsoScattering],
                            make_float3(0.0f),
                            make_float3(0.0f),
                            make_float3(0.1f),
                            MEDIUM_SURVIVAL_PROB);

    // weak yellow isoscattering
    init_homogeneous_medium(scene.media[SceneConfig::MediumType::kWeakYellowIsoScattering],
                            make_float3(0.0f),
                            make_float3(0.0f),
                            make_float3(0.1f, 0.1f, 0.0f),
                            MEDIUM_SURVIVAL_PROB);

    // weak white anisoscattering
    init_homogeneous_medium(scene.media[SceneConfig::MediumType::kWeakWhiteAnisoScattering],
                            make_float3(0.0f),
                            make_float3(0.0f),
                            make_float3(0.1f),
                            MEDIUM_SURVIVAL_PROB,
                            0.6f);

    // reddish light
    init_homogeneous_medium(scene.media[SceneConfig::MediumType::kLightReddishMedium],
                            make_float3(0.0f, 0.02f, 0.0f),
                            make_float3(0.0f),
                            make_float3(0.002f, 0.002f, 0.0f),
                            MEDIUM_SURVIVAL_PROB,
                            0.6f);

    // absorbing and anisoscattering
    init_homogeneous_medium(scene.media[SceneConfig::MediumType::kAbsorbingAnisoScattering],
                            make_float3(0.02f, 2.0f, 2.0f),
                            make_float3(0.0f),
                            make_float3(12.0f, 20.0f, 20.0f),
                            MEDIUM_SURVIVAL_PROB,
                            -0.3f);

    // global infinite medium
    scene.global_medium_idx = scene_config.global_medium;

    /** geometry */

    // Cornell box vertices
    float3 cb[8] = {make_float3(-1.27029f, 1.30455f, -1.28002f),
                    make_float3(1.28975f, 1.30455f, -1.28002f),
                    make_float3(1.28975f, 1.30455f, 1.28002f),
                    make_float3(-1.27029f, 1.30455f, 1.28002f),
                    make_float3(-1.27029f, -1.25549f, -1.28002f),
                    make_float3(1.28975f, -1.25549f, -1.28002f),
                    make_float3(1.28975f, -1.25549f, 1.28002f),
                    make_float3(-1.27029f, -1.25549f, 1.28002f)};
    float3 center_of_floor = (cb[0] + cb[1] + cb[4] + cb[5]) * (1.f / 4.f);
    float3 center_of_box = (cb[0] + cb[1] + cb[2] + cb[3] + cb[4] + cb[5] + cb[6] + cb[7]) * (1.f / 8.f);

    // Spheres params
    float large_sphere_radius = 0.8f;
    float3 large_sphere_center = center_of_floor + make_float3(0, 0, large_sphere_radius + 0.01f);
    float small_sphere_radius = 0.5f;
    float3 left_wall_center = (cb[0] + cb[4]) * (1.f / 2.f) + make_float3(0, 0, small_sphere_radius + 0.01f);
    float3 right_wall_center = (cb[1] + cb[5]) * (1.f / 2.f) + make_float3(0, 0, small_sphere_radius + 0.01f);
    float xlen = right_wall_center.x - left_wall_center.x;
    float3 left_small_sphere_center = left_wall_center + make_float3(2.f * xlen / 7.f, 0, 0);
    float3 right_small_sphere_center = right_wall_center - make_float3(2.f * xlen / 7.f, 0, 0);
    float overlap = 0.5f * small_sphere_radius;
    float3 bottom_left_small_sphere_center = center_of_box + make_float3(-overlap, 0, -overlap);
    float3 bottom_right_small_sphere_center = center_of_box + make_float3(overlap, 0, -overlap);
    float3 top_small_sphere_center = center_of_box + make_float3(0, 0, overlap);
    float3 very_large_sphere_center = center_of_box;
    float very_large_sphere_radius = 1.f;

    // Already added geometry map
    std::vector<bool> used(SceneConfig::GeometryType::kGeometryCount, false);

    std::vector<PredefTriangle> real_triangle_list;
    std::vector<PredefTriangle> imag_triangle_list;
    std::vector<PredefSphere> real_sphere_list;
    std::vector<PredefSphere> imag_sphere_list;

    // Geometry construction
    for (auto i = scene_config.elements.begin(); i < scene_config.elements.end(); ++i) {
        // skip if already added
        if (used[i->geom]) {
            continue;
        } else {
            used[i->geom] = true;
        }

        std::vector<PredefTriangle>* triangle_list;
        std::vector<PredefSphere>* sphere_list;
        if (i->mat >= 0) {
            // real material
            triangle_list = &real_triangle_list;
            sphere_list = &real_sphere_list;
        } else {
            // imaginary material
            triangle_list = &imag_triangle_list;
            sphere_list = &imag_sphere_list;
        }

        // create new material based on the rules laid out in SmallUPBP thesis
        int priority = (1 + i->mat) * 10 + 1 + i->med;
        int mat_id = SceneConfig::MaterialType::kMaterialsCount;
        if (i->geom > SceneConfig::GeometryType::kFloor) {
            for (; mat_id != materials.size(); ++mat_id) {
                if (materials[mat_id].priority == priority) break;
            }
            if (mat_id == materials.size()) {
                reset_material(mat);
                if (i->mat >= 0) mat = materials[i->mat];
                mat.priority = priority;
                mat.real = i->mat >= 0;
                materials.push_back(mat);
            }
        } else {
            mat_id = i->mat;
        }
        materials[mat_id].med_idx = i->med;

        switch (i->geom) {
        case SceneConfig::GeometryType::kLeftWall:
            triangle_list->push_back(PredefTriangle(cb[3], cb[7], cb[4], mat_id, i->med));
            triangle_list->push_back(PredefTriangle(cb[4], cb[0], cb[3], mat_id, i->med));
            break;
        case SceneConfig::GeometryType::kRightWall:
            triangle_list->push_back(PredefTriangle(cb[1], cb[5], cb[6], mat_id, i->med));
            triangle_list->push_back(PredefTriangle(cb[6], cb[2], cb[1], mat_id, i->med));
            break;
        case SceneConfig::GeometryType::kFrontFacingFrontWall:
            triangle_list->push_back(PredefTriangle(cb[4], cb[5], cb[6], mat_id, i->med));
            triangle_list->push_back(PredefTriangle(cb[6], cb[7], cb[4], mat_id, i->med));
            break;
        case SceneConfig::GeometryType::kBackFacingFrontWall:
            triangle_list->push_back(PredefTriangle(cb[6], cb[5], cb[4], mat_id, i->med));
            triangle_list->push_back(PredefTriangle(cb[4], cb[7], cb[6], mat_id, i->med));
            break;
        case SceneConfig::GeometryType::kBackWall:
            triangle_list->push_back(PredefTriangle(cb[0], cb[1], cb[2], mat_id, i->med));
            triangle_list->push_back(PredefTriangle(cb[2], cb[3], cb[0], mat_id, i->med));
            break;
        case SceneConfig::GeometryType::kCeiling:
            if (scene_config.light != SceneConfig::LightType::kLightCeilingAreaBig) {
                triangle_list->push_back(PredefTriangle(cb[2], cb[6], cb[7], mat_id, i->med));
                triangle_list->push_back(PredefTriangle(cb[7], cb[3], cb[2], mat_id, i->med));
            }
            break;
        case SceneConfig::GeometryType::kFloor:
            triangle_list->push_back(PredefTriangle(cb[0], cb[4], cb[5], mat_id, i->med));
            triangle_list->push_back(PredefTriangle(cb[5], cb[1], cb[0], mat_id, i->med));
            break;
        case SceneConfig::GeometryType::kLargeSphereMiddle:
            sphere_list->push_back(PredefSphere(large_sphere_center, large_sphere_radius, mat_id, i->med));
            break;
        case SceneConfig::GeometryType::kSmallSphereLeft:
            sphere_list->push_back(PredefSphere(left_small_sphere_center, small_sphere_radius, mat_id, i->med));
            break;
        case SceneConfig::GeometryType::kSmallSphereRight:
            sphere_list->push_back(PredefSphere(right_small_sphere_center, small_sphere_radius, mat_id, i->med));
            break;
        case SceneConfig::GeometryType::kSmallSphereBottomLeft:
            sphere_list->push_back(PredefSphere(bottom_left_small_sphere_center, small_sphere_radius, mat_id, i->med));
            break;
        case SceneConfig::GeometryType::kSmallSphereBottomRight:
            sphere_list->push_back(PredefSphere(bottom_right_small_sphere_center, small_sphere_radius, mat_id, i->med));
            break;
        case SceneConfig::GeometryType::kSmallSphereTop:
            sphere_list->push_back(PredefSphere(top_small_sphere_center, small_sphere_radius, mat_id, i->med));
            break;
        case SceneConfig::GeometryType::kVeryLargeSphere:
            sphere_list->push_back(PredefSphere(very_large_sphere_center, very_large_sphere_radius, mat_id, i->med));
            break;
        case SceneConfig::GeometryType::kVeryLargeBox:
            const float mult = 1.0f;
            triangle_list->push_back(PredefTriangle(cb[7] * mult, cb[3] * mult, cb[4] * mult, mat_id, i->med));
            triangle_list->push_back(PredefTriangle(cb[0] * mult, cb[4] * mult, cb[3] * mult, mat_id, i->med));
            triangle_list->push_back(PredefTriangle(cb[5] * mult, cb[1] * mult, cb[6] * mult, mat_id, i->med));
            triangle_list->push_back(PredefTriangle(cb[2] * mult, cb[6] * mult, cb[1] * mult, mat_id, i->med));
            triangle_list->push_back(PredefTriangle(cb[4] * mult, cb[5] * mult, cb[6] * mult, mat_id, i->med));
            triangle_list->push_back(PredefTriangle(cb[6] * mult, cb[7] * mult, cb[4] * mult, mat_id, i->med));
            triangle_list->push_back(PredefTriangle(cb[1] * mult, cb[0] * mult, cb[2] * mult, mat_id, i->med));
            triangle_list->push_back(PredefTriangle(cb[3] * mult, cb[2] * mult, cb[0] * mult, mat_id, i->med));
            triangle_list->push_back(PredefTriangle(cb[6] * mult, cb[2] * mult, cb[7] * mult, mat_id, i->med));
            triangle_list->push_back(PredefTriangle(cb[3] * mult, cb[7] * mult, cb[2] * mult, mat_id, i->med));
            triangle_list->push_back(PredefTriangle(cb[4] * mult, cb[0] * mult, cb[5] * mult, mat_id, i->med));
            triangle_list->push_back(PredefTriangle(cb[1] * mult, cb[5] * mult, cb[0] * mult, mat_id, i->med));
            break;
        }
    }

    /** lights */

    // there is only one light
    switch (scene_config.light) {
    case SceneConfig::LightType::kLightCeilingAreaBig: {
        scene.light_count = 2;
        scene.lights = static_cast<AbstractLight*>(malloc(sizeof(AbstractLight) * scene.light_count));

        init_area_light(scene.lights[0], cb[2], cb[6], cb[7], make_float3(0.95492965f), -1, -1);
        init_area_light(scene.lights[1], cb[7], cb[3], cb[2], make_float3(0.95492965f), -1, -1);

        // always real geometry
        real_triangle_list.push_back(PredefTriangle(cb[2], cb[6], cb[7], SceneConfig::MaterialType::kLight, -1, 0));
        real_triangle_list.push_back(PredefTriangle(cb[7], cb[3], cb[2], SceneConfig::MaterialType::kLight, -1, 1));
    } break;
    case SceneConfig::LightType::kLightCeilingAreaSmall: {
        // Box vertices
        const float3 lb[8] = {make_float3(-0.25f, 0.25f, 1.26002f),
                              make_float3(0.25f, 0.25f, 1.26002f),
                              make_float3(0.25f, 0.25f, 1.28002f),
                              make_float3(-0.25f, 0.25f, 1.28002f),
                              make_float3(-0.25f, -0.25f, 1.26002f),
                              make_float3(0.25f, -0.25f, 1.26002f),
                              make_float3(0.25f, -0.25f, 1.28002f),
                              make_float3(-0.25f, -0.25f, 1.28002f)};

        scene.light_count = 2;
        scene.lights = static_cast<AbstractLight*>(malloc(sizeof(AbstractLight) * scene.light_count));

        init_area_light(scene.lights[0], lb[0], lb[5], lb[4], make_float3(25.03329895614464f), -1, -1);
        init_area_light(scene.lights[1], lb[5], lb[0], lb[1], make_float3(25.03329895614464f), -1, -1);

        // Back wall
        real_triangle_list.push_back(PredefTriangle(lb[0], lb[2], lb[1], SceneConfig::MaterialType::kDiffuseWhite));
        real_triangle_list.push_back(PredefTriangle(lb[2], lb[0], lb[3], SceneConfig::MaterialType::kDiffuseWhite));
        // Left wall
        real_triangle_list.push_back(PredefTriangle(lb[3], lb[4], lb[7], SceneConfig::MaterialType::kDiffuseWhite));
        real_triangle_list.push_back(PredefTriangle(lb[4], lb[3], lb[0], SceneConfig::MaterialType::kDiffuseWhite));
        // Right wall
        real_triangle_list.push_back(PredefTriangle(lb[1], lb[6], lb[5], SceneConfig::MaterialType::kDiffuseWhite));
        real_triangle_list.push_back(PredefTriangle(lb[6], lb[1], lb[2], SceneConfig::MaterialType::kDiffuseWhite));
        // Front wall
        real_triangle_list.push_back(PredefTriangle(lb[4], lb[5], lb[6], SceneConfig::MaterialType::kDiffuseWhite));
        real_triangle_list.push_back(PredefTriangle(lb[6], lb[7], lb[4], SceneConfig::MaterialType::kDiffuseWhite));
        // Floor
        real_triangle_list.push_back(PredefTriangle(lb[0], lb[5], lb[4], SceneConfig::MaterialType::kLight, -1, 0));
        real_triangle_list.push_back(PredefTriangle(lb[5], lb[0], lb[1], SceneConfig::MaterialType::kLight, -1, 1));
    } break;
    case SceneConfig::LightType::kLightCeilingAreaSmallDistant: {
        const float posY = -1.5f;
        // Box vertices
        const float3 lb[8] = {make_float3(-0.25f, 0.25f + posY, 1.26002f),
                              make_float3(0.25f, 0.25f + posY, 1.26002f),
                              make_float3(0.25f, 0.25f + posY, 1.28002f),
                              make_float3(-0.25f, 0.25f + posY, 1.28002f),
                              make_float3(-0.25f, -0.25f + posY, 1.26002f),
                              make_float3(0.25f, -0.25f + posY, 1.26002f),
                              make_float3(0.25f, -0.25f + posY, 1.28002f),
                              make_float3(-0.25f, -0.25f + posY, 1.28002f)};

        scene.light_count = 2;
        scene.lights = static_cast<AbstractLight*>(malloc(sizeof(AbstractLight) * scene.light_count));

        init_area_light(scene.lights[0], lb[0], lb[5], lb[4], make_float3(25.03329895614464f), -1, -1);
        init_area_light(scene.lights[1], lb[5], lb[0], lb[1], make_float3(25.03329895614464f), -1, -1);

        real_triangle_list.push_back(PredefTriangle(lb[0], lb[5], lb[4], SceneConfig::MaterialType::kLight, -1, 0));
        real_triangle_list.push_back(PredefTriangle(lb[5], lb[0], lb[1], SceneConfig::MaterialType::kLight, -1, 1));
    } break;
    case SceneConfig::LightType::kLightCeilingPoint: {

        scene.light_count = 1;
        scene.lights = static_cast<AbstractLight*>(malloc(sizeof(AbstractLight) * scene.light_count));

        init_point_light(scene.lights[0], make_float3(0.0, -0.5, 1.0), make_float3(70.f * (INV_PI_F * 0.25f)));
    } break;
    case SceneConfig::LightType::kLightFacingAreaSmall: {
        // Rectangle vertices
        float3 lr[8] = {make_float3(-0.25f, 0, 0.25f),
                        make_float3(0.25f, 0, 0.25f),
                        make_float3(-0.25f, 0, -0.25f),
                        make_float3(0.25f, 0, -0.25f)};

        scene.light_count = 2;
        scene.lights = static_cast<AbstractLight*>(malloc(sizeof(AbstractLight) * scene.light_count));

        init_area_light(scene.lights[0], lr[2], lr[3], lr[0], make_float3(25.03329895614464f), -1, -1);
        init_area_light(scene.lights[1], lr[1], lr[0], lr[3], make_float3(25.03329895614464f), -1, -1);

        real_triangle_list.push_back(PredefTriangle(lr[2], lr[3], lr[0], SceneConfig::MaterialType::kLight, -1, 0));
        real_triangle_list.push_back(PredefTriangle(lr[1], lr[0], lr[3], SceneConfig::MaterialType::kLight, -1, 1));
    } break;
    case SceneConfig::LightType::kLightFacingPoint: {
        scene.light_count = 1;
        scene.lights = static_cast<AbstractLight*>(malloc(sizeof(AbstractLight) * scene.light_count));

        init_point_light(scene.lights[0], make_float3(0.0f, 0.0f, 0.0f), make_float3(70.f * (INV_PI_F * 0.25f)));
    } break;
    case SceneConfig::LightType::kLightSun: {
        scene.light_count = 1;
        scene.lights = static_cast<AbstractLight*>(malloc(sizeof(AbstractLight) * scene.light_count));

        init_directional_light(scene.lights[0], make_float3(-1.f, 1.5f, -1.f), make_float3(0.5f, 0.2f, 0.f) * 20.f);
    } break;
    case SceneConfig::LightType::kLightBackground: {
        scene.light_count = 1;
        scene.lights = static_cast<AbstractLight*>(malloc(sizeof(AbstractLight) * scene.light_count));

        init_background_light(scene.lights[0], make_float3(135, 206, 250) / 255.f, 1.f);
        scene.background_light_idx = 0;
    } break;
    }

    /** finalize materials */
    scene.mat_count = static_cast<uint32_t>(materials.size());
    scene.materials = static_cast<Material*>(malloc(sizeof(Material) * scene.mat_count));
    memcpy(scene.materials, materials.data(), sizeof(Material) * scene.mat_count);

    /** triangles and spheres */
    // create one record per triangle/sphere, performance is not an issue for
    // the predefined scenes
    record_count = 0;
    set_triangles(real_triangle_list, scene.triangles_real, record_count, bbox_min, bbox_max);
    record_count += static_cast<uint32_t>(real_triangle_list.size());
    set_spheres(real_sphere_list, scene.spheres_real, record_count, bbox_min, bbox_max);
    record_count += static_cast<uint32_t>(real_sphere_list.size());
    set_triangles(imag_triangle_list, scene.triangles_imag, record_count, bbox_min, bbox_max);
    record_count += static_cast<uint32_t>(imag_triangle_list.size());
    set_spheres(imag_sphere_list, scene.spheres_imag, record_count, bbox_min, bbox_max);
    record_count += static_cast<uint32_t>(imag_sphere_list.size());

    /** records */
    records = static_cast<Record*>(malloc(sizeof(Record) * record_count));
    Record* record = records;
    for (uint32_t i = 0; i < real_triangle_list.size(); i++) {
        record->id_material = real_triangle_list[i].mat_id;
        record->id_light = real_triangle_list[i].light_id;
        record++;
    }
    for (uint32_t i = 0; i < real_sphere_list.size(); i++) {
        record->id_material = real_sphere_list[i].mat_id;
        record->id_light = real_sphere_list[i].light_id;
        record++;
    }
    for (uint32_t i = 0; i < imag_triangle_list.size(); i++) {
        record->id_material = imag_triangle_list[i].mat_id;
        record->id_light = imag_triangle_list[i].light_id;
        record++;
    }
    for (uint32_t i = 0; i < imag_sphere_list.size(); i++) {
        record->id_material = imag_sphere_list[i].mat_id;
        record->id_light = imag_sphere_list[i].light_id;
        record++;
    }
    assert(record - records == record_count);
}
