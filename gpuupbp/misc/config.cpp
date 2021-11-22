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

#include "config.hpp"
#include "../shared/vec_math.h"

#include <cstdio>
#include <iomanip>

std::string get_description(const Config& config)
{
    std::ostringstream oss;
    std::string leading_spaces = "          ";

    oss << "techniques from upbp:";
    if (config.algorithm_flags & BPT) oss << " bpt";
    if (config.algorithm_flags & SURF) oss << " surf";
    if (config.algorithm_flags & PP3D) oss << " pp3d";
    if (config.algorithm_flags & PB2D) oss << " pb2d";
    if (config.algorithm_flags & BB1D) oss << " bb1d";
    if (config.algorithm_flags & BP2D) oss << " bp2d";
    oss << '\n';

    oss << leading_spaces << "resolution:        " << config.resolution.x << "x" << config.resolution.y << '\n'
        << leading_spaces << "max path length:   " << config.max_path_length << '\n';

    if (config.algorithm_flags & BB1D) {
        oss << leading_spaces << "bb1d radius init:  " << config.bb1d_radius_initial << '\n'
            << leading_spaces << "     radius alpha: " << config.bb1d_radius_alpha << '\n';

        oss << leading_spaces << "     used l.paths: " << config.bb1d_used_light_subpath_count << '\n';
    }

    if (config.algorithm_flags & PP3D) {
        oss << leading_spaces << "pp3d radius init:  " << config.pp3d_radius_initial << '\n'
            << leading_spaces << "     radius alpha: " << config.pp3d_radius_alpha << '\n';
    }

    if (config.algorithm_flags & SURF) {
        oss << leading_spaces << "surf radius init:  " << config.surf_radius_initial << '\n'
            << leading_spaces << "     radius alpha: " << config.surf_radius_alpha << '\n';
    }

    if (config.algorithm_flags & PB2D) {
        oss << leading_spaces << "pb2d radius init:  " << config.pb2d_radius_initial << '\n'
            << leading_spaces << "     radius alpha: " << config.pb2d_radius_alpha << '\n';
    }

    if ((config.algorithm_flags & BP2D) || (config.algorithm_flags & BB1D)) {
        if (config.photon_beam_type == SHORT_BEAM) {
            oss << leading_spaces << "photon beam type:  SHORT\n";
        } else {
            oss << leading_spaces << "photon beam type:  LONG\n";
        }
    }

    if ((config.algorithm_flags & PB2D) || (config.algorithm_flags & BB1D)) {
        if (config.query_beam_type == SHORT_BEAM) {
            oss << leading_spaces << "query beam type:   SHORT\n";
        } else {
            oss << leading_spaces << "query beam type:   LONG\n";
        }
    }

    oss << leading_spaces << "     paths/iter:   " << config.path_count_per_iter << '\n';

    if (config.algorithm_flags & COMPATIBLE)
        oss << leading_spaces << "compatible mode" << '\n';
    else if (config.algorithm_flags & PREVIOUS)
        oss << leading_spaces << "previous mode" << '\n';

    if (config.algorithm_flags & SPECULAR_ONLY) oss << leading_spaces << "specular only" << '\n';

    return oss.str();
}

std::vector<SceneConfig> init_scene_configs()
{
    typedef SceneConfig SC;
    std::vector<SC> configs;

    // Preparing a few empty Cornell boxes used in the scenes.

    std::vector<SC::Element> cornell_box_without_floor;
    cornell_box_without_floor.push_back(SC::Element(SC::GeometryType::kLeftWall, SC::MaterialType::kDiffuseGreen));
    cornell_box_without_floor.push_back(SC::Element(SC::GeometryType::kRightWall, SC::MaterialType::kDiffuseRed));
    cornell_box_without_floor.push_back(SC::Element(SC::GeometryType::kBackWall, SC::MaterialType::kDiffuseBlue));
    cornell_box_without_floor.push_back(SC::Element(SC::GeometryType::kCeiling, SC::MaterialType::kDiffuseWhite));

    std::vector<SC::Element> cornell_box_with_diffuse_floor(cornell_box_without_floor);
    cornell_box_with_diffuse_floor.push_back(SC::Element(SC::GeometryType::kFloor, SC::MaterialType::kDiffuseWhite));

    std::vector<SC::Element> cornell_box_with_glossy_floor(cornell_box_without_floor);
    cornell_box_with_glossy_floor.push_back(SC::Element(SC::GeometryType::kFloor, SC::MaterialType::kGlossyWhite));

    std::vector<SC::Element> cornell_box_with_white_back_wall;
    cornell_box_with_white_back_wall.push_back(
        SC::Element(SC::GeometryType::kLeftWall, SC::MaterialType::kDiffuseGreen));
    cornell_box_with_white_back_wall.push_back(
        SC::Element(SC::GeometryType::kRightWall, SC::MaterialType::kDiffuseRed));
    cornell_box_with_white_back_wall.push_back(
        SC::Element(SC::GeometryType::kBackWall, SC::MaterialType::kDiffuseWhite));
    cornell_box_with_white_back_wall.push_back(
        SC::Element(SC::GeometryType::kCeiling, SC::MaterialType::kDiffuseWhite));
    cornell_box_with_white_back_wall.push_back(SC::Element(SC::GeometryType::kFloor, SC::MaterialType::kDiffuseWhite));

    // Each scene is defined by its short and long name, light, global medium and a vector of elements.
    // Each element consists of a geometry, material (optional) and medium (optional).

    // 0
    SC config("abssph",
              "large absorbing sphere + big area light",
              SC::LightType::kLightCeilingAreaBig,
              SC::MediumType::kClear);
    config.add_all_elements(cornell_box_with_diffuse_floor);
    config.add_element(SC::Element(SC::GeometryType::kLargeSphereMiddle, SC::MediumType::kRedAbsorbing));
    configs.push_back(config);

    // 1
    config.reset("isosph",
                 "large isoscattering sphere + small area light",
                 SC::LightType::kLightCeilingAreaSmall,
                 SC::MediumType::kClear);
    config.add_all_elements(cornell_box_with_diffuse_floor);
    config.add_element(SC::Element(SC::GeometryType::kLargeSphereMiddle, SC::MediumType::kWhiteIsoScattering));
    configs.push_back(config);

    // 2
    config.reset("mirsphinfisov",
                 "large mirror sphere + weak isoscattering + small area light",
                 SC::LightType::kLightCeilingAreaSmall,
                 SC::MediumType::kWeakWhiteIsoScattering);
    config.add_all_elements(cornell_box_with_glossy_floor);
    config.add_element(SC::Element(SC::GeometryType::kLargeSphereMiddle, SC::MaterialType::kMirror));
    configs.push_back(config);

    // 3
    config.reset("smswiso",
                 "small media spheres + weak isoscattering + small area light",
                 SC::LightType::kLightCeilingAreaSmall,
                 SC::MediumType::kWeakWhiteIsoScattering);
    config.add_all_elements(cornell_box_with_diffuse_floor);
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereLeft, SC::MediumType::kBlueAbsorbingAndEmitting));
    config.add_element(
        SC::Element(SC::GeometryType::kSmallSphereRight, SC::MediumType::kYellowGreenAbsorbingEmittingAndScattering));
    configs.push_back(config);

    // 4
    config.reset("smswani",
                 "small media spheres + weak anisoscattering + small area light",
                 SC::LightType::kLightCeilingAreaSmall,
                 SC::MediumType::kWeakWhiteAnisoScattering);
    config.add_all_elements(cornell_box_with_diffuse_floor);
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereLeft, SC::MediumType::kBlueAbsorbingAndEmitting));
    config.add_element(
        SC::Element(SC::GeometryType::kSmallSphereRight, SC::MediumType::kYellowGreenAbsorbingEmittingAndScattering));
    configs.push_back(config);

    // 5
    config.reset("ssssun", "specular small spheres + sun", SC::LightType::kLightSun, SC::MediumType::kClear);
    config.add_all_elements(cornell_box_with_glossy_floor);
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereLeft, SC::MaterialType::kMirror));
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereRight, SC::MaterialType::kGlass));
    configs.push_back(config);

    // 6
    config.reset(
        "ssspoint", "specular small spheres + point light", SC::LightType::kLightCeilingPoint, SC::MediumType::kClear);
    config.add_all_elements(cornell_box_with_glossy_floor);
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereLeft, SC::MaterialType::kMirror));
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereRight, SC::MaterialType::kGlass));
    configs.push_back(config);

    // 7
    config.reset("sssback",
                 "specular small spheres + background light",
                 SC::LightType::kLightBackground,
                 SC::MediumType::kClear);
    config.add_all_elements(cornell_box_with_glossy_floor);
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereLeft, SC::MaterialType::kMirror));
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereRight, SC::MaterialType::kGlass));
    configs.push_back(config);

    // 8
    config.reset("mirsph",
                 "large mirror sphere + small area light",
                 SC::LightType::kLightCeilingAreaSmall,
                 SC::MediumType::kClear);
    config.add_all_elements(cornell_box_with_glossy_floor);
    config.add_element(SC::Element(SC::GeometryType::kLargeSphereMiddle, SC::MaterialType::kMirror));
    configs.push_back(config);

    // 9
    config.reset("emisph",
                 "large emitting sphere + small area light",
                 SC::LightType::kLightCeilingAreaSmall,
                 SC::MediumType::kClear);
    config.add_all_elements(cornell_box_with_diffuse_floor);
    config.add_element(SC::Element(SC::GeometryType::kLargeSphereMiddle, SC::MediumType::kYellowEmitting));
    configs.push_back(config);

    // 10
    config.reset("gemisph", "large glass emitting sphere", SC::LightType::kLightNone, SC::MediumType::kClear);
    config.add_all_elements(cornell_box_with_diffuse_floor);
    config.add_element(
        SC::Element(SC::GeometryType::kLargeSphereMiddle, SC::MaterialType::kGlass, SC::MediumType::kYellowEmitting));
    configs.push_back(config);

    // 11
    config.reset("nogeoisobig",
                 "no geometry + isoscattering + big area light",
                 SC::LightType::kLightCeilingAreaBig,
                 SC::MediumType::kWhiteIsoScattering);
    configs.push_back(config);

    // 12
    config.reset("nogeoisosmall",
                 "no geometry + isoscattering + small area light",
                 SC::LightType::kLightCeilingAreaSmall,
                 SC::MediumType::kWhiteIsoScattering);
    configs.push_back(config);

    // 13
    config.reset("nogeoisopt",
                 "no geometry + isoscattering + point light",
                 SC::LightType::kLightCeilingPoint,
                 SC::MediumType::kWhiteIsoScattering);
    configs.push_back(config);

    // 14
    config.reset("nogeoanibig",
                 "no geometry + anisoscattering + big area light",
                 SC::LightType::kLightCeilingAreaBig,
                 SC::MediumType::kWhiteAnisoScattering);
    configs.push_back(config);

    // 15
    config.reset("nogeoanismall",
                 "no geometry + anisoscattering + small area light",
                 SC::LightType::kLightCeilingAreaSmall,
                 SC::MediumType::kWhiteAnisoScattering);
    configs.push_back(config);

    // 16
    config.reset("nogeoanipt",
                 "no geometry + anisoscattering + point light",
                 SC::LightType::kLightCeilingPoint,
                 SC::MediumType::kWhiteAnisoScattering);
    configs.push_back(config);

    // 17
    config.reset("faceisosmall",
                 "no geometry + isoscattering + facing small area light",
                 SC::LightType::kLightFacingAreaSmall,
                 SC::MediumType::kWhiteIsoScattering);
    configs.push_back(config);

    // 18
    config.reset("faceisopt",
                 "no geometry + isoscattering + facing point light",
                 SC::LightType::kLightFacingPoint,
                 SC::MediumType::kWhiteIsoScattering);
    configs.push_back(config);

    // 19
    config.reset("faceanismall",
                 "no geometry + anisoscattering + facing small area light",
                 SC::LightType::kLightFacingAreaSmall,
                 SC::MediumType::kWhiteAnisoScattering);
    configs.push_back(config);

    // 20
    config.reset("faceanipt",
                 "no geometry + anisoscattering + facing point light",
                 SC::LightType::kLightFacingPoint,
                 SC::MediumType::kWhiteAnisoScattering);
    configs.push_back(config);

    // 21
    config.reset("overgwi",
                 "three overlapping spheres (glass, water, ice) + big area light",
                 SC::LightType::kLightCeilingAreaBig,
                 SC::MediumType::kClear);
    config.add_all_elements(cornell_box_with_white_back_wall);
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereBottomLeft, SC::MaterialType::kGlass));
    config.add_element(SC::Element(
        SC::GeometryType::kSmallSphereBottomRight, SC::MaterialType::kWaterMaterial, SC::MediumType::kWaterMedium));
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereTop, SC::MaterialType::kIce));
    configs.push_back(config);

    // 22
    config.reset("rovergwi",
                 "three overlapping spheres (glass, water, ice) + one big red sphere + big area light",
                 SC::LightType::kLightCeilingAreaBig,
                 SC::MediumType::kClear);
    config.add_all_elements(cornell_box_with_white_back_wall);
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereBottomLeft, SC::MaterialType::kGlass));
    config.add_element(SC::Element(
        SC::GeometryType::kSmallSphereBottomRight, SC::MaterialType::kWaterMaterial, SC::MediumType::kWaterMedium));
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereTop, SC::MaterialType::kIce));
    config.add_element(SC::Element(SC::GeometryType::kVeryLargeSphere, SC::MediumType::kRedAbsorbing));
    configs.push_back(config);

    // 23
    config.reset("wovergwi",
                 "three overlapping spheres (glass, water, ice) + one big water sphere + big area light",
                 SC::LightType::kLightCeilingAreaBig,
                 SC::MediumType::kClear);
    config.add_all_elements(cornell_box_with_white_back_wall);
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereBottomLeft, SC::MaterialType::kGlass));
    config.add_element(SC::Element(
        SC::GeometryType::kSmallSphereBottomRight, SC::MaterialType::kWaterMaterial, SC::MediumType::kWaterMedium));
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereTop, SC::MaterialType::kIce));
    config.add_element(SC::Element(
        SC::GeometryType::kVeryLargeSphere, SC::MaterialType::kWaterMaterial, SC::MediumType::kWaterMedium));
    configs.push_back(config);

    // 24
    config.reset("iovergwi",
                 "three overlapping spheres (glass, water, ice) + one big ice sphere + big area light",
                 SC::LightType::kLightCeilingAreaBig,
                 SC::MediumType::kClear);
    config.add_all_elements(cornell_box_with_white_back_wall);
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereBottomLeft, SC::MaterialType::kGlass));
    config.add_element(SC::Element(
        SC::GeometryType::kSmallSphereBottomRight, SC::MaterialType::kWaterMaterial, SC::MediumType::kWaterMedium));
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereTop, SC::MaterialType::kIce));
    config.add_element(SC::Element(SC::GeometryType::kVeryLargeSphere, SC::MaterialType::kIce));
    configs.push_back(config);

    // 25
    config.reset("govergwi",
                 "three overlapping spheres (glass, water, ice) + one big glass sphere + big area light",
                 SC::LightType::kLightCeilingAreaBig,
                 SC::MediumType::kClear);
    config.add_all_elements(cornell_box_with_white_back_wall);
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereBottomLeft, SC::MaterialType::kGlass));
    config.add_element(SC::Element(
        SC::GeometryType::kSmallSphereBottomRight, SC::MaterialType::kWaterMaterial, SC::MediumType::kWaterMedium));
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereTop, SC::MaterialType::kIce));
    config.add_element(SC::Element(SC::GeometryType::kVeryLargeSphere, SC::MaterialType::kGlass));
    configs.push_back(config);

    // 26
    config.reset("movergwi",
                 "three overlapping spheres (glass, water, ice) + one big mirror sphere + big area light",
                 SC::LightType::kLightCeilingAreaBig,
                 SC::MediumType::kClear);
    config.add_all_elements(cornell_box_with_white_back_wall);
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereBottomLeft, SC::MaterialType::kGlass));
    config.add_element(SC::Element(
        SC::GeometryType::kSmallSphereBottomRight, SC::MaterialType::kWaterMaterial, SC::MediumType::kWaterMedium));
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereTop, SC::MaterialType::kIce));
    config.add_element(SC::Element(SC::GeometryType::kVeryLargeSphere, SC::MaterialType::kMirror));
    configs.push_back(config);

    // 27
    config.reset("aniovergwi",
                 "three overlapping spheres (glass, water, ice) + weak anisoscattering + small area light",
                 SC::LightType::kLightCeilingAreaSmall,
                 SC::MediumType::kWeakWhiteAnisoScattering);
    config.add_all_elements(cornell_box_with_white_back_wall);
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereBottomLeft, SC::MaterialType::kGlass));
    config.add_element(SC::Element(
        SC::GeometryType::kSmallSphereBottomRight, SC::MaterialType::kWaterMaterial, SC::MediumType::kWaterMedium));
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereTop, SC::MaterialType::kIce));
    configs.push_back(config);

    // 28
    config.reset("overiwr",
                 "three overlapping glass spheres (iso, water, red absorb) + big area light",
                 SC::LightType::kLightCeilingAreaBig,
                 SC::MediumType::kClear);
    config.add_all_elements(cornell_box_with_white_back_wall);
    config.add_element(SC::Element(
        SC::GeometryType::kSmallSphereBottomLeft, SC::MaterialType::kGlass, SC::MediumType::kWhiteIsoScattering));
    config.add_element(
        SC::Element(SC::GeometryType::kSmallSphereBottomRight, SC::MaterialType::kGlass, SC::MediumType::kWaterMedium));
    config.add_element(
        SC::Element(SC::GeometryType::kSmallSphereTop, SC::MaterialType::kGlass, SC::MediumType::kRedAbsorbing));
    configs.push_back(config);

    // 29
    config.reset("isooveriwr",
                 "three overlapping spheres (iso, water, red absorb) + weak isoscattering + big area light",
                 SC::LightType::kLightCeilingAreaBig,
                 SC::MediumType::kWeakWhiteIsoScattering);
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereBottomLeft, SC::MediumType::kWhiteIsoScattering));
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereBottomRight, SC::MediumType::kWaterMedium));
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereTop, SC::MediumType::kRedAbsorbing));
    configs.push_back(config);

    // 30
    config.reset("woveriwr",
                 "three overlapping spheres (iso, water, red absorb) + one big water sphere + background light",
                 SC::LightType::kLightBackground,
                 SC::MediumType::kClear);
    config.add_all_elements(cornell_box_with_white_back_wall);
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereBottomLeft, SC::MediumType::kWhiteIsoScattering));
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereBottomRight, SC::MediumType::kWaterMedium));
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereTop, SC::MediumType::kRedAbsorbing));
    config.add_element(SC::Element(SC::GeometryType::kVeryLargeSphere, SC::MediumType::kWaterMedium));
    configs.push_back(config);

    // 31
    config.reset("backg",
                 "only background light and global medium",
                 SC::LightType::kLightBackground,
                 SC::MediumType::kLightReddishMedium);
    configs.push_back(config);

    // 32
    config.reset("backas",
                 "only background light and absorbing sphere",
                 SC::LightType::kLightBackground,
                 SC::MediumType::kClear);
    config.add_element(SC::Element(SC::GeometryType::kVeryLargeSphere, SC::MediumType::kRedAbsorbing));
    configs.push_back(config);

    // 33
    config.reset("backss",
                 "only background light and scattering sphere",
                 SC::LightType::kLightBackground,
                 SC::MediumType::kClear);
    config.add_element(SC::Element(SC::GeometryType::kVeryLargeSphere, SC::MediumType::kWhiteIsoScattering));
    configs.push_back(config);

    // 34
    config.reset("bigss",
                 "only big area light and scattering sphere",
                 SC::LightType::kLightCeilingAreaBig,
                 SC::MediumType::kClear);
    config.add_element(SC::Element(SC::GeometryType::kVeryLargeSphere, SC::MediumType::kWhiteIsoScattering));
    configs.push_back(config);

    // 35
    config.reset("smallsb",
                 "only small area light and scattering box",
                 SC::LightType::kLightCeilingAreaSmallDistant,
                 SC::MediumType::kClear);
    config.add_element(SC::Element(SC::GeometryType::kVeryLargeBox, SC::MediumType::kWeakYellowIsoScattering));
    configs.push_back(config);

    // 36
    config.reset("ssssuninfisov",
                 "specular small spheres + weak anisoscattering + sun",
                 SC::LightType::kLightSun,
                 SC::MediumType::kWeakWhiteIsoScattering);
    config.add_all_elements(cornell_box_with_glossy_floor);
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereLeft, SC::MaterialType::kMirror));
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereRight, SC::MaterialType::kGlass));
    configs.push_back(config);

    // 37
    config.reset("ssspointinfisov",
                 "specular small spheres + weak anisoscattering + point light",
                 SC::LightType::kLightCeilingPoint,
                 SC::MediumType::kWeakWhiteIsoScattering);
    config.add_all_elements(cornell_box_with_glossy_floor);
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereLeft, SC::MaterialType::kMirror));
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereRight, SC::MaterialType::kGlass));
    configs.push_back(config);

    // 38
    config.reset("sssbackinfisov",
                 "specular small spheres + weak anisoscattering + background light",
                 SC::LightType::kLightBackground,
                 SC::MediumType::kWeakWhiteIsoScattering);
    config.add_all_elements(cornell_box_with_glossy_floor);
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereLeft, SC::MaterialType::kMirror));
    config.add_element(SC::Element(SC::GeometryType::kSmallSphereRight, SC::MaterialType::kGlass));
    configs.push_back(config);

    // 39
    config.reset("glasssphbck",
                 "large glass medium sphere + background light",
                 SC::LightType::kLightBackground,
                 SC::MediumType::kClear);
    config.add_all_elements(cornell_box_with_glossy_floor);
    config.add_element(SC::Element(
        SC::GeometryType::kLargeSphereMiddle, SC::MaterialType::kGlass, SC::MediumType::kAbsorbingAnisoScattering));
    configs.push_back(config);

    // 40
    config.reset("glasssphbck",
                 "large glass medium sphere + large area light",
                 SC::LightType::kLightCeilingAreaBig,
                 SC::MediumType::kClear);
    config.add_all_elements(cornell_box_with_glossy_floor);
    config.add_element(SC::Element(
        SC::GeometryType::kLargeSphereMiddle, SC::MaterialType::kGlass, SC::MediumType::kAbsorbingAnisoScattering));
    configs.push_back(config);

    return configs;
}

std::string default_filename_without_scene(const Config& config)
{
    std::string filename("");
    std::ostringstream convert;

    // We add iterations count.
    if (config.iteration_count > 0) {
        filename += "_i";
        convert.str("");
        convert << std::setfill('0') << std::setw(2) << config.iteration_count;
        filename += convert.str();
    }

    // We add maximum path length.
    filename += "_l";
    convert.str("");
    convert << std::setfill('0') << std::setw(2) << config.max_path_length;
    filename += convert.str();

    // We add acronym of the used algorithm.
    filename += "_a";
    if (config.algorithm_flags & BPT) filename += "+bpt";
    if (config.algorithm_flags & SURF) filename += "+surf";
    if (config.algorithm_flags & PP3D) filename += "+pp3d";
    if (config.algorithm_flags & PB2D) filename += "+pb2d";
    if (config.algorithm_flags & BB1D) filename += "+bb1d";
    if (config.algorithm_flags & BP2D) filename += "+bp2d";

    // we add query beam type (if relevant)
    if ((config.algorithm_flags & PB2D) || (config.algorithm_flags & BB1D)) {
        filename += "_qbt";
        if (config.query_beam_type == SHORT_BEAM)
            filename += "S";
        else
            filename += "L";
    }

    // w add photon beam type (if relevant)
    if ((config.algorithm_flags & BP2D) || (config.algorithm_flags & BB1D)) {
        filename += "_pbt";
        if (config.photon_beam_type == SHORT_BEAM)
            filename += "S";
        else
            filename += "L";
    }

    // we add other args specified on command line
    filename += config.additional_args;

    // And it will be written as exr.
    filename += ".exr";

    return filename;
}

std::string default_filename(const int scene_id, const Config& config)
{
    std::string filename;

    // Name starts with a scene number.
    filename = "s";
    std::ostringstream convert;
    convert << std::setfill('0') << std::setw(2) << scene_id;
    filename += convert.str();

    // We add the rest.
    filename += default_filename_without_scene(config);

    return filename;
}

std::string default_filename(const std::string& scene_file_path, const Config& config)
{
    std::string filename;

    // Name starts with scene file name
    filename = "s-";
    size_t last_slash_pos = scene_file_path.find_last_of("\\/");
    size_t last_dot_pos = scene_file_path.find_last_of('.');
    filename += scene_file_path.substr(last_slash_pos + 1, last_dot_pos - last_slash_pos - 1);

    // We add the rest
    filename += default_filename_without_scene(config);

    return filename;
}

void print_help(const char* argv[])
{
    printf("\n");
    printf("Usage: %s <options>\n\n", argv[0]);
    printf("    Basic options:\n\n");

    printf("    -s  <scene_id> Selects the scene (default 0):\n");
    printf("        -1    user must supply additional argument which contains path to an obj scene file, paths "
           "seperated by a '/' \n");
    for (int i = 0; i < g_scene_configs.size(); i++)
        printf("        %-2d    %s\n", i, g_scene_configs[i].long_name.c_str());

    printf("\n");

    printf("    -a  <algorithm> Selects the rendering algorithm (default upbp_all):\n");
    printf("        %-10s  %s\n",
           "upbp_<tech>[+<tech>]*",
           "custom selection of techniques from upbp (tech=bpt|surf|pp3d|pb2d|bb1d|bp2d)");
    printf("        %-10s  %s\n", "upbp_all", "upbp_bpt+surf+pp3d+pb2d+bb1d");
    printf("\n");

    printf("    -l <length>    Maximum length of traced paths (default 10).\n");
    printf("    -t <sec>       Number of seconds to run the algorithm.\n");
    printf("    -i <iter>      Number of iterations to run the algorithm (default 1).\n");
    printf("    -o <name>      User specified output name, with extension .bmp or .exr (default .exr). The name can be "
           "prefixed with relative or absolute path but the path must exists.\n");
    printf("    -r <res>       Image resolution in format WIDTHxHEIGHT (default 256x256).\n");
    printf("    -assert        Enable GPU assertions.\n");
    printf("    -ioff <iter>   Offset the random number generator by this amount of iterations.\n");
    printf("    -log           Whether to output a log file.\n");
    printf("\n    Note: Time (-t) takes precedence over iterations (-i) if both are defined.\n");

    printf("\n    Radius options:\n\n");
    printf(
        "    -r_alpha <alpha>       Sets same radius reduction parameter for techniques surf, pp3d, pb2d and bb1d.\n");
    printf("    -r_alpha_bb1d <alpha>  Sets radius reduction parameter for technique bb1d (default 1, value of 1 "
           "implies no radius reduction).\n");
    printf("    -r_alpha_pb2d <alpha>  Sets radius reduction parameter for technique bp2d (default 1, value of 1 "
           "implies no radius reduction).\n");
    printf("    -r_alpha_pb2d <alpha>  Sets radius reduction parameter for technique pb2d (default 1, value of 1 "
           "implies no radius reduction).\n");
    printf("    -r_alpha_pp3d <alpha>  Sets radius reduction parameter for technique pp3d (default 1, value of 1 "
           "implies no radius reduction).\n");
    printf("    -r_alpha_surf <alpha>  Sets radius reduction parameter for technique surf (default 0.75, value of 1 "
           "implies no radius reduction).\n");
    printf("    -r_initial <initial>       Sets same initial radius for techniques surf, pp3d, pb2d and bb1d (if "
           "positive, absolute, if negative, relative to scene size).\n");
    printf("    -r_initial_bb1d <initial>  Sets initial radius for technique bb1d (default -0.001,  if positive, "
           "absolute, if negative, relative to scene size).\n");
    printf("    -r_initial_bp2d <initial>  Sets initial radius for technique bp2d (default -0.001,  if positive, "
           "absolute, if negative, relative to scene size).\n");
    printf("    -r_initial_pb2d <initial>  Sets initial radius for technique pb2d (default -0.001,  if positive, "
           "absolute, if negative, relative to scene size).\n");
    printf("    -r_initial_pp3d <initial>  Sets initial radius for technique pp3d (default -0.001,  if positive, "
           "absolute, if negative, relative to scene size).\n");
    printf("    -r_initial_surf <initial>  Sets initial radius for technique surf (default -0.0015, if positive, "
           "absolute, if negative, relative to scene size).\n");

    printf("\n    Light transport options:\n\n");
    printf("    -previous            Simulates \"previous work\" (paths ES*M(S|D|M)*L with camera paths stopping at "
           "the first M).\n");
    printf("    -compatible          Restricts traced paths to be comparable with the \"previous work\" (paths "
           "ES*M(S|D|M)*L).\n");
    printf("    -speconly            Traces only purely specular paths.\n");
    printf("    -ignorespec <option> Sets whether upbp will ignore fully specular paths from camera "
           "(0=no(default),1=yes).\n");

    printf("\n    Beams options:\n\n");
    printf("    -qbt <type>             Sets query beam type: S = uses short query beams,  L = uses long query beams "
           "(default).\n");
    printf("    -pbt <type>             Sets photon beam type: S = uses short photon beams (default), L = uses long "
           "photon beams.\n");
    printf("    -pbc <count>            First <count> traced light paths will generate photon beams (default -1, if "
           "positive, absolute, if negative, relative to total number of pixels).\n");

    printf("\n    Other options:\n\n");
    printf("    -continuous_output <iter_count>  Sets whether we should continuously output images (<iter_count> > 0 "
           "says output image once per <iter_count> iterations, 0(default) no cont. output).\n");
    printf("    -pcpi <path_count>               Light path count per iteration (default -1, if positive, absolute, if "
           "negative, relative to total number of traced light paths). Works only for vlt, pb2d, bb1d and upbp "
           "algorithms.\n");
    printf("    -sn <option>                     Whether to use shading normals: 0 = does not use shading normals, 1 = "
           "uses shading normals (default).\n");
}

void print_short_help(const char* argv[])
{
    printf("\n");
    printf("Usage: %s [ -s <scene_id> | -a <algorithm> | -l <path length> |\n", argv[0]);
    printf("           -t <time> | -i <iteration> | -o <output_name> ]\n\n");

    printf("    -s  Selects the scene (default 0):\n");
    printf("        -1    user must supply additional argument which contains path to an obj scene file, paths "
           "seperated by a '/' \n");
    for (int i = 0; i < 5; i++)
        printf("          %-2d    %s\n", i, g_scene_configs[i].long_name.c_str());
    printf("          5..%zu other predefined simple scenes, for complete list, please see full help (-hf)\n",
           g_scene_configs.size() - 1);

    printf("    -a  Selects the rendering algorithm (default upbp_all):\n");
    printf("        %-10s  %s\n",
           "upbp_<tech>[+<tech>]*",
           "custom selection of techniques from upbp (tech=bpt|surf|pp3d|pb2d|bb1d|bp2d)");
    printf("        %-10s  %s\n", "upbp_all", "upbp_bpt+surf+pp3d+pb2d+bb1d");
    printf("\n");

    printf("    -l  Maximum length of traced paths (default 10).\n");
    printf("    -t  Number of seconds to run the algorithm.\n");
    printf("    -i  Number of iterations to run the algorithm (default 1).\n");
    printf("    -o  User specified output name, with extension .bmp or .exr (default .exr). The name can be prefixed "
           "with relative or absolute path but the path must exists.\n");
    printf("\n    Note: Time (-t) takes precedence over iterations (-i) if both are defined.\n");
    printf("\n    For more options, please see full help (-hf)\n");
}

/// Splitting string by a given character.
std::vector<std::string> split_by_char(const std::string& string_to_split, char char_to_split_by)
{
    // http://stackoverflow.com/a/236803
    std::vector<std::string> elems;
    std::stringstream ss(string_to_split);
    std::string item;
    while (std::getline(ss, item, char_to_split_by)) {
        elems.push_back(item);
    }
    return elems;
}

int32_t parse_commandline(int argc, const char* argv[], Config& config)
{
    // Setting defaults.

    // by default use upbp_all
    config.algorithm_flags = BPT | SURF | PP3D | PB2D | BB1D;
    ;

    config.iteration_count = 1;
    config.max_time = -1.f;
    config.output_name = "";
    config.resolution = make_uint2(256, 256);

    config.max_path_length = 10;
    config.min_path_length = 0;

    config.path_count_per_iter = -1;

    config.query_beam_type = LONG_BEAM;
    config.photon_beam_type = SHORT_BEAM;

    config.surf_radius_initial = -0.0015f;
    config.surf_radius_alpha = 0.75f;

    config.pp3d_radius_initial = -0.001f;
    config.pp3d_radius_alpha = 1.0f;

    config.pb2d_radius_initial = -0.001f;
    config.pb2d_radius_alpha = 1.0f;

    config.bb1d_radius_initial = -0.001f;
    config.bb1d_radius_alpha = 1.0f;
    config.bb1d_used_light_subpath_count = -1;

    config.bp2d_radius_initial = -0.001f;
    config.bp2d_radius_alpha = 1.0f;

    config.continuous_output = 0;

    config.ignore_fully_spec_paths = false;

    config.gpu_assert = false;

    config.iteration_offset = 0;

    config.do_log = false;

    config.use_shading_normal = true;

    config.scene_id = 0;
    config.scene_obj_file = "";

    // string to append to saved file
    std::ostringstream additional_args;

    // To deal with options priorities.
    bool r_alpha_surf_set = false;
    bool r_alpha_pp3d_set = false;
    bool r_alpha_pb2d_set = false;
    bool r_alpha_bb1d_set = false;
    bool r_alpha_bp2d_set = false;
    bool r_init_surf_set = false;
    bool r_init_pp3d_set = false;
    bool r_init_pb2d_set = false;
    bool r_init_bb1d_set = false;
    bool r_init_bp2d_set = false;

    // Load arguments.
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);

        // Print help string (at any position).
        if (arg == "-h" || arg == "--help" || arg == "/?") {
            print_short_help(argv);
            return 1;
        } else if (arg == "-hf" || arg == "--help_full" || arg == "/??") {
            print_help(argv);
            return 1;
        }

        if (arg[0] != '-') // all our commands start with -
        {
            continue;
        }

        /** basic options */

        else if (arg == "-s") {
            // scene to load
            if (++i == argc) {
                fprintf(stderr, "missing argument of -s option, please see help (-h)");
                return -1;
            }

            std::istringstream iss(argv[i]);
            iss >> config.scene_id;

            if (iss.fail() || config.scene_id >= (int)g_scene_configs.size()) {
                fprintf(stderr, "invalid argument of -s option, please see help (-h)");
                return -1;
            } else if (config.scene_id == 3 || config.scene_id == 4 || config.scene_id == 9 || config.scene_id == 10) {
                fprintf(stderr, "scenes 3, 4, 9, and 10 cannot be rendered with UPBP, please see help (-h)");
                return -1;
            }

            if (config.scene_id == -1) {
                // try to load obj
                if (++i == argc) {
                    fprintf(stderr, "missing file name argument of -s option, please see help (-h)");
                    return -1;
                }

                config.scene_obj_file = argv[i];
            }
        } else if (arg == "-a") {
            // algorithm to use
            if (++i == argc) {
                fprintf(stderr, "missing argument of -a option, please see help (-h)");
                return -1;
            }

            std::string alg(argv[i]);

            if (alg == "upbp_all") {
                // note: BP2D is not part of upbp_all
                config.algorithm_flags |= BPT | SURF | PP3D | PB2D | BB1D;
            } else if (alg.size() >= 8 && alg.substr(0, 4) == "upbp" && (alg[4] == '+' || alg[4] == '_')) {
                config.algorithm_flags = 0;

                if (alg[4] == '+') config.algorithm_flags |= BPT | SURF;

                // split the string after upbp_ by '+'
                std::vector<std::string> techniques = split_by_char(alg.substr(5), '+');

                for (auto i = techniques.cbegin(); i != techniques.cend(); ++i) {
                    if (*i == "bpt") {
                        config.algorithm_flags |= BPT;
                    } else if (*i == "surf") {
                        config.algorithm_flags |= SURF;
                    } else if (*i == "pp3d") {
                        config.algorithm_flags |= PP3D;
                    } else if (*i == "pb2d") {
                        config.algorithm_flags |= PB2D;
                    } else if (*i == "bb1d") {
                        config.algorithm_flags |= BB1D;
                    } else if (*i == "bp2d") {
                        config.algorithm_flags |= BP2D;
                    } else {
                        fprintf(stderr, "invalid argument of -a option, please see help (-h)");
                        return -1;
                    }
                }
            }
        } else if (arg == "-l") {
            // maximum path length
            if (++i == argc) {
                fprintf(stderr, "missing argument of -l option, please see help (-h)");
                return -1;
            }

            std::istringstream iss(argv[i]);
            iss >> config.max_path_length;

            if (iss.fail() || config.max_path_length < 1) {
                fprintf(stderr, "invalid argument of -l option, please see help (-h)");
                return -1;
            }
        } else if (arg == "-t") {
            // number of seconds to run
            if (++i == argc) {
                fprintf(stderr, "missing argument of -t option, please see help (-h)");
                return -1;
            }

            std::istringstream iss(argv[i]);
            iss >> config.max_time;

            if (iss.fail() || config.max_time < 0) {
                fprintf(stderr, "invalid argument of -t option, please see help (-h)");
                return -1;
            }

            additional_args << "_t" << argv[i];
        } else if (arg == "-i") {
            // number of iterations to run
            if (++i == argc) {
                fprintf(stderr, "missing argument of -i option, please see help (-h)");
                return -1;
            }

            std::istringstream iss(argv[i]);
            iss >> config.iteration_count;

            if (iss.fail() || config.iteration_count < 1) {
                fprintf(stderr, "invalid argument of -i option, please see help (-h)");
                return -1;
            }
        } else if (arg == "-o") {
            // output name
            if (++i == argc) {
                fprintf(stderr, "missing argument of -o option, please see help (-h)");
                return -1;
            }

            config.output_name = argv[i];

            if (config.output_name.length() == 0) {
                fprintf(stderr, "invalid argument of -o option, please see help (-h)");
                return -1;
            }
        } else if (arg == "-r") {
            // resolution
            if (++i == argc) {
                fprintf(stderr, "missing argument of -r option, please see help (-hf)");
                return -1;
            }

            int w = -1, h = -1;
            sscanf(argv[i], "%dx%d", &w, &h);
            if (w <= 0 || h <= 0) {
                fprintf(stderr, "invalid argument of -r option, please see help (-hf)");
                return -1;
            }
            config.resolution = make_uint2(w, h);

            additional_args << "_r" << argv[i];
        }

        /** radius options */

        else if (arg == "-r_alpha") {
            // radius reduction factor
            if (++i == argc) {
                fprintf(stderr, "missing argument of -r_alpha option, please see help (-hf)");
                return -1;
            }

            float alpha;
            sscanf(argv[i], "%f", &alpha);
            if (alpha <= 0.0f) {
                fprintf(stderr, "invalid argument of -r_alpha option, please see help (-hf)");
                return -1;
            }

            if (!r_alpha_surf_set) config.surf_radius_alpha = alpha;
            if (!r_alpha_pp3d_set) config.pp3d_radius_alpha = alpha;
            if (!r_alpha_pb2d_set) config.pb2d_radius_alpha = alpha;
            if (!r_alpha_bb1d_set) config.bb1d_radius_alpha = alpha;
            if (!r_alpha_bp2d_set) config.bp2d_radius_alpha = alpha;

            additional_args << "_ralpha" << argv[i];
        } else if (arg == "-r_alpha_surf") {
            // radius reduction factor for surface photon mapping
            if (++i == argc) {
                fprintf(stderr, "missing argument of -r_alpha_surf option, please see help (-hf)");
                return -1;
            }

            sscanf(argv[i], "%f", &config.surf_radius_alpha);
            if (config.surf_radius_alpha <= 0.0f) {
                fprintf(stderr, "invalid argument of -r_alpha_surf option, please see help (-hf)");
                return -1;
            }

            r_alpha_surf_set = true;
            additional_args << "_ralphasurf" << argv[i];
        } else if (arg == "-r_alpha_pp3d") {
            // radius reduction factor for PP3D
            if (++i == argc) {
                fprintf(stderr, "missing argument of -r_alpha_pp3d option, please see help (-hf)");
                return -1;
            }

            sscanf(argv[i], "%f", &config.pp3d_radius_alpha);
            if (config.pp3d_radius_alpha <= 0.0f) {
                fprintf(stderr, "invalid argument of -r_alpha_pp3d option, please see help (-hf)");
                return -1;
            }

            r_alpha_pp3d_set = true;
            additional_args << "_ralphapp3d" << argv[i];
        } else if (arg == "-r_alpha_pb2d") {
            // radius reduction factor for PB2D
            if (++i == argc) {
                fprintf(stderr, "missing argument of -r_alpha_pb2d option, please see help (-hf)");
                return -1;
            }

            sscanf(argv[i], "%f", &config.pb2d_radius_alpha);
            if (config.pb2d_radius_alpha <= 0.0f) {
                fprintf(stderr, "invalid argument of -r_alpha_pb2d option, please see help (-hf)");
                return -1;
            }

            r_alpha_pb2d_set = true;
            additional_args << "_ralphapb2d" << argv[i];
        } else if (arg == "-r_alpha_bb1d") {
            // radius reduction factor for BB1D
            if (++i == argc) {
                fprintf(stderr, "missing argument of -r_alpha_bb1d option, please see help (-hf)");
                return -1;
            }

            sscanf(argv[i], "%f", &config.bb1d_radius_alpha);
            if (config.bb1d_radius_alpha <= 0.0f) {
                fprintf(stderr, "invalid argument of -r_alpha_bb1d option, please see help (-hf)");
                return -1;
            }

            r_alpha_bb1d_set = true;
            additional_args << "_ralphabb1d" << argv[i];
        } else if (arg == "-r_alpha_bp2d") {
            // radius reduction factor for BP2D
            if (++i == argc) {
                fprintf(stderr, "missing argument of -r_alpha_bp2d option, please see help (-hf)");
                return -1;
            }

            sscanf(argv[i], "%f", &config.bp2d_radius_alpha);
            if (config.bp2d_radius_alpha <= 0.0f) {
                fprintf(stderr, "invalid argument of -r_alpha_bp2d option, please see help (-hf)");
                return -1;
            }

            r_alpha_bp2d_set = true;
            additional_args << "_ralphabp2d" << argv[i];
        } else if (arg == "-r_initial") {
            // initial radius
            if (++i == argc) {
                fprintf(stderr, "missing argument of -r_initial option, please see help (-hf)");
                return -1;
            }

            float init;
            sscanf(argv[i], "%f", &init);
            if (init == 0.0f) {
                fprintf(stderr, "invalid argument of -r_initial option, please see help (-hf)");
                return -1;
            }

            if (!r_init_surf_set) config.surf_radius_initial = init;
            if (!r_init_pp3d_set) config.pp3d_radius_initial = init;
            if (!r_init_pb2d_set) config.pb2d_radius_initial = init;
            if (!r_init_bb1d_set) config.bb1d_radius_initial = init;
            if (!r_init_bp2d_set) config.bp2d_radius_initial = init;

            additional_args << "_rinit" << argv[i];
        } else if (arg == "-r_initial_surf") {
            // initial radius for surface photon mapping
            if (++i == argc) {
                fprintf(stderr,
                        "missing argument of -r_initial_surf option, "
                        "please see help (-hf)");
                return -1;
            }

            sscanf(argv[i], "%f", &config.surf_radius_initial);
            if (config.surf_radius_initial == 0.0f) {
                fprintf(stderr,
                        "invalid argument of -r_initial_surf option, "
                        "please see help (-hf)");
                return -1;
            }

            r_init_surf_set = true;
            additional_args << "_rinitsurf" << argv[i];
        } else if (arg == "-r_initial_pp3d") {
            // initial radius for PP3D
            if (++i == argc) {
                fprintf(stderr,
                        "missing argument of -r_initial_pp3d option, "
                        "please see help (-hf)");
                return -1;
            }

            sscanf(argv[i], "%f", &config.pp3d_radius_initial);
            if (config.pp3d_radius_initial == 0.0f) {
                fprintf(stderr,
                        "invalid argument of -r_initial_pp3d option, "
                        "please see help (-hf)");
                return -1;
            }

            r_init_pp3d_set = true;
            additional_args << "_rinitpp3d" << argv[i];
        } else if (arg == "-r_initial_pb2d") {
            // initial radius for PB2D
            if (++i == argc) {
                fprintf(stderr,
                        "missing argument of -r_initial_pb2d option, "
                        "please see help (-hf)");
                return -1;
            }

            sscanf(argv[i], "%f", &config.pb2d_radius_initial);
            if (config.pb2d_radius_initial == 0.0f) {
                fprintf(stderr,
                        "invalid argument of -r_initial_pb2d option, "
                        "please see help (-hf)");
                return -1;
            }

            r_init_pb2d_set = true;
            additional_args << "_rinitpb2d" << argv[i];
        } else if (arg == "-r_initial_bb1d") {
            // initial radius for BB1D
            if (++i == argc) {
                fprintf(stderr,
                        "missing argument of -r_initial_bb1d option, "
                        "please see help (-hf)");
                return -1;
            }

            sscanf(argv[i], "%f", &config.bb1d_radius_initial);
            if (config.bb1d_radius_initial == 0.0f) {
                fprintf(stderr,
                        "invalid argument of -r_initial_bb1d option, "
                        "please see help (-hf)");
                return -1;
            }

            r_init_bb1d_set = true;
            additional_args << "_rinitbb1d" << argv[i];
        } else if (arg == "-r_initial_bp2d") {
            // initial radius for BP2D
            if (++i == argc) {
                fprintf(stderr,
                        "missing argument of -r_initial_bp2d option, "
                        "please see help (-hf)");
                return -1;
            }

            sscanf(argv[i], "%f", &config.bp2d_radius_initial);
            if (config.bp2d_radius_initial == 0.0f) {
                fprintf(stderr,
                        "invalid argument of -r_initial_bp2d option, "
                        "please see help (-hf)");
                return -1;
            }

            r_init_bp2d_set = true;
            additional_args << "_rinitbp2d" << argv[i];
        }

        /** light transport options */

        else if (arg == "-previous") {
            // previous mode
            config.algorithm_flags |= PREVIOUS;

            additional_args << "_prev";
        } else if (arg == "-compatible") {
            // compatible mode
            config.algorithm_flags |= COMPATIBLE;

            additional_args << "_comp";
        } else if (arg == "-speconly") // only specular paths
        {
            config.algorithm_flags |= SPECULAR_ONLY;

            additional_args << "_speconly";
        } else if (arg == "-ignorespec") {
            // upbp will ignore fully specular paths
            if (++i == argc) {
                fprintf(stderr, "missing argument of -ignorespec option, please see help (-hf)");
                return -1;
            }

            std::string option(argv[i]);
            if (option == "1") {
                config.ignore_fully_spec_paths = true;
            } else if (option == "0") {
                config.ignore_fully_spec_paths = false;
            } else {
                fprintf(stderr, "invalid argument of -ignorespec option, please see help (-hf)");
                return -1;
            }
        }

        /** beams options */

        else if (arg == "-qbt") {
            // query beam type
            if (++i == argc) {
                fprintf(stderr, "missing argument of -qbt option, please see help (-hf)");
                return -1;
            }

            char a[2];
            sscanf(argv[i], "%s", a);
            if (a[0] == 'L' || a[0] == 'l') {
                config.query_beam_type = LONG_BEAM;
            } else {
                config.query_beam_type = SHORT_BEAM;
            }
        } else if (arg == "-pbt") {
            // photon beam type
            if (++i == argc) {
                fprintf(stderr, "missing argument of -pbt option, please see help (-hf)");
                return -1;
            }

            char a[2];
            sscanf(argv[i], "%s", a);
            if (a[0] == 'L' || a[0] == 'l') {
                config.photon_beam_type = LONG_BEAM;
            } else {
                config.photon_beam_type = SHORT_BEAM;
            }
        } else if (arg == "-pbc") {
            // paths with beams count
            if (++i == argc) {
                fprintf(stderr, "missing argument of -pbc option, please see help (-hf)");
                return -1;
            }

            sscanf(argv[i], "%f", &config.bb1d_used_light_subpath_count);
            if (config.bb1d_used_light_subpath_count == 0.0f) {
                fprintf(stderr, "invalid argument of -pbc option, please see help (-hf)");
                return -1;
            }

            additional_args << "_pbc" << argv[i];
        }

        /** other options */

        else if (arg == "-continuous_output") {
            // output image each x iterations
            if (++i == argc) {
                fprintf(stderr, "missing argument of -continuous_output option, please see help (-hf)");
                return -1;
            }

            std::istringstream iss(argv[i]);
            iss >> config.continuous_output;

            if (iss.fail()) {
                fprintf(stderr, "invalid argument of -continuous_output option, please see help (-hf)");
                return -1;
            }
        } else if (arg == "-pcpi") {
            // path count per iteration
            if (++i == argc) {
                fprintf(stderr, "missing argument of -pcpi option, please see help (-hf)");
                return -1;
            }

            sscanf(argv[i], "%f", &config.path_count_per_iter);
            if (std::floor(config.path_count_per_iter) == 0.0f) {
                fprintf(stderr, "invalid argument of -pcpi option, please see help (-hf)");
                return -1;
            }

            additional_args << "_pcpi" << argv[i];
        } else if (arg == "-sn") {
            // surface normals
            if (++i == argc) {
                fprintf(stderr, "missing argument of -sn option, please see help (-hf)");
                return -1;
            }

            int a;
            sscanf(argv[i], "%d", &a);
            if (a == 0) {
                config.use_shading_normal = false;
            } else {
                config.use_shading_normal = true;
            }

            additional_args << "_sn" << argv[i];
        }

        /** GPU options */

        else if (arg == "-assert") {
            config.gpu_assert = true;

            additional_args << "_assert";
        } else if (arg == "-ioff") {
            if (++i == argc) {
                fprintf(stderr, "missing argument of -ioff option, please see help (-hf)");
                return -1;
            }

            std::istringstream iss(argv[i]);
            iss >> config.iteration_offset;

            if (iss.fail() || config.iteration_offset < 0) {
                fprintf(stderr, "invalid argument of -ioff option, please see help (-hf)");
                return -1;
            }
        } else if (arg == "-log") {
            // enable logging
            config.do_log = true;

            additional_args << "_log";
        } else {
            fprintf(stderr, "invalid option \"%s\", please see help (-hf)", arg.c_str());
            return -1;
        }
    }

    config.additional_args = additional_args.str();

    // Computing path counts if specified negative, i.e. relative.
    if (config.path_count_per_iter < 0) {
        config.path_count_per_iter =
            std::floor(-config.path_count_per_iter * config.resolution.x * config.resolution.y);
    } else {
        config.path_count_per_iter = std::floor(config.path_count_per_iter);
    }

    // If no output name is chosen, create a default one.
    if (config.output_name.length() == 0) {
        if (config.scene_id > -1) {
            config.output_name = default_filename(config.scene_id, config);
        } else {
            config.output_name = default_filename(config.scene_obj_file, config);
        }
    }

    // Check if output name has valid extension (.bmp or .exr) and if not add .exr
    std::string extension = "";

    // must be at least 1 character before .exr
    if (config.output_name.length() > 4) {
        extension = config.output_name.substr(config.output_name.length() - 4, 4);
    }

    if (extension != ".bmp" && extension != ".exr") {
        config.output_name += ".exr";
    }

    return 0;
}

std::vector<SceneConfig> g_scene_configs = init_scene_configs();
