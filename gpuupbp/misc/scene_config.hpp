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

#ifndef SCENE_SCENE_CONFIG_HPP
#define SCENE_SCENE_CONFIG_HPP

#include <vector>

struct SceneConfig {
    enum LightType {
        kLightCeilingAreaBig,
        kLightCeilingAreaSmall,
        kLightCeilingAreaSmallDistant,
        kLightCeilingPoint,
        kLightFacingAreaSmall,
        kLightFacingPoint,
        kLightSun,
        kLightBackground,
        kLightNone,
        kLightsCount
    };

    enum GeometryType {
        kLeftWall,
        kRightWall,
        kFrontFacingFrontWall,
        kBackFacingFrontWall,
        kBackWall,
        kCeiling,
        kFloor,
        kLargeSphereMiddle,
        kSmallSphereLeft,
        kSmallSphereRight,
        kSmallSphereBottomLeft,
        kSmallSphereBottomRight,
        kSmallSphereTop,
        kVeryLargeSphere,
        kVeryLargeBox,
        kGeometryCount
    };

    enum MaterialType {
        kNoMaterial = -1,
        kWaterMaterial,
        kIce,
        kGlass,
        kDiffuseRed,
        kDiffuseGreen,
        kDiffuseBlue,
        kDiffuseWhite,
        kGlossyWhite,
        kMirror,
        kLight,
        kMaterialsCount
    };

    enum MediumType {
        kNoMedium = -1,
        kClear,
        kWaterMedium,
        kWhiteIsoScattering, // color in name is color of result
        kWhiteAnisoScattering,
        kWeakWhiteIsoScattering,
        kWeakYellowIsoScattering,
        kWeakWhiteAnisoScattering,
        kRedAbsorbing,
        kYellowEmitting,
        kYellowGreenAbsorbingEmittingAndScattering,
        kBlueAbsorbingAndEmitting,
        kLightReddishMedium,
        kAbsorbingAnisoScattering,
        kMediaCount
    };

    struct Element {
        Element(GeometryType geom, MaterialType mat, MediumType med) { setup(geom, mat, med); }

        Element(GeometryType geom, MaterialType mat) { setup(geom, mat, MediumType::kNoMedium); }

        Element(GeometryType geom, MediumType med) { setup(geom, MaterialType::kNoMaterial, med); }

        Element(GeometryType geom) { setup(geom, MaterialType::kDiffuseWhite, MediumType::kNoMedium); }

        GeometryType geom;
        MaterialType mat;
        MediumType med;

        void setup(GeometryType p_geom, MaterialType p_mat, MediumType p_med)
        {
            geom = p_geom;
            mat = p_mat;
            med = p_med;
        }
    };

    SceneConfig(std::string s_name, std::string l_name, LightType l, MediumType glob_med)
    {
        reset(s_name, l_name, l, glob_med);
    }

    void reset(std::string s_name, std::string l_name, LightType l, MediumType glob_med)
    {
        short_name = s_name;
        long_name = l_name;
        light = l;
        global_medium = glob_med;
        elements.clear();
    }

    void add_element(Element element) { elements.push_back(element); }

    void add_all_elements(std::vector<Element> aElements)
    {
        elements.insert(elements.end(), aElements.begin(), aElements.end());
    }

    std::string short_name;
    std::string long_name;
    LightType light;
    MediumType global_medium;
    std::vector<Element> elements;
};

#endif // SCENE_SCENE_CONFIG_HPP
