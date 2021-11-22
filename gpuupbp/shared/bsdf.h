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

#ifndef SHARED_BSDF_H
#define SHARED_BSDF_H

#include "medium.h"

struct BSDF {
    enum BSDFEvent {
        NONE = 0,
        DIFFUSE = (1u << 0),
        PHONG = (1u << 1),
        REFLECT = (1u << 2),
        REFRACT = (1u << 3),
        SCATTER = (1u << 4), // media
        SPECULAR = (REFLECT | REFRACT),
        NON_SPECULAR = (DIFFUSE | PHONG | SCATTER),
        ALL = (SPECULAR | NON_SPECULAR)
    };

    enum BSDFDirection { FROM_CAMERA, FROM_LIGHT };

    enum PdfDir { FORWARD, REVERSE };

    enum FlagBits {
        /// Set if BSDF is in medium, not on surface.
        IN_MEDIUM = (1u << 0),
        /// Set if ray comes from light, not from camera.
        DIR_FROM_LIGHT = (1u << 1),
        /// Set when BSDF is valid.
        VALID = (1u << 2),
        /// Set when material/medium is purely specular.
        DELTA = (1u << 3)
    };

    /// Data only for surface.
    struct SrfData {
        /// Material of the surface in the BSDF location.
        const Material* mat_ptr;
        /// Fresnel reflection coefficient (for glass).
        float reflect_coeff;
        /// Incoming (fixed) direction, in local.
        float3 local_dir_fix;
        /// Sampling probabilities.
        float diff_prob;
        /// Sampling probabilities.
        float phong_prob;
        /// Sampling probabilities.
        float refl_prob;
        /// Sampling probabilities.
        float refr_prob;
        /// Surface geometry normal (in local).
        float3 geometry_normal;
        /// Cosine of angle between the incoming direction and the geometry
        /// normal.
        float cos_theta_fix_geom;
        /// Relative index of refraction.
        float ior;
    };

    /// Data only for medium.
    struct MedData {
        /// Medium in the BSDF location.
        const Medium* med_ptr;
        /// Coefficient of anisotropy.
        float mean_cosine;
        /// Incoming (fixed) direction, in world.
        float3 world_dir_fix;
        /// Medium scattering coefficient.
        float3 scatter_coef;
        /// Whether the medium scattering coefficient is positive.
        bool scatter_coef_is_pos;
    };

    union {
        /// Data if BSDF is on surface.
        SrfData srf_data;
        /// Data is BSDF is in medium.
        MedData med_data;
    } data;

    /// Local frame of reference.
    Frame frame;
    /// Russian roulette probability
    float continuation_prob;
    /// BSDF flags.
    int flags;
};

#endif // SHARED_BSDF_H
