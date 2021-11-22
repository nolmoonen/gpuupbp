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

#ifndef HOST_MEDIUM_HPP
#define HOST_MEDIUM_HPP

#include "../shared/medium.h"
#include "../shared/vec_math_ext.h"

#include <vector_types.h>

#include <cassert>

#define MEDIUM_SURVIVAL_PROB .8f

inline void init_homogeneous_medium(Medium& med,
                                    const float3& absorption_coef,
                                    const float3& emission_coef,
                                    const float3& scattering_coef,
                                    const float continuation_prob,
                                    const float mean_cosine = 0.f)
{
    assert(continuation_prob >= 0 && continuation_prob <= 1);
    assert(mean_cosine >= -1 && mean_cosine <= 1);
    med.continuation_prob = continuation_prob;
    med.mean_cosine = mean_cosine;

    med.absorption_coeff = abs(absorption_coef);
    med.emission_coeff = abs(emission_coef);
    med.scattering_coeff = abs(scattering_coef);

    med.attenuation_coeff = med.absorption_coeff + med.scattering_coeff;

    med.min_positive_attenuation_coeff_comp = med.attenuation_coeff.x;
    med.min_positive_attenuation_coeff_comp_idx = 0;
    if (med.attenuation_coeff.y > 0.f && (med.min_positive_attenuation_coeff_comp == 0 ||
                                          med.attenuation_coeff.y < med.min_positive_attenuation_coeff_comp)) {
        med.min_positive_attenuation_coeff_comp = med.attenuation_coeff.y;
        med.min_positive_attenuation_coeff_comp_idx = 1;
    }
    if (med.attenuation_coeff.z > 0.f && (med.min_positive_attenuation_coeff_comp == 0 ||
                                          med.attenuation_coeff.z < med.min_positive_attenuation_coeff_comp)) {
        med.min_positive_attenuation_coeff_comp = med.attenuation_coeff.z;
        med.min_positive_attenuation_coeff_comp_idx = 2;
    }

    if (med.scattering_coeff.x > 0 || med.scattering_coeff.y > 0 || med.scattering_coeff.z > 0) {
        med.has_scattering = true;
    } else {
        med.has_scattering = false;
    }

    if (med.attenuation_coeff.x > 0 || med.attenuation_coeff.y > 0 || med.attenuation_coeff.z > 0) {
        // has attenuation
        med.has_attenuation = true;
        med.max_beam_length = -logf(1e-9f) / fmaxf(med.attenuation_coeff);
        med.mean_free_path = 1.0f / fmaxf(med.attenuation_coeff);
    } else {
        // has no attenuation
        med.has_attenuation = false;
        med.max_beam_length = INFTY;
        med.mean_free_path = INFTY;
    }
}

inline void init_medium_clear(Medium& med)
{
    // automatically sets max beam length and mean free path to INFINITY
    init_homogeneous_medium(med, make_float3(0.0f), make_float3(0.0f), make_float3(0.0f), 1.0f);
}

#endif // HOST_MEDIUM_HPP
