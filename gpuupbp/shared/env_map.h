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

#ifndef SHARED_ENV_MAP_H
#define SHARED_ENV_MAP_H

#include "preprocessor.h"

#include <vector_types.h>

struct Distribution1D {
    float func_int;
    // todo remove this parameter as it is the same for all 1d distributions?
    int count;
};

struct Distribution2D {
    /// One distribution for every unit of height.
    Distribution1D* conditional_v;
    Distribution1D marginal;
    /// Contains the func (width) and cdf (width + 1) for (height) pConditionalV
    /// 1d distributions (in that order) and for pMarginal.
    /// Total size is ((height + 1) * (2 * width + 1))
    float* data;
    /// Width
    int nu;
    /// Height
    int nv;
};

struct Image {
    float3* data;
    int width;
    int height;

    SHARED_INLINE SHARED_HOSTDEVICE float3& element_at(int x, int y) const { return data[width * y + x]; }

    SHARED_INLINE SHARED_HOSTDEVICE float3& element_at(int idx) const { return data[idx]; }
};

struct EnvMap {
    /// The environment map.
    Image img;
    /// Environment map converted to 2D distribution.
    Distribution2D distribution;
    /// PDF normalization factor.
    float norm;
};

#endif // SHARED_ENV_MAP_H
