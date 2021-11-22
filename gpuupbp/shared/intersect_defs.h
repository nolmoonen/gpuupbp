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

#ifndef SHARED_INTERSECT_DEFS_H
#define SHARED_INTERSECT_DEFS_H

#include "scene.h"

// maximum number of bytes is 8
// (https://raytracing-docs.nvidia.com/optix7/guide/index.html#limits#limits)
enum GeomMask {
    GEOM_MASK_REAL = (1u << 0u),
    GEOM_MASK_IMAG = (1u << 1u),
    GEOM_MASK_BB1D = (1u << 2u),
    GEOM_MASK_POINT_MEDIUM = (1u << 3u),
    GEOM_MASK_PP2D = (1u << 4u),
    GEOM_MASK_BP2D = (1u << 5u)
};

enum RayTypeScene {
    /// Closest hit with real geometry.
    RAY_TYPE_SCENE_REAL = 0,
    /// All hits with imaginary geometry.
    RAY_TYPE_SCENE_IMAG,
    RAY_TYPE_COUNT
};

enum RayTypePDE { RAY_TYPE2_BB1D = 0, RAY_TYPE2_PB2D, RAY_TYPE2_PP2D, RAY_TYPE2_PP3D, RAY_TYPE2_BP2D, RAY_TYPE2_COUNT };

/// Data passed from host to ray generation program.
struct RayGenData {};

/// Data passed from host to miss program.
struct MissData {};

/// Data passed from host to hit group programs.
struct HitgroupData {
    Record gpu_mat;
};

/// Data passed from host to exception program.
struct ExceptionData {};

#endif // SHARED_INTERSECT_DEFS_H
