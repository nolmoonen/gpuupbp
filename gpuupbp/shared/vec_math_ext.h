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

#ifndef SHARED_SUTIL_EXT_H
#define SHARED_SUTIL_EXT_H

/**
 * Extension utility for sutil library.
 */

#include "../shared/shared_defs.h"
#include "matrix.h"
#include "vec_math.h"

/// Return the component-wise absolute value of a vector.
SHARED_INLINE SHARED_HOSTDEVICE float3 abs(const float3& a) { return make_float3(fabsf(a.x), fabsf(a.y), fabsf(a.z)); }

/// Returns the square of a vector: the dot product with itself.
SHARED_INLINE SHARED_HOSTDEVICE float square(const float3& a) { return dot(a, a); }

namespace sutil {
/// Returns a matrix for perspective transformation. Fov in degrees.
SHARED_INLINE SHARED_HOSTDEVICE Matrix<4, 4> perspective(float fov, float near_, float far_)
{
    Matrix<4, 4> mat = Matrix<4, 4>::identity();
    float* m = mat.getData();

    // Camera points towards -z.  0 < near < far.
    // Matrix maps z range [-near, -far] to [-1, 1], after homogeneous division.
    float f = 1.f / (tanf(fov * PI_F / 360.f));
    float d = 1.f / (near_ - far_);

    m[0 * 4 + 0] = f;
    m[0 * 4 + 1] = 0.f;
    m[0 * 4 + 2] = 0.f;
    m[0 * 4 + 3] = 0.f;

    m[1 * 4 + 0] = 0.f;
    m[1 * 4 + 1] = -f;
    m[1 * 4 + 2] = 0.f;
    m[1 * 4 + 3] = 0.f;

    m[2 * 4 + 0] = 0.f;
    m[2 * 4 + 1] = 0.f;
    m[2 * 4 + 2] = (near_ + far_) * d;
    m[2 * 4 + 3] = 2.f * near_ * far_ * d;

    m[3 * 4 + 0] = 0.f;
    m[3 * 4 + 1] = 0.f;
    m[3 * 4 + 2] = -1.f;
    m[3 * 4 + 3] = 0.f;

    return Matrix<4, 4>(m);
}

/// Transforms the given point. That is performs matrix * [vec, 1]^T
/// and homogeneous division.
SHARED_INLINE SHARED_HOSTDEVICE float3 transform_point(const Matrix<4, 4>& mat, const float3& vec)
{
    float w = mat.getRow(3).w;

    w += mat.getRow(3).x * vec.x;
    w += mat.getRow(3).y * vec.y;
    w += mat.getRow(3).z * vec.z;

    const float invW = 1.f / w;

    float3 res = make_float3(0.f);

    res.x = mat.getRow(0).w;
    res.x += vec.x * mat.getRow(0).x;
    res.x += vec.y * mat.getRow(0).y;
    res.x += vec.z * mat.getRow(0).z;
    res.x *= invW;

    res.y = mat.getRow(1).w;
    res.y += vec.x * mat.getRow(1).x;
    res.y += vec.y * mat.getRow(1).y;
    res.y += vec.z * mat.getRow(1).z;
    res.y *= invW;

    res.z = mat.getRow(2).w;
    res.z += vec.x * mat.getRow(2).x;
    res.z += vec.y * mat.getRow(2).y;
    res.z += vec.z * mat.getRow(2).z;
    res.z *= invW;

    return res;
}
} // namespace sutil

#endif // SHARED_SUTIL_EXT_H
