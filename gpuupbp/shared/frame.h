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

#ifndef SHARED_FRAME_H
#define SHARED_FRAME_H

#include "material.h"
#include "matrix.h"

/// Frame of reference.
struct Frame {
    SHARED_INLINE SHARED_HOSTDEVICE Frame()
        : x(make_float3(1.f, 0.f, 0.f)), y(make_float3(0.f, 1.f, 0.f)), z(make_float3(0.f, 0.f, 1.f))
    {
    }

    SHARED_INLINE SHARED_HOSTDEVICE Frame(const float3& x, const float3& y, const float3& z) : x(x), y(y), z(z) {}

    /// Initializes a frame of reference using a direction.
    SHARED_INLINE SHARED_HOSTDEVICE void set_from_z(const float3& pz)
    {
        float3 tmp_z = z = normalize(pz);
        float3 tmp_x = (fabsf(tmp_z.x) > .99f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
        y = normalize(cross(tmp_z, tmp_x));
        x = cross(y, tmp_z);
    }

    SHARED_INLINE SHARED_HOSTDEVICE float3 to_world(const float3& a) const { return x * a.x + y * a.y + z * a.z; }

    SHARED_INLINE SHARED_HOSTDEVICE float3 to_local(const float3& a) const
    {
        return make_float3(dot(a, x), dot(a, y), dot(a, z));
    }

    SHARED_INLINE SHARED_HOSTDEVICE const float3& binormal() const { return x; }

    SHARED_INLINE SHARED_HOSTDEVICE const float3& tangent() const { return y; }

    SHARED_INLINE SHARED_HOSTDEVICE const float3& normal() const { return z; }

    float3 x, y, z;
};

#endif // SHARED_FRAME_H
