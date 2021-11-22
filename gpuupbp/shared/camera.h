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

#ifndef SHARED_CAMERA_H
#define SHARED_CAMERA_H

#include "matrix.h"

struct Camera {
    /// Camera origin.
    float3 origin;
    /// Camera look direction.
    float3 direction;
    /// Resolution of the framebuffer.
    float2 resolution;
    /// Matrix for transforming a position in raster space to world space.
    sutil::Matrix4x4 raster_to_world;
    /// Matrix for transforming a position in world space to raster space.
    sutil::Matrix4x4 world_to_raster;
    /// Distance from the camera to the image plane.
    float image_plane_dist;
    /// If the camera is not in the global medium, it is enclosed with a material.
    /// Material id associated with the camera. -1 if none.
    int mat_id;
    /// Medium id associated with the camera. -1 if none.
    int med_id;
};

#endif // SHARED_CAMERA_H
