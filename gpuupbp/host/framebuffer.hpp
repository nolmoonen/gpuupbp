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

#ifndef HOST_FRAMEBUFFER_HPP
#define HOST_FRAMEBUFFER_HPP

#include "../shared/framebuffer.h"

#include <vector_types.h>

#include <cstdint>
#include <string>

/// Setups the framebuffer.
void init(Framebuffer& fb, const uint2& resolution);

/// Clears the framebuffer.
void clear(Framebuffer& fb);

/// Cleans up the framebuffer.
void cleanup(Framebuffer& fb);

/// Adds the given color to pixel containing the given position.
void add_color(Framebuffer& fb, const float2& sample, const float3& color);

/// Adds other framebuffer.
void add_framebuffer(Framebuffer& fb, const Framebuffer& fb_other);

/// Adds other framebuffer scaled.
void add_scaled(Framebuffer& fb, const Framebuffer& fb_other, float scale);

/// Scales values in this framebuffer.
void scale(Framebuffer& fb, float scale);

/// Saves this framebuffer as an image in PPM format.
void save_ppm(Framebuffer& fb, const char* filename, float gamma = 1.f);

/// Saves this framebuffer as an image in PFM format.
void save_pfm(Framebuffer& fb, const char* filename);

/// Saves this framebuffer as an image in a format corresponding to
/// the given file name (BMP, HDR, OpenEXR).
void save(Framebuffer& fb, const char* filename, float gamma = 2.2f);

/// Saves this framebuffer as an image in a format corresponding to
/// the given file name (BMP, HDR, OpenEXR).
void save(Framebuffer& fb, const std::string& filename, float gamma = 2.2f);

/// Saves this framebuffer as an image in BMP format.
void save_bmp(Framebuffer& fb, const char* filename, float gamma = 1.f);

/// Saves this framebuffer as an image in HDR format.
void save_hdr(Framebuffer& fb, const char* filename);

#endif // HOST_FRAMEBUFFER_HPP
