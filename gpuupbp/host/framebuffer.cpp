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

#include "framebuffer.hpp"
#include "../misc/tinyexr_wrapper.hpp"
#include "../shared/vec_math.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

void init(Framebuffer& fb, const uint2& resolution)
{
    fb.resolution = resolution;
    fb.color = static_cast<float3*>(calloc(resolution.x * resolution.y, sizeof(float3)));
}

void clear(Framebuffer& fb) { memset(fb.color, 0, sizeof(float3) * fb.resolution.x * fb.resolution.y); }

void cleanup(Framebuffer& fb) { free(fb.color); }

void add_color(Framebuffer& fb, const float2& sample, const float3& color)
{
    if (sample.x < 0 || sample.x >= fb.resolution.x) return;
    if (sample.y < 0 || sample.y >= fb.resolution.y) return;

    int x = int(sample.x);
    int y = int(sample.y);

    fb.color[x + y * fb.resolution.x] += color;
}

void add_framebuffer(Framebuffer& fb, const Framebuffer& fb_other)
{
    for (size_t i = 0; i < fb.resolution.x * fb.resolution.y; i++) {
        fb.color[i] += fb_other.color[i];
    }
}

void add_scaled(Framebuffer& fb, const Framebuffer& fb_other, float scale)
{
    for (size_t i = 0; i < fb.resolution.x * fb.resolution.y; i++) {
        fb.color[i] += fb_other.color[i] * scale;
    }
}

void scale(Framebuffer& fb, float scale)
{
    for (size_t i = 0; i < fb.resolution.x * fb.resolution.y; i++) {
        fb.color[i] *= make_float3(scale);
    }
}

void save_ppm(Framebuffer& fb, const char* filename, float gamma)
{
    const float inv_gamma = 1.f / gamma;

    std::ofstream ppm(filename);
    ppm << "P3" << std::endl;
    ppm << fb.resolution.x << " " << fb.resolution.y << std::endl;
    ppm << "255" << std::endl;

    for (uint32_t y = 0; y < fb.resolution.y; y++) {
        for (uint32_t x = 0; x < fb.resolution.x; x++) {
            float3* ptr = &fb.color[x + y * fb.resolution.x];
            int r = int(std::pow(ptr->x, inv_gamma) * 255.f);
            int g = int(std::pow(ptr->y, inv_gamma) * 255.f);
            int b = int(std::pow(ptr->z, inv_gamma) * 255.f);

            ppm << std::min(255, std::max(0, r)) << " " << std::min(255, std::max(0, g)) << " "
                << std::min(255, std::max(0, b)) << " ";
        }

        ppm << std::endl;
    }
}

void save_pfm(Framebuffer& fb, const char* filename)
{
    std::ofstream ppm(filename, std::ios::binary);
    ppm << "PF" << std::endl;
    ppm << fb.resolution.x << " " << fb.resolution.y << std::endl;
    ppm << "-1" << std::endl;

    ppm.write(reinterpret_cast<const char*>(fb.color), fb.resolution.x * fb.resolution.y * sizeof(float3));
}

void save(Framebuffer& fb, const char* filename, float gamma) { save(fb, std::string(filename), gamma); }

void save(Framebuffer& fb, const std::string& filename, float gamma)
{
    std::string extension = filename.substr(filename.length() - 3, 3);
    if (extension == "bmp") {
        save_bmp(fb, filename.c_str(), gamma /*gamma*/);
    } else if (extension == "hdr") {
        save_hdr(fb, filename.c_str());
    } else if (extension == "exr") {
        tinyexr_wrapper::save_exr(filename.c_str(), fb.color, fb.resolution.x, fb.resolution.y);
    } else {
        std::cerr << "Error: used unknown extension " << extension << std::endl;
        exit(2);
    }
}

void save_bmp(Framebuffer& fb, const char* filename, float gamma)
{
    struct bmp_header {
        /// Size of file in bytes.
        uint32_t file_size;
        /// 2x 2 reserved bytes.
        uint32_t reserved_01;
        /// Offset in bytes where data can be found (54).
        uint32_t data_offset;
        /// 40B.
        uint32_t header_size;
        /// Width in pixels.
        int width;
        /// Height in pixels.
        int height;
        /// Must be 1.
        short color_plates;
        /// We use 24bpp.
        short bits_per_pixel;
        /// We use BI_RGB ~ 0, uncompressed.
        uint32_t compression;
        /// mWidth x mHeight x 3B.
        uint32_t image_size;
        /// Pixels per meter (75dpi ~ 2953ppm).
        uint32_t horiz_res;
        /// Pixels per meter (75dpi ~ 2953ppm).
        uint32_t vert_res;
        /// Not using palette - 0.
        uint32_t palette_colors;
        /// 0 - all are important.
        uint32_t important_colors;
    };

    // fixme does not account for four-byte width alignment
    std::ofstream bmp(filename, std::ios::binary);
    bmp_header header;
    bmp.write("BM", 2);
    header.file_size = uint32_t(sizeof(bmp_header) + 2) + fb.resolution.x * fb.resolution.y * 3;
    header.reserved_01 = 0;
    header.data_offset = uint32_t(sizeof(bmp_header) + 2);
    header.header_size = 40;
    header.width = fb.resolution.x;
    header.height = fb.resolution.y;
    header.color_plates = 1;
    header.bits_per_pixel = 24;
    header.compression = 0;
    header.image_size = fb.resolution.x * fb.resolution.y * 3;
    header.horiz_res = 2953;
    header.vert_res = 2953;
    header.palette_colors = 0;
    header.important_colors = 0;

    bmp.write((char*)&header, sizeof(header));

    const float inv_gamma = 1.f / gamma;
    for (uint32_t y = 0; y < fb.resolution.y; y++) {
        for (uint32_t x = 0; x < fb.resolution.x; x++) {
            // BMP is stored from bottom up.
            const float3& rgbF = fb.color[x + (fb.resolution.y - y - 1) * fb.resolution.x];
            typedef unsigned char byte;
            float gamma_bgr[3];
            gamma_bgr[0] = std::pow(rgbF.z, inv_gamma) * 255.f;
            gamma_bgr[1] = std::pow(rgbF.y, inv_gamma) * 255.f;
            gamma_bgr[2] = std::pow(rgbF.x, inv_gamma) * 255.f;

            byte bgrB[3];
            bgrB[0] = byte(std::min(255.f, std::max(0.f, gamma_bgr[0])));
            bgrB[1] = byte(std::min(255.f, std::max(0.f, gamma_bgr[1])));
            bgrB[2] = byte(std::min(255.f, std::max(0.f, gamma_bgr[2])));

            bmp.write((char*)&bgrB, sizeof(bgrB));
        }
    }
}

void save_hdr(Framebuffer& fb, const char* filename)
{
    std::ofstream hdr(filename, std::ios::binary);

    hdr << "#?RADIANCE" << '\n';
    hdr << "# SmallUPBP" << '\n';
    hdr << "FORMAT=32-bit_rle_rgbe" << '\n' << '\n';
    hdr << "-Y " << fb.resolution.y << " +X " << fb.resolution.x << '\n';

    for (uint32_t y = 0; y < fb.resolution.y; y++) {
        for (uint32_t x = 0; x < fb.resolution.x; x++) {
            typedef unsigned char byte;
            byte rgbe[4] = {0, 0, 0, 0};

            const float3& rgb_f = fb.color[x + y * fb.resolution.x];
            float v = std::max(rgb_f.x, std::max(rgb_f.y, rgb_f.z));

            if (v >= 1e-32f) {
                int e;
                v = float(frexp(v, &e) * 256.f / v);
                rgbe[0] = byte(rgb_f.x * v);
                rgbe[1] = byte(rgb_f.y * v);
                rgbe[2] = byte(rgb_f.z * v);
                rgbe[3] = byte(e + 128);
            }

            hdr.write((char*)&rgbe[0], 4);
        }
    }
}
