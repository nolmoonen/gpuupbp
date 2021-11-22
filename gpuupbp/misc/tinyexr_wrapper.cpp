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

#include "tinyexr_wrapper.hpp"
#include "../host/env_map.hpp"

#define TINYEXR_IMPLEMENTATION

#include <tinyexr.h>

void tinyexr_wrapper::save_exr(const char* filename, float3* color, int res_x, int res_y)
{
    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 3;

    std::vector<float> images[3];
    images[0].resize(res_x * res_y);
    images[1].resize(res_x * res_y);
    images[2].resize(res_x * res_y);

    // Split RGBRGBRGB... into R, G and B layer
    for (int i = 0; i < res_x * res_y; i++) {
        images[0][i] = color[i].x;
        images[1][i] = color[i].y;
        images[2][i] = color[i].z;
    }

    float* image_ptr[3];
    image_ptr[0] = &(images[2].at(0)); // B
    image_ptr[1] = &(images[1].at(0)); // G
    image_ptr[2] = &(images[0].at(0)); // R

    image.images = (unsigned char**)image_ptr;
    image.width = res_x;
    image.height = res_y;

    header.num_channels = 3;
    header.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * header.num_channels);
    // must be (A)BGR order, since most of EXR viewers expect this channel order
    strncpy(header.channels[0].name, "B", 255);
    header.channels[0].name[strlen("B")] = '\0';
    strncpy(header.channels[1].name, "G", 255);
    header.channels[1].name[strlen("G")] = '\0';
    strncpy(header.channels[2].name, "R", 255);
    header.channels[2].name[strlen("R")] = '\0';

    header.pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
    header.requested_pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
    for (int i = 0; i < header.num_channels; i++) {
        // pixel type of input image
        header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
        // pixel type of output image to be stored in .exr
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF;
    }

    const char* err = NULL;
    int ret = SaveEXRImageToFile(&image, &header, filename, &err);
    if (ret != TINYEXR_SUCCESS) {
        fprintf(stderr, "tinyexr: %s\n", err);
        FreeEXRErrorMessage(err);
        exit(2);
    }

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);
}

Image tinyexr_wrapper::load_image(const char* filename, float rotate, float scale)
{
    Image image{};

    float* out; // width * height * RGBA
    int width;
    int height;
    const char* err = nullptr;

    int ret = LoadEXR(&out, &width, &height, filename, &err);

    if (ret != TINYEXR_SUCCESS) {
        if (err) {
            fprintf(stderr, "tinyexr: %s\n", err);
            FreeEXRErrorMessage(err);
            exit(2);
        }

        return image;
    }

    init_image(image, width, height);

    int c = 0;
    int iRot = (int)(rotate * width);
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int x = i + iRot;
            if (x >= width) x -= width;
            element_at(image, x, j).x = out[4 * c + 0] * scale;
            element_at(image, x, j).y = out[4 * c + 1] * scale;
            element_at(image, x, j).z = out[4 * c + 2] * scale;
            c++;
        }
    }

    free(out);

    return image;
}
