// Copyright (C) 2021, Nol Moonen
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
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

void compare(const unsigned char* d0, const unsigned char* d1, int x, int y, int n)
{
    size_t diff = 0;
    for (int i = 0; i < x * y * n; i++) {
        unsigned char d = d0[i] > d1[i] ? d0[i] - d1[i] : d1[i] - d0[i];
        diff += d;
    }

    if (diff > 0) {
        float error = ((float)diff * 100.f) / (float)(x * y * n * 255);
        printf("not equal! error is %f %%\n", error);

        return;
    }

    printf("equal!\n");
}

void mse(const unsigned char* d0, const unsigned char* d1, int x, int y, int n)
{
    float val = 0.f;
    for (int i = 0; i < x * y * n; i++) {
        unsigned char d = d0[i] > d1[i] ? d0[i] - d1[i] : d1[i] - d0[i];
        float diff = (float)d;
        val += diff * diff;
    }
    val /= (float)n;

    printf("mse=%.5e\n", val);
}

int main(int argc, char** argv)
{
    if (argc != 4) {
        printf("usage: <first file> <second file> <mse>\n");
        return EXIT_FAILURE;
    }

    char* ptr;
    int mode = strtol(argv[3], &ptr, 10);
    if (mode == 0) {
        // compare mode
    } else if (mode == 1) {
        // mse mode
    } else {
        printf("unknown mode\n");
        return EXIT_FAILURE;
    }

    printf("comparing \"%s\" to \"%s\": ", argv[1], argv[2]);

    // read first file
    int x0, y0, n0;
    unsigned char* d0 = stbi_load(argv[1], &x0, &y0, &n0, 0);
    if (!d0) {
        fprintf(stderr, "cannot open \"%s\"\n", argv[1]);

        return EXIT_FAILURE;
    }

    // read second file
    int x1, y1, n1;
    unsigned char* d1 = stbi_load(argv[2], &x1, &y1, &n1, 0);
    if (!d1) {
        fprintf(stderr, "cannot open \"%s\"\n", argv[2]);

        stbi_image_free(d0);
        return EXIT_FAILURE;
    }

    if (x0 != x1 || y0 != y1 || n0 != n1) {
        fprintf(stderr, "width, height, or channel count not equal\n");

        stbi_image_free(d1);
        stbi_image_free(d0);

        return EXIT_FAILURE;
    }

    // compare files, (x0 == x1 && y0 == y1 && n0 == n1) at this point

    if (mode == 0) {
        compare(d0, d1, x0, y0, n0);
    } else { // if (mode == 1)
        mse(d0, d1, x0, y0, n0);
    }

    // free data
    stbi_image_free(d1);
    stbi_image_free(d0);

    return EXIT_SUCCESS;
}
