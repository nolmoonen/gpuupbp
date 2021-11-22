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

#include "defs.cuh"

#include <optix.h>

extern "C" __global__ void __exception__upbp()
{
    const int exc = optixGetExceptionCode();

    printf("[ 5][            USER]: ");
    switch (exc) {
    case 0:
        printf("assertion failed\n");
        break;
    case ERR_BOUNDARY_STACK:
        printf("boundary_stack too small\n");
        break;
    case ERR_INTERSECTION_ARRAY:
        printf("intersection_array too small\n");
        break;
    case ERR_VOL_SEGMENT_ARRAY:
        printf("vol_segment_array or vol_lite_segment_array too small\n");
        break;
    case ERR_INTERSECTION_STACK:
        printf("intersection_stack too small\n");
        break;
    default:
        printf("unknown exception\n");
        break;
    }
}
