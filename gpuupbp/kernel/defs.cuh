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

#ifndef KERNEL_DEFS_CUH
#define KERNEL_DEFS_CUH

#include <cuda_runtime.h> // for printf
#include <optix.h>

/// Methods may return a status code.
/// 0 on success, an integer with a positive status code on failure.
/// The failure is not graceful and indicates the sample should be discarded.
typedef unsigned int gpuupbp_status;

/// No error.
#define RET_SUCCESS 0
/// Tried to push to {boundary_stack} but it was full.
#define ERR_BOUNDARY_STACK 1
/// Tried to push to {intersection_array} but it was full.
#define ERR_INTERSECTION_ARRAY 2
/// Tried to push to {vol_segment_array} or {vol_lite_segment_array}
/// but it was full.
#define ERR_VOL_SEGMENT_ARRAY 3
/// Tried to push to {intersection_stack} but it was full.
#define ERR_INTERSECTION_STACK 4

/// Added to reduce memory usage of volume segments.
/// Tested for all .obj scenes and predefined scenes.
/// Bottleneck is predefined scene 30 (9).
#define MAX_VOLUME_SEGMENTS 10 // +1 for stability

/// Increased for sanmiguel with length 50.
/// (was 30 for all except sanmiguel and bistro)
/// Bottleneck is stilllife (21) works for the 1380-1390 bug
/// 22 and 23 do no for the next bug in stilllife, 30 does.
/// Set from 30 to 40 for bistro and sanmiguel testing.
/// 20 < suntemple (+ assertion somewhere)
/// Maximum amount of intersections with real faces (with priority higher
/// than -1) that may occur for any ray.
#define MAX_INTERSECTIONS 10

/// 20 < bistro <= 40
#define MAX_STACK_ELEMENT 20

/// Maximum amount of intersections with imaginary faces that may occur for
/// a ray between two real faces. Bottleneck is predef 30.
#define MAX_IMAGINARY_INTERSECTIONS 8

/// Maximum value of a 32-bits unsigned integer.
// #define UINT_MAX 0xffffffff

/// If exceptions are enabled, throws an exception. Else, return the error code.
/// In functions where an error may occur, this macro is used to throw an
/// exception if these are enabled. The caller of the throwing function must
/// gracefully handle the return code (this means to discard the sample).
#define RETURN_WITH_ERR(code)                                                                                          \
    do {                                                                                                               \
        optixThrowException(code);                                                                                     \
        return code;                                                                                                   \
    } while (0)

///// Assertion, throws an exception if condition is false.
///// Should compile to non-statement if exceptions are disabled.
#define GPU_ASSERT(cond)                                                                                               \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            optixThrowException(0);                                                                                    \
        }                                                                                                              \
    } while (0)
//#define GPU_ASSERT(cond)                                                       \
//    do                                                                         \
//    {                                                                          \
//        if( !(cond) )                                                          \
//        {                                                                      \
//            printf("%s (%d): %s\n", __FILE__, __LINE__, #cond);                \
//            char *c = nullptr; *c = 0;                                         \
//        }                                                                      \
//    } while( 0 )

enum IntersectOptions {
    /// Sample scattering distance in the medium
    /// (if not set, always returns the nearest surface as the scattering point)
    SAMPLE_VOLUME_SCATTERING = (1u << 1),
    OCCLUSION_TEST = (1u << 2)
};

enum RaySamplingFlags { ORIGIN_IN_MEDIUM = (1u << 0), END_IN_MEDIUM = (1u << 1) };

/// An epsilon value (small non-zero positive number).
/// Used in ray vs geometry intersection tests.
#define EPS_RAY 1e-3f

/// Epsilon value for making point queries for optixTrace.
#define EPS_LAUNCH 1e-5f

/// Epsilon value used in BSDF computations.
#define EPS_COSINE 1e-6f

/// Epsilon value used in BSDF computations.
#define EPS_PHONG 1e-3f

__forceinline__ __device__ bool is_black_or_negative(const float3& a) { return a.x <= 0.f && a.y <= 0.f && a.z <= 0.f; }

__forceinline__ __device__ bool is_positive(const float& a) { return a > 1e-20f; }

__forceinline__ __device__ bool is_positive(const float3& a)
{
    return is_positive(a.x) && is_positive(a.y) && is_positive(a.z);
}

__forceinline__ __device__ float clampf(float x, float l, float u) { return l > (x > u ? u : x) ? l : (x > u ? u : x); }

#endif // KERNEL_DEFS_CUH
