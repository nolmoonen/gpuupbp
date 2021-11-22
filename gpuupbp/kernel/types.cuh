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
#ifndef KERNEL_TYPES_CUH
#define KERNEL_TYPES_CUH

#include "../shared/frame.h"
#include "../shared/shared_defs.h"
#include "../shared/vec_math.h"
#include "path_weight.cuh"
#include "pstack.cuh"
#include "sarray.cuh"
#include "vector_types.h"

/// Segment of a ray in a single medium.
struct VolSegment {
    __forceinline__ VolSegment() = default;

    __forceinline__ __device__ VolSegment(const float3& attenuation,
                                          const float3& emission,
                                          float dist_min,
                                          float dist_max,
                                          float ray_sample_pdf_for,
                                          float ray_sample_pdf_rev,
                                          int med_id)
        : attenuation(attenuation), emission(emission), dist_min(dist_min), dist_max(dist_max),
          ray_sample_pdf_for(ray_sample_pdf_for), ray_sample_pdf_rev(ray_sample_pdf_rev), med_id(med_id)
    {
    }

    /// Attenuation caused by this segment (not divided by PDF).
    float3 attenuation;
    /// Emission coming from this segment (neither attenuated, nor divided by PDF).
    float3 emission;
    /// Distance of the segment begin from the ray origin.
    float dist_min;
    /// Distance of the segment end from the ray origin.
    float dist_max;
    /// If scattering occurred in this medium: PDF of having samples in this segment; otherwise: PDF of passing through
    /// the entire medium.
    float ray_sample_pdf_for;
    /// Similar to mRaySamplePdf but in a reverse direction.
    float ray_sample_pdf_rev;
    /// ID of the medium in this segment.
    int med_id;
};

/// Same as vol_segment, but when the PDFs are not needed.
struct VolLiteSegment {
    __forceinline__ VolLiteSegment() = default;

    __forceinline__ __device__ VolLiteSegment(float dist_min, float dist_max, int med_id)
        : dist_min(dist_min), dist_max(dist_max), med_id(med_id)
    {
    }

    /// Distance of the segment begin from the ray origin.
    float dist_min;
    /// Distance of the segment end from the ray origin.
    float dist_max;
    /// ID of the medium in this segment.
    int med_id;
};

// todo use unsigned ints for mat and med indices, and compress this struct
struct Boundary {
    int mat_id;
    int med_id;

    __forceinline__ Boundary() = default;

    __forceinline__ __device__ Boundary(int mat_id, int med_id) : mat_id(mat_id), med_id(med_id) {}

    __forceinline__ __device__ bool operator==(const Boundary& elem) const
    {
        return elem.med_id == med_id && elem.mat_id == mat_id;
    }

    __forceinline__ __device__ bool operator!=(const Boundary& elem) const
    {
        return elem.med_id != med_id || elem.mat_id != mat_id;
    }
};

struct Intersection {
    /// Distance to the closest intersection on a ray (serves as ray.tmax).
    float dist;
    /// ID of intersected material, -1 indicates scattering event inside medium.
    int mat_id;
    /// ID of interacting medium or medium behind the hit surface, -1 means that
    /// crossing the hit surface does not affect medium.
    int med_id;
    /// ID of intersected light, -1 means none.
    int light_id;
    /// normal at the intersection.
    float3 normal;
    /// Whether the ray enters geometry at this intersection (cosine of its
    /// direction and the normal is negative).
    bool enter;
    /// Shading normal.
    float3 shading_normal;

    __forceinline__ __device__ Intersection(float max_dist = INFTY) : dist(max_dist)
    {
        dist = max_dist;
        mat_id = -1;
        med_id = -1;
        light_id = -1;
        normal = make_float3(0.f);
        enter = false;
    }

    /// True if this intersection represents a scattering event inside a medium.
    __forceinline__ __device__ bool is_in_medium() { return mat_id < 0; }

    /// True if this intersection represents an intersection with geometry.
    __forceinline__ __device__ bool is_on_surface() const { return mat_id >= 0; }

    /// Whether properties of this intersection have valid values.
    __forceinline__ __device__ bool is_valid() const
    {
        return dist > 0.f &&
               ((mat_id < 0 && med_id >= 0) ||
                (mat_id >= 0 && !(normal.x == 0.f && normal.y == 0.f && normal.z == 0.f))) &&
               (light_id < 0 || mat_id >= 0);
    }
};

/// Compressed intersection struct, for when normals etc. are not needed.
struct LiteIntersection {
    float dist;
    int mat_id;
    int med_id;
    int enter;

    __forceinline__ LiteIntersection() = default;

    __forceinline__ __device__ LiteIntersection(float dist, int mat_id, int med_id, bool enter)
        : dist(dist), mat_id(mat_id), med_id(med_id), enter(enter)
    {
    }

    __forceinline__ __device__ int get_mat_id() const { return mat_id; }

    __forceinline__ __device__ int get_med_id() const { return med_id; }

    __forceinline__ __device__ bool get_enter() const { return enter; }
};

typedef SArray<LiteIntersection, MAX_INTERSECTIONS> IntersectionArray;
typedef SArray<VolSegment, MAX_VOLUME_SEGMENTS> VolSegmentArray;
typedef SArray<VolLiteSegment, MAX_VOLUME_SEGMENTS> VolLiteSegmentArray;
typedef PStack<Boundary, int, MAX_STACK_ELEMENT> BoundaryStack;
typedef PStack<LiteIntersection, float, MAX_IMAGINARY_INTERSECTIONS> IntersectionStack;

/// Maintains data for vertex i, which is initialized as 0 by making a call to
/// generate_light_sample or generate_camera_sample, i is incremented in
/// sample_scattering.
struct SubpathState {
    /// Stack of crossed boundaries.
    BoundaryStack stack;
    /// y_i or z_i.
    float3 origin;
    /// y_i->y_{i+1} or z_i->z_{i+1}.
    float3 direction;
    /// Path throughput.
    float3 throughput;
    /// Number of path segments, including this.
    unsigned int path_length;
    /// Whether the light that generated this sample was infinite.
    bool is_infinite_light;
    /// All scattering events so far were specular. Only used for camera side.
    bool is_specular_path;
    /// The weights.
    StateWeights weights;
};

#endif // KERNEL_TYPES_CUH
