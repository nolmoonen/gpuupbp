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

#ifndef KERNEL_INTERSECTION_CUH
#define KERNEL_INTERSECTION_CUH

#include "../shared/vec_math.h"

/// Trimmed version of {test_intersection_beam_beam} for intersection program.
__forceinline__ __device__ bool
beam_beam(const float3& ro1, const float3& rd1, float rt1, const float3& ro2, const float3& rd2, float rt2, float ra2)
{
    const float3 d1d2c = cross(rd1, rd2);
    const float sin_theta_sqr = dot(d1d2c, d1d2c);
    const float ad = dot((ro2 - ro1), d1d2c);
    if (ad * ad >= ra2 * sin_theta_sqr) return false;
    const float d1d2 = dot(rd1, rd2);
    const float d1d2_sqr = d1d2 * d1d2;
    const float d1d2_sqr_sub1 = d1d2_sqr - 1.0f;
    if (d1d2_sqr_sub1 < 1e-5f && d1d2_sqr_sub1 > -1e-5f) return false;
    const float d1o1 = dot(rd1, ro1);
    const float d1o2 = dot(rd1, ro2);
    const float oT1 = (d1o1 - d1o2 - d1d2 * (dot(rd2, ro1) - dot(rd2, ro2))) / d1d2_sqr_sub1;
    if (oT1 <= 0.f || oT1 >= rt1) return false;
    const float oT2 = (oT1 + d1o1 - d1o2) / d1d2;
    if (oT2 <= 0.f || oT2 >= rt2) return false;

    return true;
}

/// Trimmed version of {test_intersection_bre} for in intersection program.
__forceinline__ __device__ bool point_beam(const float3& ro, const float3& rd, float rt, const float3& ce, float ra2)
{
    // point to segment origin
    const float3 eo = ce - ro;
    // dist of point projected on line formed by segment
    const float t = dot(eo, rd);
    // if dist falls out of line segment, discard
    if (t <= 0.f || t >= rt) return false;
    // line segment perp to original line segment, to point
    const float3 tmp = ro + rd * t - ce;
    // squared length of tmp line
    const float h = dot(tmp, tmp);
    return h <= ra2;
}

/// Simple point-point test, ra2 is squared radius.
__forceinline__ __device__ bool point_point(const float3& p0, const float3& p1, float ra2)
{
    const float3 dist = p0 - p1;
    return dot(dist, dist) <= ra2;
}

/// Test intersection between a ray and a 'photon disc'.
///
/// The photon disc is specified by the position photon_pos and squared radius
/// photon_rad_sqr. The disc is assumed to face the ray (i.e. the disc plane is
/// perpendicular to the ray). Intersections are reported only in the interval
/// [min_t, max_t)  (i.e. includes min_t but excludes max_t).
///
/// Return true is an intersection is found, false otherwise. If an intersection
/// is found, isect_dist is set to the intersection distance along the ray, and
/// isect_rad_sqr is set to the square of the distance of the intersection point
/// from the photon location.
__forceinline__ __device__ bool test_intersection_bre(float& isect_dist,
                                                      float& isect_rad_sqr,
                                                      const Ray& query_ray,
                                                      const float min_t,
                                                      const float max_t,
                                                      const float3& photon_pos,
                                                      const float photon_rad_sqr)
{
    const float3 ray_orig_to_photon = photon_pos - query_ray.origin;

    const float dist = dot(ray_orig_to_photon, query_ray.direction);

    if (dist > min_t && dist < max_t) {
        const float3 isect_rad = query_ray.origin + query_ray.direction * dist - photon_pos;
        const float isect_rad_sqr_ = dot(isect_rad, isect_rad);
        if (isect_rad_sqr_ <= photon_rad_sqr) {
            isect_dist = dist;
            isect_rad_sqr = isect_rad_sqr_;
            return true;
        }
    }
    return false;
}

/// Intersection of ray with beam.
///
/// The query ray is specified by (o1,d1) [min_t1, max_t1). The photon beam
/// segment is specified by (o2,d2) [min_t2, max_t2). Only intersection within
/// distance sqrt(max_dist_sqr) are reported.
///
/// Returns true is an intersection is found, false otherwise. If an
/// intersection is found, the following output parameters are set:
///  oDistance  distance between the two lines
///  oSinTheta  sine of the angle between the two lines
///  oT1        ray parameter of the closest point along the query ray
///  oT2        ray parameter of the closest point along the photon beam segment
__forceinline__ __device__ bool test_intersection_beam_beam(const float3& o1,
                                                            const float3& d1,
                                                            const float min_t1,
                                                            const float max_t1,
                                                            const float3& o2,
                                                            const float3& d2,
                                                            const float min_t2,
                                                            const float max_t2,
                                                            const float max_dist_sqr,
                                                            float& distance,
                                                            float& sin_theta,
                                                            float& t1,
                                                            float& t2)
{
    const float3 d1d2c = cross(d1, d2);
    // square of the sine between the two lines (||cross(d1, d2)|| = sin_theta).
    const float sin_theta_sqr = dot(d1d2c, d1d2c);

    const float ad = dot((o2 - o1), d1d2c);

    // lines too far apart
    if (ad * ad >= max_dist_sqr * sin_theta_sqr) return false;

    // cosine between the two lines
    const float d1d2 = dot(d1, d2);
    const float d1d2_sqr = d1d2 * d1d2;
    const float d1d2_sqr_minus_1 = d1d2_sqr - 1.f;

    // parallel lines?
    if (d1d2_sqr_minus_1 < 1e-5f && d1d2_sqr_minus_1 > -1e-5f) return false;

    const float d1o1 = dot(d1, o1);
    const float d1o2 = dot(d1, o2);

    t1 = (d1o1 - d1o2 - d1d2 * (dot(d2, o1) - dot(d2, o2))) / d1d2_sqr_minus_1;

    // out of range on ray 1
    if (t1 <= min_t1 || t1 >= max_t1) return false;

    t2 = (t1 + d1o1 - d1o2) / d1d2;
    // out of range on ray 2
    if (t2 <= min_t2 || t2 >= max_t2 || isnan(t2)) return false;

    sin_theta = sqrtf(sin_theta_sqr);

    distance = fabsf(ad) / sin_theta;

    // found an intersection
    return true;
}

#endif // KERNEL_INTERSECTION_CUH
