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

#ifndef KERNEL_INTERSECTOR_CUH
#define KERNEL_INTERSECTOR_CUH

#include "../shared/intersect_defs.h"
#include "../shared/shared_defs.h"
#include "../shared/vec_math.h"
#include "curand_kernel.h"
#include "medium.cuh"
#include "optix_util.cuh"
#include "params_def.cuh"
#include "rng.cuh"
#include "scene.cuh"
#include "types.cuh"

#include <optix.h>
#include <vector_types.h>

/// result.dist contains the maximum distance in the case of an occlusion test.
__forceinline__ __device__ gpuupbp_status intersect_(bool& hit,
                                                     const Ray& ray,
                                                     Intersection& result,
                                                     BoundaryStack& b_stack,
                                                     unsigned int options,
                                                     unsigned int ray_sampling_flags,
                                                     VolSegmentArray& volume_segments_to_isect,
                                                     VolLiteSegmentArray& volume_segments_all,
                                                     RNGState& rng)
{
    hit = false;
    bool sample_media = (options & SAMPLE_VOLUME_SCATTERING) != 0;
    bool test_occlusion = (options & OCCLUSION_TEST) != 0;

    // we cannot sample media if they are ignored, if we know that the ray has
    // already got through them
    if (test_occlusion) sample_media = false;

    Intersection tmp_result = result;
    float dist = 0.f;

    ////////////////////////////////////////////////////////////////////////////
    // Real intersection
    // We can compute this separately only because we assume that no medium
    // without a material has greater priority than or equal to any of media
    // with a material!
    // In other words, no medium without a material can be inside a medium with
    // a material (it will be ignored).
    // We also assume that no boundary of a medium without material intersects
    // any boundary of media with material.

    // crossed boundaries with low priority are stored in list and their
    // insertion in boundary stack is postponed until media interaction
    // is processed
    IntersectionArray hits_with_material;

    GPU_ASSERT(!b_stack.is_empty());
    int topPriority = b_stack.top_priority();

    // epsilon offset of the ray origin, zero in media
    const float first_eps = (ray_sampling_flags & ORIGIN_IN_MEDIUM) ? 0.f : EPS_RAY;

    // the current maximum ray distance
    float t_max;
    // maximum distance to intersection
    float miss_dist;
    if (test_occlusion) {
        // if testing occlusion, subtract an epsilon at the end in order not to
        // hit the target surface
        t_max = tmp_result.dist - EPS_RAY;
        miss_dist = tmp_result.dist;
    } else {
        t_max = INFTY;
        miss_dist = INFTY;
    }

    bool not_yet_real;

    BoundaryStack stack_copy = {};
    memcpy(&stack_copy, &b_stack, sizeof(b_stack));

    float t_min = first_eps;
    do {
        not_yet_real = false;

        // try to find intersection. we use origin epsilon offset for numerical
        // stable intersection
        tmp_result.dist = INFTY;
        {
            unsigned int u0, u1;
            pack_pointer(&tmp_result, u0, u1);
#ifdef DBG_CORE_UTIL
            // this trace call does not make recursive trace calls
            params.trace_count[optixGetLaunchIndex().x]++;
#endif
            optixTrace(params.handle,
                       ray.origin,
                       ray.direction,
                       t_min,
                       t_max,
                       0.f,
                       GEOM_MASK_REAL,
                       OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                       RAY_TYPE_SCENE_REAL,
                       RAY_TYPE_COUNT,
                       RAY_TYPE_SCENE_REAL,
                       u0,
                       u1);
        }
        // this test indicates whether a hit was found
        hit = tmp_result.dist != INFTY;

        if (hit) {
            // the actual distance of the intersection
            dist = tmp_result.dist;

            GPU_ASSERT(tmp_result.mat_id >= 0);

            int priority = get_medium_boundary_priority(params.scene, tmp_result.mat_id);

            Boundary top = stack_copy.top();

            if (tmp_result.med_id < 0) tmp_result.med_id = top.med_id;

            // check if it enters or leaves material with lower priority
            if (priority < topPriority && priority != THIN_WALL_PRIORITY) {
                Boundary element(tmp_result.mat_id, tmp_result.med_id);
                if (tmp_result.enter) {
                    if (!hits_with_material.push_back(LiteIntersection(
                            tmp_result.dist, tmp_result.mat_id, tmp_result.med_id, tmp_result.enter))) {
                        RETURN_WITH_ERR(ERR_INTERSECTION_ARRAY);
                    }
                    if (!stack_copy.push(element, priority)) {
                        RETURN_WITH_ERR(ERR_BOUNDARY_STACK);
                    }
                } else {
                    // ignore hit, if it was not in a stack. this 'if' is why
                    // we need to update (a copy of) the stack.
                    if (stack_copy.pop(element, priority)) {
                        if (!hits_with_material.push_back(LiteIntersection(
                                tmp_result.dist, tmp_result.mat_id, tmp_result.med_id, tmp_result.enter))) {
                            RETURN_WITH_ERR(ERR_INTERSECTION_ARRAY);
                        }
                    }
                }

                not_yet_real = true;
                // prepare the ray for the next step: apply epsilon at start
                t_min = dist + EPS_RAY;
                if (test_occlusion) {
                    // subtract an epsilon and the end if occlusion testing
                    t_max = miss_dist - dist - EPS_RAY;
                } else {
                    t_max = INFTY;
                }
            }
        }
        // if the occlusion search distance is still positive
    } while (not_yet_real && t_max > t_min);

    if (!hit || not_yet_real) {
        hit = false;
        dist = miss_dist;
    }

    result = tmp_result;
    result.dist = dist;

    ////////////////////////////////////////////////////////////////////////////
    // Media interaction

    bool scattering_occurred = false;

    IntersectionStack* imaginary_isects = nullptr;

    // If we are not already behind any material and there is enough space
    // before the first material intersection
    // we can check intersection with the media without a material
    // (the same assumptions as above apply here)
    if (!is_this_real_geometry(params.scene, b_stack.top().mat_id) && dist > first_eps + EPS_RAY) {
        IntersectionStack imaginary_isects_;
        imaginary_isects = &imaginary_isects_;
        {
            unsigned int u0, u1;
            pack_pointer(imaginary_isects, u0, u1);
#ifdef DBG_CORE_UTIL
            // this trace call does not make recursive trace calls
            params.trace_count[optixGetLaunchIndex().x]++;
#endif
            optixTrace(params.handle,
                       ray.origin,
                       ray.direction,
                       first_eps,
                       dist - EPS_RAY,
                       0.f,
                       GEOM_MASK_IMAG,
                       OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                       RAY_TYPE_SCENE_IMAG,
                       RAY_TYPE_COUNT,
                       RAY_TYPE_SCENE_IMAG,
                       u0,
                       u1);
            if (u0 == 0xFFFFFFFF && u1 == 0xFFFFFFFF) {
                // special value was set in the anyhit program
                RETURN_WITH_ERR(ERR_INTERSECTION_STACK);
            }
        }

        memcpy(&stack_copy, &b_stack, sizeof(b_stack));
        float dist_min = 0.f;

        for (int i = 0; i < imaginary_isects->size(); i++) {
            const LiteIntersection& isect = (*imaginary_isects)[i];
            int priority = get_medium_boundary_priority(params.scene, isect.get_mat_id());
            Boundary element(isect.get_mat_id(), isect.get_med_id());

            if (isect.get_enter()) {
                if (priority >= stack_copy.top_priority()) {
                    if (!volume_segments_all.push_back(VolLiteSegment(dist_min, isect.dist, stack_copy.top().med_id))) {
                        RETURN_WITH_ERR(ERR_VOL_SEGMENT_ARRAY);
                    }
                    dist_min = isect.dist;
                }

                if (!stack_copy.push(element, priority)) {
                    RETURN_WITH_ERR(ERR_BOUNDARY_STACK);
                }
            } else {
                if (priority == stack_copy.top_priority() && element == stack_copy.top()) {
                    if (!volume_segments_all.push_back(VolLiteSegment(dist_min, isect.dist, stack_copy.top().med_id))) {
                        RETURN_WITH_ERR(ERR_VOL_SEGMENT_ARRAY);
                    }
                    dist_min = isect.dist;
                }

                stack_copy.pop(element, priority);
            }
        }

        if (dist_min < dist) {
            if (!volume_segments_all.push_back(VolLiteSegment(dist_min, dist, stack_copy.top().med_id))) {
                RETURN_WITH_ERR(ERR_VOL_SEGMENT_ARRAY);
            }
        }
    } else {
        // otherwise, there is only one segment
        if (!volume_segments_all.push_back(VolLiteSegment(0.f, dist, b_stack.top().med_id))) {
            RETURN_WITH_ERR(ERR_VOL_SEGMENT_ARRAY);
        }
    }

    bool origin_in_medium = (ray_sampling_flags & ORIGIN_IN_MEDIUM) != 0;
    bool end_in_medium = (ray_sampling_flags & END_IN_MEDIUM) != 0;

    int isects_popped = 0;

    for (int i = 0; i < volume_segments_all.size(); i++) {
        const VolLiteSegment& segment = volume_segments_all[i];
        // Get medium in current segment
        int current_medium_id = segment.med_id;
        if (current_medium_id < 0) continue;
        Medium* current_medium_ptr = get_medium_ptr(params.scene, current_medium_id);
        GPU_ASSERT(current_medium_ptr != nullptr);

        unsigned int ray_sampling_flags_ = 0;
        if (origin_in_medium && segment.dist_min == 0) {
            ray_sampling_flags_ |= ORIGIN_IN_MEDIUM;
        }
        if (end_in_medium && segment.dist_max == dist) {
            ray_sampling_flags_ |= END_IN_MEDIUM;
        }

        float ray_sample_for_pdf = 1.f;
        float ray_sample_rev_pdf = 1.f;

        float dist_max = segment.dist_max;
        if (current_medium_ptr->has_scattering) {
            if (sample_media) {
                // find random value in (0,1)
                float random;
                do {
                    random = get_rnd(rng); // gets value in [0,1)
                } while (random == 0.f);
                // sample ray within the bounds of the segment
                float dist_to_next_segment = segment.dist_max - segment.dist_min;
                float dist_to_medium = sample_ray(current_medium_ptr,
                                                  dist_to_next_segment,
                                                  random,
                                                  &ray_sample_for_pdf,
                                                  ray_sampling_flags_,
                                                  &ray_sample_rev_pdf);

                // Output medium hit point if sampled inside medium
                if (dist_to_medium < dist_to_next_segment) {
                    result.dist = segment.dist_min + dist_to_medium;
                    GPU_ASSERT(result.dist > 0.f);
                    result.mat_id = -1;
                    result.med_id = current_medium_id;
                    result.light_id = -1;
                    result.normal = make_float3(0.f);
                    result.enter = false;
                    hit = true;
                    scattering_occurred = true;
                    dist_max = result.dist;
                }
            } else {
                ray_sample_for_pdf = ray_sample_pdf(
                    current_medium_ptr, segment.dist_min, segment.dist_max, ray_sampling_flags_, &ray_sample_rev_pdf);
            }
        }

        GPU_ASSERT(ray_sample_for_pdf > 0.f);
        GPU_ASSERT(ray_sample_rev_pdf > 0.f);

        if (!volume_segments_to_isect.push_back(
                VolSegment(eval_attenuation(current_medium_ptr, segment.dist_min, dist_max),
                           eval_emission(current_medium_ptr, segment.dist_min, dist_max),
                           segment.dist_min,
                           dist_max,
                           ray_sample_for_pdf,
                           ray_sample_rev_pdf,
                           current_medium_id))) {
            RETURN_WITH_ERR(ERR_VOL_SEGMENT_ARRAY);
        }

        // stack update
        if (imaginary_isects != nullptr) {
            for (int j = isects_popped; j < imaginary_isects->size(); j++) {
                const LiteIntersection& isect = (*imaginary_isects)[j];
                if (isect.dist > dist_max) break;

                if (isect.get_enter()) {
                    if (!b_stack.push(Boundary(isect.get_mat_id(), isect.get_med_id()),
                                      get_medium_boundary_priority(params.scene, isect.get_mat_id()))) {
                        RETURN_WITH_ERR(ERR_BOUNDARY_STACK);
                    }
                } else if (b_stack.size() > 1) {
                    b_stack.pop(Boundary(isect.get_mat_id(), isect.get_med_id()),
                                get_medium_boundary_priority(params.scene, isect.get_mat_id()));
                }

                isects_popped++;
            }
        }

        if (scattering_occurred) break;
    }

    // update boundary stack with hit materials
    for (int j = 0; j < hits_with_material.size(); j++) {
        const LiteIntersection& i = hits_with_material[j];
        if (i.dist > result.dist) break;

        if (i.get_enter()) {
            if (!b_stack.push(Boundary(i.get_mat_id(), i.get_med_id()),
                              get_medium_boundary_priority(params.scene, i.get_mat_id()))) {
                RETURN_WITH_ERR(ERR_BOUNDARY_STACK);
            }
        } else if (b_stack.size() > 1) {
            b_stack.pop(Boundary(i.get_mat_id(), i.get_med_id()),
                        get_medium_boundary_priority(params.scene, i.get_mat_id()));
        }
    }

    // shading normal is always computed, and this work is performed
    // in the hit programs

    return RET_SUCCESS;
}

/// Test for occlusion.
__forceinline__ __device__ unsigned int occluded(bool& hit,
                                                 const float3& point,
                                                 const float3& dir,
                                                 float t_max,
                                                 const BoundaryStack& b_stack,
                                                 unsigned int ray_sampling_flags,
                                                 VolSegmentArray& volume_segments,
                                                 RNGState& rng)
{
    BoundaryStack stack_copy;
    memcpy(&stack_copy, &b_stack, sizeof(b_stack));
    Intersection tmp(t_max);
    VolLiteSegmentArray volume_segments_all;
    return intersect_(hit,
                      Ray(point, dir),
                      tmp,
                      stack_copy,
                      OCCLUSION_TEST,
                      ray_sampling_flags,
                      volume_segments,
                      volume_segments_all,
                      rng);
}

/// Test for intersection.
__forceinline__ __device__ unsigned int intersect(bool& hit,
                                                  const Ray& ray,
                                                  unsigned int ray_sampling_flags,
                                                  Intersection& result,
                                                  BoundaryStack& boundary_stack,
                                                  VolSegmentArray& volume_segments_to_isect,
                                                  VolLiteSegmentArray& volume_segments_all,
                                                  RNGState& rng)
{
    return intersect_(hit,
                      ray,
                      result,
                      boundary_stack,
                      SAMPLE_VOLUME_SCATTERING,
                      ray_sampling_flags,
                      volume_segments_to_isect,
                      volume_segments_all,
                      rng);
}

#endif // KERNEL_INTERSECTOR_CUH
