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

#ifndef KERNEL_SCENE_CUH
#define KERNEL_SCENE_CUH

#include "../shared/shared_defs.h"
#include "camera.cuh"
#include "light.cuh"
#include "medium.cuh"
#include "types.cuh"

__forceinline__ __device__ int get_medium_boundary_priority(const Scene& scene, int mat_idx)
{
    return mat_idx >= 0 ? scene.materials[mat_idx].priority : GLOBAL_MEDIUM_PRIORITY;
}

/// Clear boundary stack and push in the global medium without any material.
__forceinline__ __device__ gpuupbp_status init_boundary_stack(const Scene& scene, BoundaryStack& stack)
{
    stack.clear();
    // Global medium has implicitly the lowest possible priority
    if (!stack.push({-1, static_cast<int>(scene.global_medium_idx)}, GLOBAL_MEDIUM_PRIORITY)) {
        RETURN_WITH_ERR(ERR_BOUNDARY_STACK);
    }

    return RET_SUCCESS;
}

__forceinline__ __device__ gpuupbp_status add_to_boundary_stack(const Scene& scene,
                                                                int mat_id,
                                                                int med_id,
                                                                BoundaryStack& stack)
{
    Boundary boundary;
    boundary.mat_id = mat_id;
    boundary.med_id = med_id;
    if (!stack.push(boundary, get_medium_boundary_priority(scene, mat_id))) {
        RETURN_WITH_ERR(ERR_BOUNDARY_STACK);
    }

    return RET_SUCCESS;
}

__forceinline__ __device__ gpuupbp_status update_boundary_stack_on_refract(const Scene& scene,
                                                                           const Intersection& isect,
                                                                           BoundaryStack& stack)
{
    if (isect.enter) {
        if (isect.med_id >= 0) {
            Boundary boundary;
            boundary.mat_id = isect.mat_id;
            boundary.med_id = isect.med_id;
            if (!stack.push(boundary, get_medium_boundary_priority(scene, isect.mat_id))) {
                RETURN_WITH_ERR(ERR_BOUNDARY_STACK);
            }
        }
    } else {
        if (stack.size() > 1) stack.pop_top();
    }

    return RET_SUCCESS;
}

__forceinline__ __device__ const Material& get_material(const Scene& scene, const int mat_idx)
{
    return scene.materials[mat_idx];
}

__forceinline__ __device__ Medium* get_medium_ptr(const Scene& scene, int med_idx)
{
    unsigned int idx = (int)fminf(med_idx, scene.med_count - 1);
    return &scene.media[idx];
}

__forceinline__ __device__ Medium* get_global_medium_ptr(const Scene& scene)
{
    return &scene.media[scene.global_medium_idx];
}

__forceinline__ __device__ bool is_this_real_geometry(const Scene& scene, int mat_idx)
{
    return mat_idx >= 0 && scene.materials[mat_idx].real;
}

__forceinline__ __device__ float relative_ior(const Scene& scene, const Intersection& isect, const BoundaryStack& stack)
{
    if (isect.is_on_surface()) {
        const Material& mat1 = get_material(scene, isect.mat_id);
        float ior1 = mat1.ior;

        if (ior1 < 0.f) return -1.f;

        if (isect.enter) {
            float ior2 = 1.f;

            int matId = stack.top().mat_id;
            if (matId >= 0) {
                const Material& mat2 = get_material(scene, matId);
                if (mat2.real) ior2 = mat2.ior;
            }

            GPU_ASSERT(ior1 != 0.f);

            if (ior2 < 0.f) {
                return -1.f;
            } else {
                return ior2 / ior1;
            }
        } else {
            float ior2 = 1.f;

            int matId = stack.size() > 1 ? stack.second_from_top().mat_id : stack.top().mat_id;
            if (matId >= 0) {
                const Material& mat2 = get_material(scene, matId);
                if (mat2.real) ior2 = mat2.ior;
            }

            GPU_ASSERT(ior2 != 0.f);

            if (ior2 < 0) {
                return -1.f;
            } else {
                return ior1 / ior2;
            }
        }
    } else {
        return -1.f;
    }
}

#endif // KERNEL_SCENE_CUH
