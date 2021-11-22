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

#include "../shared/launch_params.h"
#include "../shared/vec_math.h"
#include "intersector.cuh"
#include "medium.cuh"
#include "optix_util.cuh"
#include "sarray.cuh"
#include "types.cuh"

#include <optix.h>

#define float3_as_ints(u) __float_as_int(u.x), __float_as_int(u.y), __float_as_int(u.z)

__forceinline__ __device__ void isect_sphere(OptixAabb* aabb)
{
    const int prim_idx = optixGetPrimitiveIndex(); // relative to build input
    const float3 aabb_min = make_float3(aabb[prim_idx].minX, aabb[prim_idx].minY, aabb[prim_idx].minZ);
    const float3 aabb_max = make_float3(aabb[prim_idx].maxX, aabb[prim_idx].maxY, aabb[prim_idx].maxZ);
    const float3 size = (aabb_max - aabb_min);
    const float3 center = aabb_min + .5f * size;
    // size.x == size.y == size.z
    const float radius = size.x * .5f;

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float ray_tmin = optixGetRayTmin();
    const float ray_tmax = optixGetRayTmax();

    const float3 o = ray_orig - center;
    const float l = 1.f / length(ray_dir);
    const float3 d = ray_dir * l;

    float b = dot(o, d);
    float c = dot(o, o) - radius * radius;
    float disc = b * b - c;
    if (disc > 0.f) {
        float sdisc = sqrtf(disc);
        float root1 = (-b - sdisc);
        float root11 = 0.f;
        bool check_second = true;

        const bool do_refine = fabsf(root1) > (1.0f * radius);

        if (do_refine) {
            // refine root1
            float3 O1 = o + root1 * d;
            b = dot(O1, d);
            c = dot(O1, O1) - radius * radius;
            disc = b * b - c;

            if (disc > 0.f) {
                sdisc = sqrtf(disc);
                root11 = (-b - sdisc);
            }
        }

        float t;
        float3 normal;
        t = (root1 + root11) * l;
        if (t > ray_tmin && t < ray_tmax) {
            normal = (o + (root1 + root11) * d) / radius;
            if (optixReportIntersection(t, 0, float3_as_ints(normal))) {
                check_second = false;
            }
        }

        if (check_second) {
            float root2 = (-b + sdisc) + (do_refine ? root1 : 0);
            t = root2 * l;
            normal = (o + root2 * d) / radius;
            if (t > ray_tmin && t < ray_tmax) {
                optixReportIntersection(t, 0, float3_as_ints(normal));
            }
        }
    }
}

extern "C" __global__ void __intersection__sphere_real() { isect_sphere(params.scene.spheres_real.aabbs); }

extern "C" __global__ void __intersection__sphere_imag() { isect_sphere(params.scene.spheres_imag.aabbs); }

__forceinline__ __device__ void fill_isect(Intersection* isect, float4* vertices, float3* normals)
{
    const float3 ray_dir = optixGetWorldRayDirection();
    // smallest reported hitT for CH, reported hitT for AH
    const float dist = optixGetRayTmax();
    float3 normal;
    float3 shading_normal;
    if (optixIsTriangleHit()) {
        // relative to build input
        const int prim_idx = optixGetPrimitiveIndex();

        // todo vertex data is not used. maybe deallocate this after
        //  creating the gases?

        const int norm_idx_offset = prim_idx * 4;
        normal = normals[norm_idx_offset + 3];
        if (params.use_shading_normal) {
            const float3 n0 = normals[norm_idx_offset + 0];
            const float3 n1 = normals[norm_idx_offset + 1];
            const float3 n2 = normals[norm_idx_offset + 2];
            float2 barys = optixGetTriangleBarycentrics();
            shading_normal = normalize(n0 * (1.f - barys.x - barys.y) + n1 * barys.x + n2 * barys.y);
        } else {
            shading_normal = normal;
        }
    } else {
        // world normal, there are no instance nodes in the sphere gas
        normal = make_float3(__float_as_int(optixGetAttribute_0()),
                             __float_as_int(optixGetAttribute_1()),
                             __float_as_int(optixGetAttribute_2()));
        shading_normal = normal;
    }

    HitgroupData* rt_data = (HitgroupData*)optixGetSbtDataPointer();
    isect->dist = dist;
    isect->mat_id = rt_data->gpu_mat.id_material;
    isect->med_id = params.scene.materials[isect->mat_id].med_idx;
    isect->light_id = rt_data->gpu_mat.id_light;
    isect->normal = normal;
    isect->enter = dot(normal, ray_dir) < 0.f;
    isect->shading_normal = shading_normal;
}

extern "C" __global__ void __closesthit__closest()
{
    Intersection* prd = get_inst<Intersection>();
    fill_isect(prd, params.scene.triangles_real.verts, params.scene.triangles_real.norms);
}

extern "C" __global__ void __anyhit__all()
{
    IntersectionStack* prd = get_inst<IntersectionStack>();
    Intersection isect;
    fill_isect(&isect, params.scene.triangles_imag.verts, params.scene.triangles_imag.norms);
    if (!prd->push(LiteIntersection(isect.dist, isect.mat_id, isect.med_id, isect.enter), isect.dist)) {
        // set special value
        optixSetPayload_0(0xFFFFFFFF);
        optixSetPayload_1(0xFFFFFFFF);
        // stop traversal
        optixTerminateRay();
    }
    // always reject the intersection, since accepting shrinks the search
    // interval
    optixIgnoreIntersection();
}
