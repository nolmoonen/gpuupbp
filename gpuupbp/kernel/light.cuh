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

#ifndef KERNEL_LIGHT_CUH
#define KERNEL_LIGHT_CUH

#include "../shared/frame.h"
#include "../shared/shared_defs.h"
#include "defs.cuh"
#include "env_map.cuh"
#include "sample.cuh"
#include "types.cuh"

/** All light types, implemented as a tagged union. For details see light.h */

/** Area light. */

__forceinline__ __device__ float3 illuminate(const AreaLight& light,
                                             const SceneSphere& /*aSceneSphere*/,
                                             const float3& receiving_position,
                                             const float2& rnd,
                                             float3& direction_to_light,
                                             float& distance,
                                             float& for_pdf_w,
                                             float* emission_pdf_w = NULL,
                                             float* cos_at_light = NULL)
{
    const float2 uv = sample_uniform_triangle(rnd);
    const float3 light_point = light.p0 + light.e1 * uv.x + light.e2 * uv.y;

    direction_to_light = light_point - receiving_position;
    const float dist_sqr = dot(direction_to_light, direction_to_light);
    distance = sqrt(dist_sqr);
    direction_to_light = direction_to_light / distance;

    const float cos_normal_dir = dot(light.frame.normal(), -direction_to_light);

    // too close to, or under, tangent
    if (cos_normal_dir < EPS_COSINE) return make_float3(0.f);

    for_pdf_w = light.inv_area * dist_sqr / cos_normal_dir;

    if (cos_at_light) *cos_at_light = cos_normal_dir;

    if (emission_pdf_w) *emission_pdf_w = light.inv_area * cos_normal_dir * INV_PI_F;

    return light.intensity;
}

__forceinline__ __device__ float3 emit(const AreaLight& light,
                                       const SceneSphere& /*aSceneSphere*/,
                                       const float2& dir_rnd,
                                       const float2& pos_rnd,
                                       float3& position,
                                       float3& direction,
                                       float& emission_pdf_w,
                                       float* for_pdf_a,
                                       float* cos_theta_light)
{
    const float2 uv = sample_uniform_triangle(pos_rnd);
    position = light.p0 + light.e1 * uv.x + light.e2 * uv.y;

    float3 local_dir_out = sample_cos_hemisphere_w(dir_rnd, &emission_pdf_w);

    emission_pdf_w *= light.inv_area;

    // cannot really not emit the particle, so just bias it to the correct angle
    local_dir_out.z = fmaxf(local_dir_out.z, EPS_COSINE);
    direction = light.frame.to_world(local_dir_out);

    if (for_pdf_a) *for_pdf_a = light.inv_area;

    if (cos_theta_light) *cos_theta_light = local_dir_out.z;

    return light.intensity * local_dir_out.z;
}

__forceinline__ __device__ float3 get_radiance(const AreaLight& light,
                                               const SceneSphere& /*aSceneSphere*/,
                                               const float3& ray_direction,
                                               const float3& hitpoint,
                                               float* for_pdf_a = NULL,
                                               float* emission_pdf_w = NULL)
{
    const float cos_out_l = fmaxf(0.f, dot(light.frame.normal(), -ray_direction));

    if (cos_out_l == 0.f) return make_float3(0.f);

    if (for_pdf_a) *for_pdf_a = light.inv_area;

    if (emission_pdf_w) {
        *emission_pdf_w = cos_hemisphere_pdf_w(light.frame.normal(), -ray_direction);
        *emission_pdf_w *= light.inv_area;
    }

    return light.intensity;
}

__forceinline__ __device__ bool is_finite(const AreaLight& light) { return true; }

__forceinline__ __device__ bool is_delta(const AreaLight& light) { return false; }

/** Background light. */

__forceinline__ __device__ float3 illuminate(const BackgroundLight& light,
                                             const SceneSphere& scene_sphere,
                                             const float3& receiving_position,
                                             const float2& rnd,
                                             float3& direction_to_light,
                                             float& distance,
                                             float& for_pdf_w,
                                             float* emission_pdf_w = NULL,
                                             float* cos_at_light = NULL)
{
    float3 radiance;

    if (light.uses_env_map && params.scene.has_env_map) {
        direction_to_light = sample(params.scene.env_map, rnd, for_pdf_w, &radiance);
        radiance *= light.scale;
    } else {
        direction_to_light = sample_uniform_sphere_w(rnd, &for_pdf_w);
        radiance = light.background_color * light.scale;
    }

    // this stays even with image sampling
    distance = INFTY;
    if (emission_pdf_w) {
        *emission_pdf_w = for_pdf_w * concentric_disc_pdf_a() * scene_sphere.inv_radius_sqr;
    }

    if (cos_at_light) *cos_at_light = 1.f;

    return radiance;
}

__forceinline__ __device__ float3 emit(const BackgroundLight& light,
                                       const SceneSphere& scene_sphere,
                                       const float2& dir_rnd,
                                       const float2& pos_rnd,
                                       float3& position,
                                       float3& direction,
                                       float& emission_pdf_w,
                                       float* for_pdf_a,
                                       float* cos_theta_light)
{
    float for_pdf;
    float3 radiance;

    if (light.uses_env_map && params.scene.has_env_map) {
        direction = -sample(params.scene.env_map, dir_rnd, for_pdf, &radiance);
        radiance *= light.scale;
    } else {
        direction = sample_uniform_sphere_w(dir_rnd, &for_pdf);
        radiance = light.background_color * light.scale;
    }

    // stays even with image sampling
    const float2 xy = sample_concentric_disc(pos_rnd);

    Frame frame;
    frame.set_from_z(direction);

    position =
        scene_sphere.center + scene_sphere.radius * (-direction + frame.binormal() * xy.x + frame.tangent() * xy.y);

    emission_pdf_w = for_pdf * concentric_disc_pdf_a() * scene_sphere.inv_radius_sqr;

    // for background we lie about PDF being in area measure
    if (for_pdf_a) *for_pdf_a = for_pdf;

    // not used for infinite or delta lights
    if (cos_theta_light) *cos_theta_light = 1.f;

    return radiance;
}

__forceinline__ __device__ float3 get_radiance(const BackgroundLight& light,
                                               const SceneSphere& scene_sphere,
                                               const float3& ray_direction,
                                               const float3& /*hitpoint*/,
                                               float* for_pdf_a = NULL,
                                               float* emission_pdf_w = NULL)
{
    float for_pdf;
    float3 radiance;

    if (light.uses_env_map && params.scene.has_env_map) {
        radiance = light.scale * lookup(params.scene.env_map, ray_direction, &for_pdf);
    } else {
        for_pdf = uniform_sphere_pdf_w();
        radiance = light.background_color * light.scale;
    }

    const float position_pdf = concentric_disc_pdf_a() * scene_sphere.inv_radius_sqr;

    if (for_pdf_a) *for_pdf_a = for_pdf;

    if (emission_pdf_w) *emission_pdf_w = for_pdf * position_pdf;

    return radiance;
}

__forceinline__ __device__ bool is_finite(const BackgroundLight& light) { return false; }

__forceinline__ __device__ bool is_delta(const BackgroundLight& light) { return false; }

/** Directional light. */

__forceinline__ __device__ float3 illuminate(const DirectionalLight& light,
                                             const SceneSphere& scene_sphere,
                                             const float3& /*receiving_position*/,
                                             const float2& /*rnd*/,
                                             float3& direction_to_light,
                                             float& distance,
                                             float& for_pdf_w,
                                             float* emission_pdf_w = NULL,
                                             float* cos_at_light = NULL)
{
    direction_to_light = -light.frame.normal();
    distance = INFTY;
    for_pdf_w = 1.f;

    if (cos_at_light) *cos_at_light = 1.f;

    if (emission_pdf_w) {
        *emission_pdf_w = concentric_disc_pdf_a() * scene_sphere.inv_radius_sqr;
    }

    return light.intensity;
}

__forceinline__ __device__ float3 emit(const DirectionalLight& light,
                                       const SceneSphere& scene_sphere,
                                       const float2& /*dir_rnd*/,
                                       const float2& pos_rnd,
                                       float3& position,
                                       float3& direction,
                                       float& emission_pdf_w,
                                       float* for_pdf_a,
                                       float* cos_theta_light)
{
    const float2 xy = sample_concentric_disc(pos_rnd);

    position = scene_sphere.center + scene_sphere.radius * (-light.frame.normal() + light.frame.binormal() * xy.x +
                                                            light.frame.tangent() * xy.y);

    direction = light.frame.normal();
    emission_pdf_w = concentric_disc_pdf_a() * scene_sphere.inv_radius_sqr;

    if (for_pdf_a) *for_pdf_a = 1.f;

    // not used for infinite or delta lights
    if (cos_theta_light) *cos_theta_light = 1.f;

    return light.intensity;
}

__forceinline__ __device__ float3 get_radiance(const DirectionalLight& light,
                                               const SceneSphere& /*scene_sphere*/,
                                               const float3& /*ray_direction*/,
                                               const float3& /*hitpoint*/,
                                               float* for_pdf_a = NULL,
                                               float* emission_pdf_w = NULL)
{
    return make_float3(0.f);
}

__forceinline__ __device__ bool is_finite(const DirectionalLight& light) { return false; }

__forceinline__ __device__ bool is_delta(const DirectionalLight& light) { return true; }

/** Point light. */

__forceinline__ __device__ float3 illuminate(const PointLight& light,
                                             const SceneSphere& /*scene_sphere*/,
                                             const float3& receiving_position,
                                             const float2& rnd,
                                             float3& direction_to_light,
                                             float& distance,
                                             float& for_pdf_w,
                                             float* emission_pdf_w = NULL,
                                             float* cos_at_light = NULL)
{
    direction_to_light = light.position - receiving_position;
    const float dist_sqr = dot(direction_to_light, direction_to_light);
    for_pdf_w = dist_sqr;
    distance = sqrtf(dist_sqr);
    direction_to_light = direction_to_light / distance;

    if (cos_at_light) *cos_at_light = 1.f;

    if (emission_pdf_w) *emission_pdf_w = uniform_sphere_pdf_w();

    return light.intensity;
}

__forceinline__ __device__ float3 emit(const PointLight& light,
                                       const SceneSphere& /*scene_sphere*/,
                                       const float2& dir_rnd,
                                       const float2& /*pos_rnd*/,
                                       float3& position,
                                       float3& direction,
                                       float& emission_pdf_w,
                                       float* for_pdf_a,
                                       float* cos_theta_light)
{
    position = light.position;
    direction = sample_uniform_sphere_w(dir_rnd, &emission_pdf_w);

    if (for_pdf_a) *for_pdf_a = 1.f;

    // not used for infinite or delta lights
    if (cos_theta_light) *cos_theta_light = 1.f;

    return light.intensity;
}

__forceinline__ __device__ float3 get_radiance(const PointLight& light,
                                               const SceneSphere& /*scene_sphere*/,
                                               const float3& /*ray_direction*/,
                                               const float3& /*hitpoint*/,
                                               float* for_pdf_a = NULL,
                                               float* emission_pdf_w = NULL)
{
    return make_float3(0.f);
}

__forceinline__ __device__ bool is_finite(const PointLight& light) { return true; }

__forceinline__ __device__ bool is_delta(const PointLight& light) { return true; }

/** Abstract light. */

/// Illuminates a given point in the scene.
/// Given a point and two random samples (e.g., for position on area lights),
/// this method returns direction from point to light, distance,
/// PDF of having chosen this direction (e.g., 1 / area).
/// Optionally also returns PDF of emitting particle in this direction,
/// and cosine from lights normal (helps with PDF of hitting the light,
/// but set to 1 for point lights).
/// Returns radiance.
__forceinline__ __device__ float3 illuminate(const AbstractLight& light,
                                             const SceneSphere& scene_sphere,
                                             const float3& receiving_position,
                                             const float2& rnd,
                                             float3& direction_to_light,
                                             float& distance,
                                             float& for_pdf_w,
                                             float* emission_pdf_w = NULL,
                                             float* cos_at_light = NULL)
{
    switch (light.type) {
    case AREA:
        return illuminate(light.area,
                          scene_sphere,
                          receiving_position,
                          rnd,
                          direction_to_light,
                          distance,
                          for_pdf_w,
                          emission_pdf_w,
                          cos_at_light);
    case BACKGROUND:
        return illuminate(light.background,
                          scene_sphere,
                          receiving_position,
                          rnd,
                          direction_to_light,
                          distance,
                          for_pdf_w,
                          emission_pdf_w,
                          cos_at_light);
    case DIRECTIONAL:
        return illuminate(light.directional,
                          scene_sphere,
                          receiving_position,
                          rnd,
                          direction_to_light,
                          distance,
                          for_pdf_w,
                          emission_pdf_w,
                          cos_at_light);
    case POINT_LIGHT:
        return illuminate(light.point,
                          scene_sphere,
                          receiving_position,
                          rnd,
                          direction_to_light,
                          distance,
                          for_pdf_w,
                          emission_pdf_w,
                          cos_at_light);
    }
    GPU_ASSERT(false);
    return make_float3(0.f);
}

/// Emits particle from the light.
/// Given two sets of random numbers (e.g., position and direction on area light),
/// this method generates a position and direction for light particle, along
/// with the PDF.
/// Can also supply PDF (w.r.t. area) of choosing this position when calling
/// illuminate. Also provides cosine on the light (this is 1 for point lights etc.).
/// Returns "energy" that particle carries
__forceinline__ __device__ float3 emit(const AbstractLight& light,
                                       const SceneSphere& scene_sphere,
                                       const float2& dir_rnd,
                                       const float2& pos_rnd,
                                       float3& position,
                                       float3& direction,
                                       float& emission_pdf_w,
                                       float* for_pdf_a,
                                       float* cos_theta_light)
{
    switch (light.type) {
    case AREA:
        return emit(light.area,
                    scene_sphere,
                    dir_rnd,
                    pos_rnd,
                    position,
                    direction,
                    emission_pdf_w,
                    for_pdf_a,
                    cos_theta_light);
    case BACKGROUND:
        return emit(light.background,
                    scene_sphere,
                    dir_rnd,
                    pos_rnd,
                    position,
                    direction,
                    emission_pdf_w,
                    for_pdf_a,
                    cos_theta_light);
    case DIRECTIONAL:
        return emit(light.directional,
                    scene_sphere,
                    dir_rnd,
                    pos_rnd,
                    position,
                    direction,
                    emission_pdf_w,
                    for_pdf_a,
                    cos_theta_light);
    case POINT_LIGHT:
        return emit(light.point,
                    scene_sphere,
                    dir_rnd,
                    pos_rnd,
                    position,
                    direction,
                    emission_pdf_w,
                    for_pdf_a,
                    cos_theta_light);
    }
    GPU_ASSERT(false);
    return make_float3(0.f);
}

/// Returns radiance for ray randomly hitting the light
/// Given ray direction and hitpoint, it returns radiance.
/// Can also provide area PDF of sampling hitpoint in illuminate,
/// and of emitting particle along the ray (in opposite direction).
__forceinline__ __device__ float3 get_radiance(const AbstractLight& light,
                                               const SceneSphere& scene_sphere,
                                               const float3& ray_direction,
                                               const float3& hitpoint,
                                               float* for_pdf_a = NULL,
                                               float* emission_pdf_w = NULL)
{
    switch (light.type) {
    case AREA:
        return get_radiance(light.area, scene_sphere, ray_direction, hitpoint, for_pdf_a, emission_pdf_w);
    case BACKGROUND:
        return get_radiance(light.background, scene_sphere, ray_direction, hitpoint, for_pdf_a, emission_pdf_w);
    case DIRECTIONAL:
        return get_radiance(light.directional, scene_sphere, ray_direction, hitpoint, for_pdf_a, emission_pdf_w);
    case POINT_LIGHT:
        return get_radiance(light.point, scene_sphere, ray_direction, hitpoint, for_pdf_a, emission_pdf_w);
    }
    GPU_ASSERT(false);
    return make_float3(0.f);
}

/// Whether the light has a finite extent (area, point) or
/// not (directional, env. map).
__forceinline__ __device__ bool is_finite(const AbstractLight& light)
{
    switch (light.type) {
    case AREA:
        return is_finite(light.area);
    case BACKGROUND:
        return is_finite(light.background);
    case DIRECTIONAL:
        return is_finite(light.directional);
    case POINT_LIGHT:
        return is_finite(light.point);
    }
    GPU_ASSERT(false);
    return false;
}

/// Whether the light has delta function (point, directional) or
/// not (area).
__forceinline__ __device__ bool is_delta(const AbstractLight& light)
{
    switch (light.type) {
    case AREA:
        return is_delta(light.area);
    case BACKGROUND:
        return is_delta(light.background);
    case DIRECTIONAL:
        return is_delta(light.directional);
    case POINT_LIGHT:
        return is_delta(light.point);
    }
    GPU_ASSERT(false);
    return false;
}

#endif // KERNEL_LIGHT_CUH
