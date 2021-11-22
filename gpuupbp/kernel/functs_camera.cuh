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

#ifndef KERNEL_FUNCTS_CAMERA_CUH
#define KERNEL_FUNCTS_CAMERA_CUH

#include "camera.cuh"
#include "functs_shared.cuh"
#include "intersector.cuh"
#include "light.cuh"
#include "params_def.cuh"
#include "path_weight.cuh"
#include "sample.cuh"
#include "scene.cuh"
#include "types.cuh"

/// Generates a new camera sample given a pixel index.
__forceinline__ __device__ gpuupbp_status generate_camera_sample(
    float2& screen_sample, unsigned int pixel_index, SubpathState& camera_state, const Scene& scene, RNGState& state)
{
    const Camera& camera = scene.camera;
    const int resX = int(camera.resolution.x);

    // determine pixel (x, y)
    const unsigned int x = pixel_index % resX;
    const unsigned int y = pixel_index / resX;

    // jitter pixel position
    float2 rand = make_float2(get_rnd(state), get_rnd(state));
    screen_sample = make_float2(float(x), float(y)) + rand;

    // generate ray
    const Ray primary_ray = generate_ray(camera, screen_sample);

    // compute PDF conversion factor from area on image plane to
    // solid angle on ray
    const float cos_at_camera = dot(camera.direction, primary_ray.direction);
    const float image_point_to_camera_dist = camera.image_plane_dist / cos_at_camera;
    const float image_to_solid_angle_factor = (image_point_to_camera_dist * image_point_to_camera_dist) / cos_at_camera;

    // we put the virtual image plane at such a distance from the camera origin
    // that the pixel area is one and thus the image plane sampling PDF is 1.

    // the solid angle ray PDF is then equal to the conversion factor from
    // image plane area density to ray solid angle density
    const float camera_pdf_w = image_to_solid_angle_factor;

    camera_state.origin = primary_ray.origin;
    camera_state.direction = primary_ray.direction;
    camera_state.throughput = make_float3(1.f);
    camera_state.path_length = 1;
    camera_state.is_specular_path = true;

    // prepare weights for the first camera vertex, index 1
    // for the first vertex, we can set dBPT and dPDE immediately
    {
        /** Eq D.27 (camera) */
        camera_state.weights.weights.d_shared = params.light_subpath_count / camera_pdf_w;
        /** Eq D.28 (camera) */
        camera_state.weights.weights.d_bpt = 0.f;
        /** Eq D.29 (camera) */
        camera_state.weights.weights.d_pde = 0.f;

        // index 0 is on camera object
        camera_state.weights.weights.prev_specular = false;
        camera_state.weights.weights.prev_in_medium = false;
    }

    // init the boundary stack with the global medium
    gpuupbp_status ret = init_boundary_stack(scene, camera_state.stack);
    if (ret != RET_SUCCESS) return ret;

    // add enclosing material and medium if present
    if (camera.mat_id != -1 && camera.med_id != -1) {
        ret = add_to_boundary_stack(scene, camera.mat_id, camera.med_id, camera_state.stack);
        if (ret != RET_SUCCESS) return ret;
    }

    return RET_SUCCESS;
}

/// Returns the radiance of a light source when hit by a random ray,
/// multiplied by MIS weight. Can be used for both Background and Area lights.
__forceinline__ __device__ float3 get_light_radiance(const AbstractLight& light,
                                                     const SubpathState& camera_state,
                                                     const float3& hit_point,
                                                     /// pr_{T,t-1}(z)
                                                     float ray_sample_for_pdf,
                                                     /// pl_{T,t-2}(z)
                                                     float ray_sample_rev_pdf)
{
    GPU_ASSERT(params.do_bpt);
    if (camera_state.is_specular_path && params.ignore_fully_spec_paths) {
        return make_float3(0.f);
    }

    // We sample lights uniformly
    const float light_pick_prob = 1.f / float(params.scene.light_count);

    float direct_pdf_a;   // p^connect_{t-1}
    float emission_pdf_w; // p^trace_{t-1}
    const float3 radiance = get_radiance(
        light, params.scene.scene_sphere, camera_state.direction, hit_point, &direct_pdf_a, &emission_pdf_w);

    if (is_black_or_negative(radiance)) return make_float3(0.f);

    // If we see light source directly from camera, no weighting is required
    if (camera_state.path_length == 1) return radiance;

    // when using only vertex merging, we want purely specular paths
    // to give radiance (cannot get it otherwise). Rest is handled
    // by merging and we should return 0.
    if (params.estimator_techniques && !(params.estimator_techniques & BPT)) {
        return camera_state.is_specular_path ? radiance : make_float3(0.f);
    }

    direct_pdf_a *= light_pick_prob;
    emission_pdf_w *= light_pick_prob;

    GPU_ASSERT(direct_pdf_a > 0.f);
    GPU_ASSERT(emission_pdf_w > 0.f);

    /** Eq D.32 */
    const float w_light = 0.f;

    /** Eq 4.12 */
    const float w_local = 1.f;

    const VertexWeights& weight_c = camera_state.weights.weights; // t-1
    // bpt_{t-1} considers t-2 and t-1
    const float bpt_t_sub_1 = !weight_c.prev_specular; // whether t-2 is specular
    // we know t-1 to not be delta as this light would not be hit otherwise

    /** Eq D.33 */
    const float w_camera =
        direct_pdf_a * bpt_t_sub_1 * weight_c.d_shared + emission_pdf_w * ray_sample_rev_pdf * weight_c.d_bpt;

    /** Eq 4.18 */
    const float mis_weight = 1.f / (w_light + w_local + w_camera);

    return mis_weight * radiance;
}

/// Connects camera vertex to randomly chosen light point.
/// Returns emitted radiance multiplied by path MIS weight.
/// Has to be called AFTER updating the MIS quantities.
__forceinline__ __device__ unsigned int direct_illumination(float3& contrib,
                                                            const SubpathState& camera_state,
                                                            const float3& hit_point,
                                                            const BSDF& camera_bsdf,
                                                            RNGState& state,
                                                            VolSegmentArray& volume_segments)
{
    GPU_ASSERT(params.do_bpt);

    // we sample lights uniformly
    const float light_pick_prob = 1.f / float(params.scene.light_count);

    // rnd generates in [0,1)
    const int light_id = int(get_rnd(state) * float(params.scene.light_count));
    const float2 rndPosSamples = make_float2(get_rnd(state), get_rnd(state));

    const AbstractLight& light = params.scene.lights[light_id];

    // light in infinity in attenuating homogeneous global medium is always
    // reduced to zero
    if (!is_finite(light) && get_global_medium_ptr(params.scene)->has_attenuation) {
        contrib = make_float3(0.f);
        return RET_SUCCESS;
    }

    float3 direction_to_light;
    float distance;
    float direct_pdf_w, emission_pdf_w, cos_at_light;
    const float3 radiance = illuminate(light,
                                       params.scene.scene_sphere,
                                       hit_point,
                                       rndPosSamples,
                                       direction_to_light,
                                       distance,
                                       direct_pdf_w,
                                       &emission_pdf_w,
                                       &cos_at_light);

    // if radiance == 0, other values are undefined, so save to early exit
    if (is_black_or_negative(radiance)) {
        contrib = make_float3(0.f);
        return RET_SUCCESS;
    }

    GPU_ASSERT(direct_pdf_w > 0.f);
    GPU_ASSERT(emission_pdf_w > 0.f);
    GPU_ASSERT(cos_at_light > 0.f);

    // Get BSDF factor at the last camera vertex
    float bsdf_for_pdf_w, bsdf_rev_pdf_w, cos_to_light, sin_theta;
    float3 bsdfFactor = evaluate(camera_bsdf,
                                 direction_to_light,
                                 cos_to_light,
                                 params.scene.materials,
                                 &bsdf_for_pdf_w,
                                 &bsdf_rev_pdf_w,
                                 &sin_theta);

    if (is_black_or_negative(bsdfFactor)) {
        contrib = make_float3(0.f);
        return RET_SUCCESS;
    }

    const float continuation_probability = camera_bsdf.continuation_prob;

    // If the light is delta light, we can never hit it
    // by BSDF sampling, so the probability of this path is 0
    bsdf_for_pdf_w *= is_delta(light) ? 0.f : continuation_probability;
    bsdf_rev_pdf_w *= continuation_probability;

    GPU_ASSERT(bsdf_rev_pdf_w > 0.f);
    GPU_ASSERT(cos_to_light > 0.f);

    contrib = make_float3(0.f);

    // Test occlusion
    volume_segments.clear();
    bool is_occluded;
    unsigned int ret = occluded(is_occluded,
                                hit_point,
                                direction_to_light,
                                distance,
                                camera_state.stack,
                                is_in_medium(camera_bsdf) ? ORIGIN_IN_MEDIUM : 0,
                                volume_segments,
                                state);
    if (ret != RET_SUCCESS) return ret;

    if (is_occluded) return RET_SUCCESS;

    // Get attenuation from intersected media (if any)
    float next_ray_sample_for_pdf = 1.f; // pr_{T,t  }(z) = pl_{T,s-1}(y)
    float next_ray_sample_rev_pdf = 1.f; // pl_{T,t-1}(z) = pr_{T,s  }(y)
    float3 next_attenuation = make_float3(1.f);
    if (!volume_segments.empty()) {
        // PDF
        next_ray_sample_for_pdf = accumulate_for_pdf(volume_segments);
        GPU_ASSERT(next_ray_sample_for_pdf > 0.f);

        // Reverse PDF
        next_ray_sample_rev_pdf = accumulate_rev_pdf(volume_segments);
        GPU_ASSERT(next_ray_sample_rev_pdf > 0.f);

        // Attenuation (without PDF!)
        next_attenuation = accumulate_attenuation_without_pdf(volume_segments);
        if (!is_positive(next_attenuation)) {
            contrib = make_float3(0.f);
            return RET_SUCCESS;
        }
    }

    // \bpt_s considers s-1 and s
    // (1 / \btp_s) = 1, as this function is not called on delta vertices
    // and the light is also non-delta

    /** Eq. D.36 */
    // Note that wLight is a ratio of area pdfs. But since both are on the
    // light source, their distance^2 and cosine terms cancel out.
    // Therefore, we can write wLight as a ratio of solid angle pdfs,
    // both expressed w.r.t. the same shading point.
    const float w_light = bsdf_for_pdf_w / (light_pick_prob * direct_pdf_w);

    /** Eq 4.12 */
    const float w_local = 1.f;

    const VertexWeights& weight_c = camera_state.weights.weights; // t-1

    // f_t considers a subpath where the last vertex is t-1
    GPU_ASSERT(!is_delta(camera_bsdf)); // we pass false
    float ft = get_pde_factor(is_in_medium(camera_bsdf),
                              false,
                              1.f / next_ray_sample_rev_pdf,      // 1          /pr_{T,t-1}(x)
                              weight_c.ray_sample_rev_pdfs_ratio, // prob_r{t-1}/pr_{T,t-1}(x)
                              weight_c.ray_sample_for_pdf_inv,    // 1          /pl_{T,t-1}(x)
                              weight_c.ray_sample_for_pdfs_ratio, // prob_l{t-1}/pl_{T,t-1}(x)
                              sin_theta);

    // bpt_{t-1} considers t-2 and t-1
    const float b_t_sub_1 = !weight_c.prev_specular; // whether t-2 is specular
    // t-1 is not delta (as it is on the light)

    // In front of the sum in the parenthesis we have (ratio), where
    //    ratio = emissionPdfA / directPdfA,
    // with emissionPdfA being the product of the pdfs for choosing the
    // point on the light source and sampling the outgoing direction.
    // What we are given by the light source instead are emissionPdfW
    // and directPdfW. Converting to area pdfs and plugging into ratio:
    //    emissionPdfA = emissionPdfW * cosToLight / dist^2
    //    directPdfA   = directPdfW * cosAtLight / dist^2
    //    ratio = (emissionPdfW * cosToLight / dist^2) /
    //            (directPdfW * cosAtLight / dist^2)
    //    ratio = (emissionPdfW * cosToLight) / (directPdfW * cosAtLight)
    //
    // Also note that both emissionPdfW and directPdfW should be
    // multiplied by lightPickProb, so it cancels out.

    /** Eq. D.37 */
    // 1 / bpt_t = 1 (considers t-1 and t)
    // ignore division by n_\bpt (which is only non-one if BPT is disabled)
    const float w_camera =
        (emission_pdf_w * cos_to_light / (direct_pdf_w * cos_at_light)) * next_ray_sample_rev_pdf *
        (ft + b_t_sub_1 * weight_c.d_shared + (bsdf_rev_pdf_w / weight_c.ray_sample_rev_pdf_inv) * weight_c.d_bpt);

    /** Eq 4.18 */
    const float mis_weight = 1.f / (w_light + w_local + w_camera);

    contrib =
        ((cos_to_light / (light_pick_prob * direct_pdf_w)) * (radiance * next_attenuation * bsdfFactor)) * mis_weight;

    if (is_black_or_negative(contrib)) {
        contrib = make_float3(0.f);
    }

    return RET_SUCCESS;
}

/// Connects an eye and a light vertex. Result multiplied by MIS weight, but
/// not multiplied by vertex throughputs. Has to be called AFTER updating MIS
/// constants. 'direction' is FROM eye TO light vertex.
__forceinline__ __device__ gpuupbp_status connect_vertices(float3& contrib,
                                                           const LightVertex& light_vertex,
                                                           const BSDF& camera_bsdf,
                                                           const float3& camera_hitpoint,
                                                           const SubpathState& camera_state,
                                                           RNGState state,
                                                           VolSegmentArray& volume_segments)
{
    const BSDF& bsdf_c = camera_bsdf;
    const BSDF& bsdf_l = light_vertex.bsdf;

    // get the connection
    float3 direction = light_vertex.hitpoint - camera_hitpoint;
    const float dist2 = dot(direction, direction);
    float distance = sqrtf(dist2);
    direction /= distance;

    // evaluate BSDF at camera vertex
    float cos_camera, camera_bsdf_for_pdf_w, camera_bsdf_rev_pdf_w, sin_theta_camera;
    float3 camera_bsdf_factor = evaluate(camera_bsdf,
                                         direction,
                                         cos_camera,
                                         params.scene.materials,
                                         &camera_bsdf_for_pdf_w,
                                         &camera_bsdf_rev_pdf_w,
                                         &sin_theta_camera);

    if (is_black_or_negative(camera_bsdf_factor)) {
        contrib = make_float3(0.f);
        return RET_SUCCESS;
    }

    // camera continuation probability (for Russian roulette)
    const float camera_cont = camera_bsdf.continuation_prob;
    camera_bsdf_for_pdf_w *= camera_cont;
    camera_bsdf_rev_pdf_w *= camera_cont;
    GPU_ASSERT(camera_bsdf_for_pdf_w > 0.f);
    GPU_ASSERT(camera_bsdf_rev_pdf_w > 0.f);

    // evaluate BSDF at light vertex
    float cos_light, light_bsdf_for_pdf_w, light_bsdf_rev_pdf_w, sin_theta_light;
    const float3 light_bsdf_factor = evaluate(light_vertex.bsdf,
                                              -direction,
                                              cos_light,
                                              params.scene.materials,
                                              &light_bsdf_for_pdf_w,
                                              &light_bsdf_rev_pdf_w,
                                              &sin_theta_light);

    if (is_black_or_negative(light_bsdf_factor)) {
        contrib = make_float3(0.f);
        return RET_SUCCESS;
    }

    // light continuation probability (for Russian roulette)
    const float light_cont = light_vertex.bsdf.continuation_prob;
    light_bsdf_for_pdf_w *= light_cont;
    light_bsdf_rev_pdf_w *= light_cont;
    GPU_ASSERT(light_bsdf_for_pdf_w > 0.f);
    GPU_ASSERT(light_bsdf_rev_pdf_w > 0.f);

    // compute geometry term
    const float geometry_term = cos_light * cos_camera / dist2;
    if (geometry_term < 0.f) {
        contrib = make_float3(0.f);
        return RET_SUCCESS;
    }

    // convert PDFs to area PDF
    const float camera_bsdf_for_pdf_a = pdf_w_to_a(camera_bsdf_for_pdf_w, distance, cos_light);
    const float light_bsdf_for_pdf_a = pdf_w_to_a(light_bsdf_for_pdf_w, distance, cos_camera);
    GPU_ASSERT(camera_bsdf_for_pdf_a > 0.f);
    GPU_ASSERT(light_bsdf_for_pdf_a > 0.f);

    unsigned int ray_sampling_flags = 0;
    if (is_in_medium(camera_bsdf)) {
        ray_sampling_flags |= ORIGIN_IN_MEDIUM;
    }
    if (light_vertex.is_in_medium()) {
        ray_sampling_flags |= END_IN_MEDIUM;
    }

    // test occlusion, from camera to light
    volume_segments.clear();
    bool is_occluded;
    unsigned int ret = occluded(is_occluded,
                                camera_hitpoint,
                                direction,
                                distance,
                                camera_state.stack,
                                ray_sampling_flags,
                                volume_segments,
                                state);
    if (ret != RET_SUCCESS) return ret;
    if (is_occluded) {
        contrib = make_float3(0.f);
        return RET_SUCCESS;
    }

    // note that s-1=t and s=t-1

    // attenuation from intersected media (if any), from camera to light
    float ray_sample_for_pdf = 1.f; // pr_{T,t  }(z) = pl_{T,s-1}(y)
    float ray_sample_rev_pdf = 1.f; // pl_{T,t-1}(z) = pr_{T,s  }(y)
    float3 attenuation = make_float3(1.f);
    if (!volume_segments.empty()) {
        ray_sample_for_pdf = accumulate_for_pdf(volume_segments);
        GPU_ASSERT(ray_sample_for_pdf > 0.f);

        ray_sample_rev_pdf = accumulate_rev_pdf(volume_segments);
        GPU_ASSERT(ray_sample_rev_pdf > 0.f);

        // attenuation (without PDF!)
        attenuation = accumulate_attenuation_without_pdf(volume_segments);
        if (!is_positive(attenuation)) {
            contrib = make_float3(0.f);
            return RET_SUCCESS;
        }
    }

    const VertexWeights& weight_l = light_vertex.weights;         // s-1
    const VertexWeights& weight_c = camera_state.weights.weights; // t-1

    // f_s considers a subpath where the last vertex is s-1
    GPU_ASSERT(!is_delta(bsdf_l)); // we pass false
    float f_s = get_pde_factor(is_in_medium(bsdf_l),
                               false,
                               weight_l.ray_sample_for_pdf_inv,    // 1             /pr_{T,s-1}(x)
                               weight_l.ray_sample_for_pdfs_ratio, // prob_r{s-1}(x)/pr_{T,s-1}(x)
                               1.f / ray_sample_for_pdf,           // 1             /pl_{T,s-1}(x)
                               weight_l.ray_sample_rev_pdfs_ratio, // prob_l{s-1}(x)/pl_{T,s-1}(x)
                               sin_theta_light);

    // bpt_{s-1} considers s-2 and s-1
    const float bpt_s_sub_1 = !weight_l.prev_specular; // whether s-2 is specular
    // s-1 is not delta (else this method is not called)

    /** Eq 4.50 */
    // 1 / bpt_s = 1 (considers s-1 and s)
    // ignore division by n_\bpt (which is only non-one if BPT is disabled)
    const float w_light = camera_bsdf_for_pdf_a * ray_sample_for_pdf *
                          (f_s + bpt_s_sub_1 * weight_l.d_shared +
                           (light_bsdf_rev_pdf_w / weight_l.ray_sample_rev_pdf_inv) * weight_l.d_bpt);

    /** Eq 4.12 */
    const float w_local = 1.f;

    // f_t considers a subpath where the last vertex is t-1
    GPU_ASSERT(!is_delta(bsdf_c)); // we pass false
    float f_t = get_pde_factor(is_in_medium(bsdf_c),
                               false,
                               1.f / ray_sample_rev_pdf,           // 1             /pr_{T,t-1}(x)
                               weight_c.ray_sample_rev_pdfs_ratio, // prob_r{t-1}(x)/pr_{T,t-1}(x)
                               weight_c.ray_sample_for_pdf_inv,    // 1             /pl_{T,t-1}(x)
                               weight_c.ray_sample_for_pdfs_ratio, // prob_l{t-1}(x)/pl_{T,t-1}(x)
                               sin_theta_camera);

    // bpt_{t-1} considers t-2 and t-1
    const float bpt_t_sub_1 = !weight_c.prev_specular; // whether t-2 is specular
    // t-1 is not delta (otherwise the vertex would not have been stored)

    /** Eq 4.51 */
    // 1 / bpt_t = 1 (considers t-1 and t)
    // ignore division by n_\bpt (which is only non-one if BPT is disabled)
    const float w_camera = light_bsdf_for_pdf_a * ray_sample_rev_pdf *
                           (f_t + bpt_t_sub_1 * weight_c.d_shared +
                            (camera_bsdf_rev_pdf_w / weight_c.ray_sample_rev_pdf_inv) * weight_c.d_bpt);

    /** Eq 4.18 */
    const float mis_weight = 1.f / (w_light + w_local + w_camera);

    contrib = (geometry_term * camera_bsdf_factor * light_bsdf_factor * attenuation) * mis_weight;

    if (is_black_or_negative(contrib)) {
        contrib = make_float3(0.f);
    }

    return RET_SUCCESS;
}

#endif // KERNEL_FUNCTS_CAMERA_CUH
