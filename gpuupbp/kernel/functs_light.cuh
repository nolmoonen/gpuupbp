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

#ifndef KERNEL_FUNCTS_LIGHT_CUH
#define KERNEL_FUNCTS_LIGHT_CUH

#include "../shared/light_vertex.h"
#include "../shared/matrix.h"
#include "curand_kernel.h"
#include "frame_buffer.cuh"
#include "functs_shared.cuh"
#include "intersector.cuh"
#include "light.cuh"
#include "params_def.cuh"
#include "path_weight.cuh"
#include "scene.cuh"
#include "types.cuh"

__forceinline__ __device__ void set_beam_instance(LightBeam* beam, OptixInstance* instance)
{
    instance->traversableHandle = params.handle_gas_beam;
    instance->flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
    instance->instanceId = 0; // not used
    instance->sbtOffset = params.sbt_offset;
    instance->visibilityMask =
        // only add bb1d mask if path index is in the subset
        GEOM_MASK_BP2D |
        ((static_cast<unsigned int>(beam->path_idx < params.bb1d_used_light_sub_path_count)) * GEOM_MASK_BB1D);
    sutil::Matrix4x4 scale = sutil::Matrix4x4::scale({params.rad_beam, beam->beam_length, params.rad_beam});
    // https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    const float3 normalDir = normalize(beam->ray.direction);
    const float3 up = make_float3(0.f, 1.f, 0.f);
    const float3 ab = normalDir + up;
    const float inner = dot(ab, ab);
    const float xy = ab.x * ab.y;
    const float xz = ab.x * ab.z;
    const float yz = ab.y * ab.z;
    const sutil::Matrix3x3 outer({ab.x * ab.x, xy, xz, xy, ab.y * ab.y, yz, xz, yz, ab.z * ab.z});
    const sutil::Matrix3x3 rot = (2.f / inner) * outer - sutil::Matrix3x3::identity();
    sutil::Matrix4x4 rotate({rot.getData()[0],
                             rot.getData()[1],
                             rot.getData()[2],
                             0.f,
                             rot.getData()[3],
                             rot.getData()[4],
                             rot.getData()[5],
                             0.f,
                             rot.getData()[6],
                             rot.getData()[7],
                             rot.getData()[8],
                             0.f,
                             0.f,
                             0.f,
                             0.f,
                             1.f});
    sutil::Matrix4x4 translate = sutil::Matrix4x4::translate(beam->ray.origin);
    sutil::Matrix4x4 transform = translate * rotate * scale;
    // only copy first three rows, the last row is always 0001 and is implied
    memcpy(instance->transform, transform.getData(), sizeof(float) * 12);
}

/// Generates a new light sample by sampling light emission.
__forceinline__ __device__ gpuupbp_status generate_light_sample(SubpathState& light_state,
                                                                const Scene& scene,
                                                                RNGState& state)
{
    const auto light_count = static_cast<float>(scene.light_count);

    // we sample lights uniformly, rnd generates in [0,1)
    const int lightID = int(get_rnd(state) * light_count);
    const float4 rnd = get_rnd4(state);
    const float2 rnd_dir_samples = make_float2(rnd.x, rnd.y);
    const float2 rnd_pos_samples = make_float2(rnd.z, rnd.w);

    const AbstractLight& light = scene.lights[lightID];

    float emission_pdf_w, direct_pdf_a, cos_light;
    light_state.throughput = emit(light,
                                  scene.scene_sphere,
                                  rnd_dir_samples,
                                  rnd_pos_samples,
                                  light_state.origin,
                                  light_state.direction,
                                  emission_pdf_w,
                                  &direct_pdf_a,
                                  &cos_light);

    // lights are sampled uniformly
    emission_pdf_w /= light_count;
    direct_pdf_a /= light_count;

    GPU_ASSERT(emission_pdf_w > 0.f);
    GPU_ASSERT(direct_pdf_a > 0.f);

    light_state.throughput /= emission_pdf_w;
    light_state.path_length = 1;
    light_state.is_infinite_light = !is_finite(light);

    // prepare weights for the first light vertex, index 1
    // for the first vertex, we can set dBPT and dPDE immediately
    {
        /** Eq D.27 (light) */
        light_state.weights.weights.d_shared = direct_pdf_a / emission_pdf_w;
        /** Eq D.28 (light) */
        if (!is_delta(light)) {
            // if light is infinite, g factor is 1
            // see [VCM tech. rep. sec. 5.1]
            const float used_cos_light = is_finite(light) ? cos_light : 1.f;
            light_state.weights.weights.d_bpt = used_cos_light / emission_pdf_w;
        } else {
            // if light is delta, ind^bpt_0 evaluates to zero
            light_state.weights.weights.d_bpt = 0.f;
        }
        /** Eq D.29 (light) */
        light_state.weights.weights.d_pde = light_state.weights.weights.d_bpt * params.subpath_count_bpt;

        // index 0 is on light source
        light_state.weights.weights.prev_specular = false;
        light_state.weights.weights.prev_in_medium = false;
    }

    // init the boundary stack with the global medium
    gpuupbp_status ret = init_boundary_stack(scene, light_state.stack);
    if (ret != RET_SUCCESS) return ret;

    // add enclosing material and medium if present
    if (light.mat_id != -1 && light.med_id != -1) {
        ret = add_to_boundary_stack(scene, light.mat_id, light.med_id, light_state.stack);
        if (ret != RET_SUCCESS) return ret;
    }

    return RET_SUCCESS;
}

/// Adds beams to beams array
__forceinline__ __device__ void add_beams(const Ray& ray,
                                          const float3& throughput,
                                          unsigned int path_length,
                                          const unsigned int ray_sampling_flags,
                                          const VolSegmentArray& v_segment_array,
                                          const VolLiteSegmentArray& v_segment_lite_array,
                                          const unsigned int path_idx,
                                          const SubpathState& state)
{
    GPU_ASSERT(ray_sampling_flags == 0 || ray_sampling_flags == ORIGIN_IN_MEDIUM);

    float3 throughput_ = throughput;
    float ray_sample_for_pdf = 1.f;
    float ray_sample_rev_pdf = 1.f;

    LightBeam beam;
    OptixInstance instance;

    if (params.photon_beam_type == SHORT_BEAM) {
        for (unsigned int i = 0; i < v_segment_array.size(); i++) {
            const VolSegment* it = &v_segment_array[i];
            GPU_ASSERT(it->med_id >= 0);
            Medium& medium = params.scene.media[it->med_id];
            if (medium.has_scattering) {
                beam.medium = &medium;
                beam.ray.origin = ray.origin + ray.direction * it->dist_min;
                beam.ray.direction = ray.direction;
                beam.beam_length = fminf(it->dist_max - it->dist_min, medium.max_beam_length);
                beam.ray_sample_for_pdf = ray_sample_for_pdf;
                beam.ray_sample_rev_pdf = ray_sample_rev_pdf;
                beam.ray_sampling_flags = END_IN_MEDIUM;
                if (i == 0) beam.ray_sampling_flags |= ray_sampling_flags;
                beam.throughput_at_origin = throughput_;
                beam.path_idx = path_idx;
                beam.weights = state.weights;
                beam.path_length = path_length;
                beam.is_infinite_light = state.is_infinite_light;

                // beam should be complete before calling this function
                set_beam_instance(&beam, &instance);

                unsigned int beam_index = atomicAdd(params.light_beam_counter, 1);
                if (beam_index < params.light_beam_count) {
                    memcpy(params.light_beams + beam_index, &beam, sizeof(LightBeam));
                    memcpy(params.instances_beam + beam_index, &instance, sizeof(OptixInstance));
                }
            }
            throughput_ *= it->attenuation / it->ray_sample_pdf_for;
            ray_sample_for_pdf *= it->ray_sample_pdf_for;
            ray_sample_rev_pdf *= it->ray_sample_pdf_rev;
        }
    } else {
        // LONG_BEAM
        for (unsigned int i = 0; i < v_segment_lite_array.size(); i++) {
            const VolLiteSegment* it = &v_segment_lite_array[i];
            GPU_ASSERT(it->med_id >= 0);
            Medium& medium = params.scene.media[it->med_id];
            if (medium.has_scattering) {
                beam.medium = &medium;
                beam.ray.origin = ray.origin + ray.direction * it->dist_min;
                beam.ray.direction = ray.direction;
                beam.beam_length = fminf(it->dist_max - it->dist_min, medium.max_beam_length);
                beam.ray_sample_for_pdf = ray_sample_for_pdf;
                beam.ray_sample_rev_pdf = ray_sample_rev_pdf;
                beam.ray_sampling_flags = END_IN_MEDIUM;
                if (i == 0) beam.ray_sampling_flags |= ray_sampling_flags;
                beam.throughput_at_origin = throughput_;
                beam.path_idx = path_idx;
                beam.weights = state.weights;
                beam.path_length = path_length;
                beam.is_infinite_light = state.is_infinite_light;

                set_beam_instance(&beam, &instance);

                unsigned int beam_index = atomicAdd(params.light_beam_counter, 1);
                if (beam_index < params.light_beam_count) {
                    memcpy(params.light_beams + beam_index, &beam, sizeof(LightBeam));
                    memcpy(params.instances_beam + beam_index, &instance, sizeof(OptixInstance));
                }
            }
            throughput_ *= eval_attenuation(&medium, it->dist_max - it->dist_min);
            float segmentRaySampleRevPdf;
            float segmentRaySamplePdf = ray_sample_pdf(
                &medium, it->dist_min, it->dist_max, i == 0 ? ray_sampling_flags : 0, &segmentRaySampleRevPdf);
            ray_sample_for_pdf *= segmentRaySamplePdf;
            ray_sample_rev_pdf *= segmentRaySampleRevPdf;
        }
    }
}

/// Computes contribution of light sample to camera by splatting is onto the
/// framebuffer. Multiplies by throughput (obviously, as nothing is returned).
__forceinline__ __device__ gpuupbp_status connect_to_camera(const SubpathState& light_state,
                                                            const float3& hitpoint,
                                                            const BSDF& light_bsdf,
                                                            RNGState& state,
                                                            VolSegmentArray& volume_segments)
{
    // can only connect non-specular vertices
    GPU_ASSERT(!is_delta(light_bsdf));

    // get camera and direction to it
    const Camera& camera = params.scene.camera;
    float3 direction_to_camera = camera.origin - hitpoint;

    // check point is in front of camera
    if (dot(camera.direction, -direction_to_camera) <= 0.f) return RET_SUCCESS;

    // check it projects to the screen (and where)
    const float2 image_pos = world_to_raster(camera, hitpoint);
    if (!check_raster(camera, image_pos)) return RET_SUCCESS;

    // compute distance and normalize direction to camera
    const float dist_eye_sqr = dot(direction_to_camera, direction_to_camera);
    const float distance = sqrtf(dist_eye_sqr);
    direction_to_camera /= distance;

    // get the BSDF factor
    float cos_to_camera, bsdf_for_pdf_w, bsdf_rev_pdf_w, sin_theta;
    float3 bsdfFactor = evaluate(light_bsdf,
                                 direction_to_camera,
                                 cos_to_camera,
                                 params.scene.materials,
                                 &bsdf_for_pdf_w,
                                 &bsdf_rev_pdf_w,
                                 &sin_theta);

    if (is_black_or_negative(bsdfFactor)) return RET_SUCCESS;

    bsdf_rev_pdf_w *= light_bsdf.continuation_prob;

    GPU_ASSERT(bsdf_for_pdf_w > 0.f);
    GPU_ASSERT(bsdf_rev_pdf_w > 0.f);
    GPU_ASSERT(cos_to_camera > 0.f);

    // compute PDF conversion factor from image plane area to surface area
    const float cos_at_camera = dot(camera.direction, -direction_to_camera);
    const float image_point_to_camera_dist = camera.image_plane_dist / cos_at_camera;
    const float image_to_solid_angle_factor = image_point_to_camera_dist * image_point_to_camera_dist / cos_at_camera;
    const float image_to_surface_factor = image_to_solid_angle_factor * fabsf(cos_to_camera) / (distance * distance);

    // we put the virtual image plane at such a distance from the camera origin
    // that the pixel area is one and thus the image plane sampling PDF is 1.

    // the area PDF of aHitpoint as sampled from the camera is then equal to the
    // conversion factor from image plane area density to surface area density
    const float camera_pdf_a = image_to_surface_factor;
    GPU_ASSERT(camera_pdf_a > 0.f);

    // test occlusion
    volume_segments.clear();
    bool is_occluded;
    unsigned int ret = occluded(is_occluded,
                                hitpoint,
                                direction_to_camera,
                                distance,
                                light_state.stack,
                                is_in_medium(light_bsdf) ? ORIGIN_IN_MEDIUM : 0,
                                volume_segments,
                                state);
    if (ret != RET_SUCCESS) return ret;
    if (is_occluded) return ret;

    // Get attenuation from intersected media (if any)
    float ray_sample_rev_pdf = 1.f; // pl_{T,s-1}(y)
    float3 attenuation = make_float3(1.f);
    if (!volume_segments.empty()) {
        // Reverse PDF
        ray_sample_rev_pdf = accumulate_rev_pdf(volume_segments);

        // Attenuation (without PDF!)
        attenuation = accumulate_attenuation_without_pdf(volume_segments);
        if (!is_positive(attenuation)) return RET_SUCCESS;
    }

    // Compute MIS weight if not doing LT
    const VertexWeights& weight_l = light_state.weights.weights; // s-1

    // f_s considers a subpath where the last vertex is s - 1
    GPU_ASSERT(!is_delta(light_bsdf)); // we pass false
    float f_s = get_pde_factor(is_in_medium(light_bsdf),
                               false,
                               weight_l.ray_sample_for_pdf_inv,    // 1           /pr_{T,s-1}(x)
                               weight_l.ray_sample_for_pdfs_ratio, // pr{L,s-1}(x)/pr_{T,s-1}(x)
                               1.f / ray_sample_rev_pdf,           // 1           /pl_{T,s-1}(x)
                               weight_l.ray_sample_rev_pdfs_ratio, // pl{L,s-1}(x)/pl_{T,s-1}(x)
                               sin_theta);

    // Note the division by mLightPathCount, which is the number of samples this
    // technique uses. This division also appears a few lines below in the
    // framebuffer accumulation.

    // bpt_{s-1} considers s-2 and s-1
    const float bpt_s_sub_1 = !weight_l.prev_specular; // whether s-2 is specular
    // s-1 is not delta (otherwise this function is not called)

    /** Eq D.38 */
    // 1 / bpt_s = 1 (considers s-1 and s)
    // ignore division by n_\bpt (which is only non-one if BPT is disabled)
    const float w_light = (camera_pdf_a / params.light_subpath_count) * ray_sample_rev_pdf *
                          (f_s + bpt_s_sub_1 * light_state.weights.weights.d_shared +
                           (bsdf_rev_pdf_w / weight_l.ray_sample_rev_pdf_inv) * light_state.weights.weights.d_bpt);

    /** Eq 4.12 */
    const float w_local = 1.f;

    /** Eq D.39 */
    const float w_camera = 0.f;

    /** Eq 4.18 */
    const float mis_weight = 1.f / (w_light + w_local + w_camera);

    const float surface_to_image_factor = 1.f / image_to_surface_factor;

    // we divide the contribution by surfaceToImageFactor to convert the
    // (already divided) PDF from surface area to image plane area, w.r.t. which
    // the pixel integral is actually defined. We also divide by the number of
    // samples this technique makes, which is equal to the number of
    // light sub-paths
    float3 contrib = mis_weight * light_state.throughput * bsdfFactor * attenuation /
                     (params.light_subpath_count * surface_to_image_factor);

    if (is_black_or_negative(contrib)) return RET_SUCCESS;

    // an atomic write is required, as multiple light threads may access
    // the same position in the framebuffer
    atomic_add_color(params.framebuffer, image_pos, contrib);

    return RET_SUCCESS;
}

#endif // KERNEL_FUNCTS_LIGHT_CUH
