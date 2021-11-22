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

#include "../shared/frame.h"
#include "../shared/light_vertex.h"
#include "../shared/vec_math.h"
#include "frame_buffer.cuh"
#include "functs_camera.cuh"
#include "functs_shared.cuh"
#include "intersector.cuh"
#include "medium.cuh"
#include "optix_util.cuh"
#include "params_def.cuh"
#include "pde/functs_bb1d.cuh"
#include "pde/functs_bp2d.cuh"
#include "pde/functs_pb2d.cuh"
#include "pde/functs_pp.cuh"
#include "pde/functs_pp3d.cuh"
#include "pde/functs_surf.cuh"
#include "rng.cuh"
#include "sarray.cuh"
#include "types.cuh"

#include <optix.h>

extern "C" __global__ void __raygen__trace_camera()
{
    // path segments intersecting media (up to scattering point)
    VolSegmentArray volume_segments;
    // lite path segments intersecting media
    // (up to intersection with solid surface)
    VolLiteSegmentArray lite_volume_segments;

    const unsigned int path_idx = optixGetLaunchIndex().x;
    // initialize curand. unique seed for every iteration and path idx
    RNGState rng_state;
    rng_init(params.iteration, path_idx, rng_state);

    // maintain a status code for functions that can fail
    gpuupbp_status ret = RET_SUCCESS;

    // generate camera path origin and direction (inits path length as 1)
    SubpathState state;
    float2 screen_sample;
    ret = generate_camera_sample(screen_sample, path_idx, state, params.scene, rng_state);

    // the color contribution for the pixel corresponding to this thread
    float3 color = make_float3(0.f);

    // we assume that the camera is on surface
    bool origin_in_medium = false;

    bool only_spec_surf = (params.estimator_techniques & (PREVIOUS | COMPATIBLE)) != 0;

    // skip loop if ret was not success
    for (; ret == RET_SUCCESS; state.path_length++) {
        // prepare ray
        Ray ray(state.origin, state.direction);
        Intersection isect(INFTY);

        // trace ray
        volume_segments.clear();
        lite_volume_segments.clear();
        bool intersected;
        ret = intersect(intersected,
                        ray,
                        origin_in_medium ? ORIGIN_IN_MEDIUM : 0,
                        isect,
                        state.stack,
                        volume_segments,
                        lite_volume_segments,
                        rng_state);
        if (ret != RET_SUCCESS) break;

        // vertex merging: point x beam 2D
        if (params.build_pb2d && *params.light_vertex_medium_counter > 0) {
            float3 contrib;
            if (isect.is_on_surface() || params.camera_beam_type == SHORT_BEAM || !intersected) {
                contrib = eval_pb2d_segments(ray,
                                             volume_segments,
                                             static_cast<unsigned int>(origin_in_medium ? ORIGIN_IN_MEDIUM : 0),
                                             state.path_length,
                                             state.weights);
            } else {
                contrib = eval_pb2d_segments(ray,
                                             lite_volume_segments,
                                             static_cast<unsigned int>(origin_in_medium ? ORIGIN_IN_MEDIUM : 0),
                                             state.path_length,
                                             state.weights);
            }
            color += state.throughput * params.pb2d_normalization * contrib;
        }

        // vertex merging: beam x beam 1D
        if (params.build_bb1d && *params.light_beam_counter > 0) {
            float3 contrib;
            if (isect.is_on_surface() || params.camera_beam_type == SHORT_BEAM || !intersected) {
                contrib = eval_bb1d_segments(ray,
                                             volume_segments,
                                             static_cast<unsigned int>(origin_in_medium ? ORIGIN_IN_MEDIUM : 0),
                                             state.path_length,
                                             state.weights);
            } else {
                contrib = eval_bb1d_segments(ray,
                                             lite_volume_segments,
                                             static_cast<unsigned int>(origin_in_medium ? ORIGIN_IN_MEDIUM : 0),
                                             state.path_length,
                                             state.weights);
            }
            color += state.throughput * params.bb1d_normalization * contrib;
        }

        if (!intersected) {
            // we cannot end yet
            if (state.path_length < params.min_path_length) break;

            // get background light
            if (params.scene.background_light_idx == UINT_MAX) break;

            // in attenuating media the ray can never travel to infinity
            if (get_global_medium_ptr(params.scene)->has_attenuation) break;
        }

        // attenuate by intersected media (if any)
        float ray_sample_for_pdf = 1.f; // pr_{T,i  }(z)
        float ray_sample_rev_pdf = 1.f; // pl_{T,i-1}(z)
        if (!volume_segments.empty()) {
            ray_sample_for_pdf = accumulate_for_pdf(volume_segments);
            GPU_ASSERT(ray_sample_for_pdf > 0.f);

            ray_sample_rev_pdf = accumulate_rev_pdf(volume_segments);
            GPU_ASSERT(ray_sample_rev_pdf > 0.f);

            // attenuation
            state.throughput *= accumulate_attenuation_without_pdf(volume_segments) / ray_sample_for_pdf;
        }

        if (is_black_or_negative(state.throughput)) break;

        // now that we have pl_{T,i-1}(z) == pr_{T,i-1}(x) we can evaluate f_i
        // note that this term is not present in the initialization of
        // dBPT and dPDE, hence we skip the first iteration
        if (state.path_length > 1u) {
            float f_i = get_pde_factor(state.weights.weights.prev_in_medium, // i - 1 in medium
                                       state.weights.weights.prev_specular,  // i - 1 specular
                                       // 1 / pr_{T,i-1}(x)
                                       1.f / ray_sample_rev_pdf,
                                       // prob_r{i-1}/pr_{T,i-1}(x)
                                       state.weights.weights.ray_sample_rev_pdfs_ratio,
                                       // 1 / pl_{T,i-1}(x)
                                       state.weights.weights.ray_sample_for_pdf_inv,
                                       // prob_l{i-1}/pl_{T,i-1}(x)
                                       state.weights.weights.ray_sample_for_pdfs_ratio,
                                       state.weights.last_sin_theta); // sin(i-2,i-1,i)

            /** Eq 4.48 (partly) */
            state.weights.weights.d_bpt = state.weights.d_bpt_a * f_i + state.weights.d_bpt_b;
            /** Eq 4.49 (partly) */
            state.weights.weights.d_pde = state.weights.d_pde_a * f_i + state.weights.d_pde_b;
        }

        // multiply by (1 / pr_{T,i}(z))
        {
            /** Eq 4.47 (partly) */
            state.weights.weights.d_shared /= ray_sample_for_pdf;
            /** Eq 4.48 (partly) */
            state.weights.weights.d_bpt /= ray_sample_for_pdf;
            /** Eq 4.49 (partly) */
            state.weights.weights.d_pde /= ray_sample_for_pdf;
        }

        if (!intersected) {
            // if we have not intersected and have a background light, evaluate
            // it. note that this may also be evaluated when not doing bpt
            color += state.throughput * get_light_radiance(params.scene.lights[params.scene.background_light_idx],
                                                           state,
                                                           make_float3(0.f),
                                                           ray_sample_for_pdf,
                                                           ray_sample_rev_pdf);
            break;
        }

        GPU_ASSERT(isect.is_valid());

        // prepare scattering function at the hitpoint (BSDF/phase depending on
        // whether the hitpoint is at surface or in media, the isect knows)
        BSDF bsdf;
        bsdf_setup(bsdf, ray, isect, params.scene, BSDF::FROM_CAMERA, relative_ior(params.scene, isect, state.stack));

        // e.g. hitting surface too parallel with tangent plane
        if (!is_valid(bsdf)) break;

        // for homogeneous media, these are the same
        float ray_sample_for_pdfs_ratio = 0.f; // prob_r{i}/pr_{T,i}(z)
        float ray_sample_rev_pdfs_ratio = 0.f; // prob_l{i}/pl_{T,i}(z)
        if (is_in_medium(bsdf)) {
            float val = 1.f / med(bsdf)->med_ptr->min_positive_attenuation_coeff_comp;
            ray_sample_for_pdfs_ratio = val;
            ray_sample_rev_pdfs_ratio = val;
        }

        // compute hitpoint
        float3 hitPoint = ray.origin + ray.direction * isect.dist;

        origin_in_medium = isect.is_in_medium();

        // now that we have the hit point, we can complete the MIS quantities
        {
            // cosine of normal at i and direction from i-1 to i for surface,
            // 1 for medium
            const float d = d_factor(bsdf);

            /** Eq 4.47 */
            // complete pr_i by completing the g_i factor
            state.weights.weights.d_shared *= isect.dist * isect.dist;
            state.weights.weights.d_shared /= d;

            // do not multiply the other quantities by the squared
            // distance as it cancels out with the squared distance of the
            // gl_{i-1} term

            /** Eq 4.48 and 4.49 */
            state.weights.weights.d_bpt /= d;
            state.weights.weights.d_pde /= d;

            // cache the ray sample probabilities
            state.weights.weights.ray_sample_for_pdf_inv = 1.f / ray_sample_for_pdf;
            state.weights.weights.ray_sample_rev_pdf_inv = 1.f / ray_sample_rev_pdf;
            state.weights.weights.ray_sample_for_pdfs_ratio = ray_sample_for_pdfs_ratio;
            state.weights.weights.ray_sample_rev_pdfs_ratio = ray_sample_rev_pdfs_ratio;
        }

        // light source has been hit; terminate afterwards, since
        // our light sources do not have reflective properties
        if (isect.light_id >= 0) {
            // we cannot end yet
            if (state.path_length < params.min_path_length) break;

            // get hit light
            const AbstractLight& light = params.scene.lights[isect.light_id];

            // add its contribution
            // note that this may also be evaluated when not doing bpt
            color +=
                state.throughput * get_light_radiance(light, state, hitPoint, ray_sample_for_pdf, ray_sample_rev_pdf);
            break;
        }

        // terminate if eye sub-path is too long for connections or merging
        if (state.path_length >= params.max_path_length) break;

        // vertex connection: connect to a light source
        if (params.do_bpt && !is_delta(bsdf) && state.path_length + 1 >= params.min_path_length &&
            params.scene.light_count > 0 && (is_in_medium(bsdf) || !only_spec_surf)) {
            float3 contrib;
            ret = direct_illumination(contrib, state, hitPoint, bsdf, rng_state, volume_segments);
            if (ret != RET_SUCCESS) break;
            color += state.throughput * contrib;
        }

        // vertex connection: connect to light vertices
        if (params.do_bpt && !is_delta(bsdf) && (is_in_medium(bsdf) || !only_spec_surf)) {
            // Determine whether the vertex is in medium behind real geometry
            bool behind_surf = false;
            if (is_in_medium(bsdf) && !state.stack.is_empty()) {
                int mat_id = state.stack.top().mat_id;
                if (mat_id >= 0) {
                    const Material& mat = get_material(params.scene, mat_id);
                    if (mat.real) behind_surf = true;
                }
            }

            // obtain the index to a random path to connect this vertex to
            const int path_idx_mod = get_rnd(rng_state) * params.light_subpath_count;

            // obtain the index of the first vertex of the path
            unsigned int vertex_idx = params.light_subpath_starts[path_idx_mod];

            // whether an error in the loop below has occurred, and we must
            // break the outer loop
            bool encountered_error = false;

            // keep iterating the linked list until no next vertex index exists
            while (vertex_idx != UINT_MAX) {
                const LightVertex& light_vertex = params.light_vertices[vertex_idx];

                const unsigned int path_length = light_vertex.path_length + 1 + state.path_length;

                // Light vertices are stored in increasing path length
                // order; once we go above the max path length, we can
                // skip the rest
                if (path_length > params.max_path_length) break;

                const bool smaller_than_allowed = path_length < params.min_path_length;

                // don't try connect vertices in different media with
                // real geometry
                const bool different_media = is_in_medium(light_vertex.bsdf) && is_in_medium(bsdf) &&
                                             med(light_vertex.bsdf)->med_ptr != med(bsdf)->med_ptr &&
                                             (light_vertex.is_behind_surf() || behind_surf);

                if (!(smaller_than_allowed || different_media)) {
                    float3 contrib;
                    ret = connect_vertices(contrib, light_vertex, bsdf, hitPoint, state, rng_state, volume_segments);
                    if (ret != RET_SUCCESS) {
                        encountered_error = true;
                        break;
                    }
                    color += state.throughput * light_vertex.throughput * contrib;
                }

                vertex_idx = light_vertex.next_idx;
            }

            if (encountered_error) break;
        }

        // vertex merging: surface photon mapping
        if (params.build_surf && *params.light_vertex_surface_counter > 0 && is_on_surface(bsdf) && !is_delta(bsdf) &&
            !only_spec_surf) {
            color += state.throughput * params.surf_normalization * eval_pp2d(hitPoint, &state, &bsdf);
        }

        // vertex merging: point x point 3D
        if (params.build_pp3d && *params.light_vertex_medium_counter > 0 && is_in_medium(bsdf) && !is_delta(bsdf)) {
            color += state.throughput * params.pp3d_normalization * eval_pp3d(hitPoint, &state, &bsdf);
        }

        // vertex merging: beam x point 2D
        if (params.build_bp2d && *params.light_beam_counter > 0 && is_in_medium(bsdf)) {
            color += state.throughput * params.bp2d_normalization * eval_bp2d(hitPoint, &state, &bsdf);
        }

        // continue random walk
        bool cont;
        ret = sample_scattering(cont, bsdf, hitPoint, isect, state, rng_state);
        if (ret != RET_SUCCESS) break;
        if (!cont) break;

        if (is_on_surface(bsdf)) {
            if (!state.weights.weights.prev_specular) {
                if (only_spec_surf || (params.estimator_techniques & SPECULAR_ONLY)) {
                    break;
                }
            }
        } else {
            if (params.estimator_techniques & SPECULAR_ONLY) {
                break;
            }

            if (only_spec_surf) {
                if (params.estimator_techniques & COMPATIBLE) {
                    only_spec_surf = false;
                } else {
                    break;
                }
            }
        }
    }

    // only contributions are added before any critical errors occurred, which
    // are valid. hence, it is okay to add the running color to the framebuffer
    // if an error occurred

    // finally, add the accumulated color to the framebuffer
    add_color(params.framebuffer, screen_sample, color);
}
