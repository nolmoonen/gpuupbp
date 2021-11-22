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

#include "../shared/launch_params.h"
#include "../shared/vec_math.h"
#include "curand_kernel.h"
#include "functs_light.cuh"
#include "functs_shared.cuh"
#include "intersector.cuh"
#include "medium.cuh"
#include "optix_util.cuh"
#include "params_def.cuh"
#include "rng.cuh"
#include "sarray.cuh"
#include "types.cuh"

#include <optix.h>

extern "C" __global__ void __raygen__trace_light()
{
    // path segments intersecting media (up to scattering point)
    VolSegmentArray volume_segments;
    // lite path segments intersecting media
    // (up to intersection with solid surface)
    VolLiteSegmentArray lite_volume_segments;

    // index of the previous stored index in the path
    // initialize to special value to indicate no previous vertex exists yet
    unsigned int prev_idx = UINT_MAX;

    // index of the first stored index in the path
    // initialize to special value to indicate no first vertex exists yet
    unsigned int first_idx = UINT_MAX;

    const unsigned int path_idx = optixGetLaunchIndex().x;
    // initialize curand. unique seed for every iteration and path idx
    RNGState rng_state;
    rng_init(params.iteration, path_idx, rng_state);

    // whether this thread should store light beams
    const bool store_beams = (params.do_bb1d && path_idx < params.bb1d_used_light_sub_path_count) || params.do_bp2d;

    // maintain a status code for functions that can fail
    gpuupbp_status ret = RET_SUCCESS;

    // generate light path origin and direction (inits path length as 1)
    SubpathState state;
    ret = generate_light_sample(state, params.scene, rng_state);

    // in attenuating media, the ray can never travel from infinity
    if (state.is_infinite_light && get_global_medium_ptr(params.scene)->has_attenuation) {
        // we are done, save and exit
        params.light_subpath_starts[path_idx] = first_idx;
        return;
    }

    // we assume that the light is on surface
    bool origin_in_medium = false;

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

        // store beam if required
        if (store_beams) {
            add_beams(ray,
                      state.throughput,
                      state.path_length,
                      origin_in_medium ? ORIGIN_IN_MEDIUM : 0,
                      volume_segments,
                      lite_volume_segments,
                      path_idx,
                      state);
        }

        if (!intersected) break;

        GPU_ASSERT(isect.is_valid());

        // attenuate by intersected media (if any)
        float ray_sample_for_pdf = 1.f; // pr_{T,i  }(y)
        float ray_sample_rev_pdf = 1.f; // pl_{T,i-1}(y)
        if (!volume_segments.empty()) {
            ray_sample_for_pdf = accumulate_for_pdf(volume_segments);
            GPU_ASSERT(ray_sample_for_pdf > 0.f);

            ray_sample_rev_pdf = accumulate_rev_pdf(volume_segments);
            GPU_ASSERT(ray_sample_rev_pdf > 0.f);

            // attenuation
            state.throughput *= accumulate_attenuation_without_pdf(volume_segments) / ray_sample_for_pdf;
        }

        if (is_black_or_negative(state.throughput)) break;

        // now that we have pl_{T,i-1}(y) == pl_{T,i-1}(x) we can evaluate f_i
        // note that this term is not present in the initialization of
        // dBPT and dPDE, hence we skip the first iteration
        if (state.path_length > 1u) {
            float f_i = get_pde_factor(state.weights.weights.prev_in_medium, // i - 1 in medium
                                       state.weights.weights.prev_specular,  // i - 1 specular
                                       // 1 / pr_{T,i-1}(x)
                                       state.weights.weights.ray_sample_for_pdf_inv,
                                       // prob_r{i-1}/pr_{T,i-1}(x)
                                       state.weights.weights.ray_sample_for_pdfs_ratio,
                                       // 1 / pl_{T,i-1}(x)
                                       1.f / ray_sample_rev_pdf,
                                       // prob_l{i-1}/pl_{T,i-1}(x)
                                       state.weights.weights.ray_sample_rev_pdfs_ratio,
                                       state.weights.last_sin_theta); // sin(i-2,i-1,i)

            /** Eq 4.48 (partly) */
            state.weights.weights.d_bpt = state.weights.d_bpt_a * f_i + state.weights.d_bpt_b;
            /** Eq 4.49 (partly) */
            state.weights.weights.d_pde = state.weights.d_pde_a * f_i + state.weights.d_pde_b;
        }

        // multiply by (1 / pr_{T,i}(y))
        {
            /** Eq 4.47 (partly) */
            state.weights.weights.d_shared /= ray_sample_for_pdf;
            /** Eq 4.48 (partly) */
            state.weights.weights.d_bpt /= ray_sample_for_pdf;
            /** Eq 4.49 (partly) */
            state.weights.weights.d_pde /= ray_sample_for_pdf;
        }

        // prepare scattering function at the hitpoint (BSDF/phase depending on
        // whether the hitpoint is at surface or in media, the isect knows)
        BSDF bsdf;
        bsdf_setup(bsdf, ray, isect, params.scene, BSDF::FROM_LIGHT, relative_ior(params.scene, isect, state.stack));

        // e.g. hitting surface too parallel with tangent plane
        if (!is_valid(bsdf)) break;

        // for homogeneous media, these are the same
        float ray_sample_for_pdfs_ratio = 0.f; // prob_r{i}/pr_{T,i}(y)
        float ray_sample_rev_pdfs_ratio = 0.f; // prob_l{i}/pl_{T,i}(y)
        if (is_in_medium(bsdf)) {
            float val = 1.f / med(bsdf)->med_ptr->min_positive_attenuation_coeff_comp;
            ray_sample_for_pdfs_ratio = val;
            ray_sample_rev_pdfs_ratio = val;
        }

        // compute hitpoint
        const float3 hit_point = ray.origin + ray.direction * isect.dist;

        origin_in_medium = isect.is_in_medium();

        // now that we have the hit point, we can complete the MIS quantities
        {
            // cosine of normal at i and direction from i-1 to i for surface,
            // 1 for medium
            const float d = d_factor(bsdf);

            /** Eq 4.47 */
            // complete pr_i by completing the g_i factor
            if (!(state.path_length == 1 && state.is_infinite_light)) {
                // infinite lights use MIS handled via solid angle integration,
                // so do not divide by the distance for such lights
                // see [VCM tech. rep. sec. 5.1]
                state.weights.weights.d_shared *= isect.dist * isect.dist;
            }
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

        /** store vertex y_i */
        // unless BSDF is purely specular, which prevents vertex
        // connections and merging
        if (!is_delta(bsdf)) {
            // path length is 1 at first iteration of this loop
            LightVertex light_vertex;
            light_vertex.flags = 0;
            light_vertex.hitpoint = hit_point;
            light_vertex.throughput = state.throughput;
            light_vertex.path_length = state.path_length;
            light_vertex.bsdf = bsdf;
            // special value to indicate no next vertex exists (yet)
            light_vertex.next_idx = UINT_MAX;

            light_vertex.set_in_medium(origin_in_medium);

            // determine whether the vertex is in medium behind real geometry
            light_vertex.set_behind_surf(false);
            if (origin_in_medium && !state.stack.is_empty()) {
                int mat_id = state.stack.top().mat_id;
                if (mat_id >= 0) {
                    const Material& mat = get_material(params.scene, mat_id);
                    if (mat.real) light_vertex.set_behind_surf(true);
                }
            }

            light_vertex.weights = state.weights.weights;

            // atomicall increment counter to obtain index to store vertex
            unsigned int vertex_index = atomicAdd(params.light_vertex_counter, 1);

            // assert that we will not be writing outside of the array
            // a warning is printed by the host if the counter exceeds the max
            if (vertex_index < params.light_vertex_count) {
                // store the vertex
                memcpy(params.light_vertices + vertex_index, &light_vertex, sizeof(light_vertex));

                // prepare the aabb
                unsigned int* vertex_counter;
                float rad;
                OptixAabb aabb;
                unsigned int* indices;
                OptixAabb* aabb_p;
                if (origin_in_medium) {
                    vertex_counter = params.light_vertex_medium_counter;
                    rad = params.rad_point_med;
                    aabb_p = params.aabb_medium;
                    indices = params.indices_medium;
                } else {
                    vertex_counter = params.light_vertex_surface_counter;
                    rad = params.radius_surf;
                    aabb_p = params.aabb_surface;
                    indices = params.indices_surface;
                }

                // index in surface and medium arrays
                unsigned int vertex_index_sur_med = atomicAdd(vertex_counter, 1);

                // set aabb in the surface or medium array
                aabb.minX = light_vertex.hitpoint.x - rad;
                aabb.minY = light_vertex.hitpoint.y - rad;
                aabb.minZ = light_vertex.hitpoint.z - rad;
                aabb.maxX = light_vertex.hitpoint.x + rad;
                aabb.maxY = light_vertex.hitpoint.y + rad;
                aabb.maxZ = light_vertex.hitpoint.z + rad;
                memcpy(aabb_p + vertex_index_sur_med, &aabb, sizeof(OptixAabb));

                // set index in surface or medium array
                indices[vertex_index_sur_med] = vertex_index;

                // if previous vertex, set it next index
                if (prev_idx != UINT_MAX) {
                    params.light_vertices[prev_idx].next_idx = vertex_index;
                }

                // if this is the first index, set the variable
                if (first_idx == UINT_MAX) {
                    first_idx = vertex_index;
                }

                prev_idx = vertex_index;
            }
        }

        // connect to camera, unless scattering function is purely specular or
        // we are not allowed to connect from surface
        if (params.do_bpt && !is_delta(bsdf) && (is_in_medium(bsdf) || params.connect_to_camera_from_surf)) {
            if (state.path_length + 1 >= params.min_path_length) {
                ret = connect_to_camera(state, hit_point, bsdf, rng_state, volume_segments);
                if (ret != RET_SUCCESS) break;
            }
        }

        // terminate if the path would become too long after scattering
        if (state.path_length + 2 > params.max_path_length) break;

        // continue random walk
        bool cont;
        ret = sample_scattering(cont, bsdf, hit_point, isect, state, rng_state);
        if (ret != RET_SUCCESS) break;
        if (!cont) break;
    }

    // set the index of the first vertex in the global array
    params.light_subpath_starts[path_idx] = first_idx;
}
