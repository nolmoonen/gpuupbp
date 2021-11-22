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

#include "renderer.hpp"
#include "../host/exception.hpp"
#include "../misc/logger.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

void Renderer::run_iteration(int iteration)
{
#ifdef DBG_CORE_UTIL
    // elapsed milliseconds for both kernels
    float light_ms, camera_ms;
    // trace counts for both kernels
    uint64_t light_trace_count, camera_trace_count;
#endif
    timer.start(); // start of light tracing

    // clear device framebuffer
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(params.framebuffer.color), 0, screen_pixel_count * sizeof(float3)));

    /** light tracing */
    unsigned int light_vertices_found; // number of vertices generated
    unsigned int light_vertices_found_surface;
    unsigned int light_vertices_found_medium;
    unsigned int light_beams_found; // number of beams generated
    if (params.trace_light_paths) {
        // radius reduction (1st iteration has aIteration == 0, thus offset)
        const float effective_iteration = 1.f + static_cast<float>(iteration);
        // BB1D
        float radius_bb1d = bb1d_radius_initial * powf(1.f + static_cast<float>(iteration) *
                                                                 bb1d_used_light_subpath_count / light_subpath_count,
                                                       config->bb1d_radius_alpha - 1.f);
        radius_bb1d = fmaxf(radius_bb1d, 1e-7f); // numeric stability
        // BP2D
        float radius_bp2d = bp2d_radius_initial * powf(effective_iteration, (config->bp2d_radius_alpha - 1.f) * .5f);
        radius_bp2d = fmaxf(radius_bp2d, 1e-7f); // numeric stability
        const float radius_bp2d_sqr = radius_bp2d * radius_bp2d;
        // PB2D
        float radius_pb2d = pb2d_radius_initial * powf(effective_iteration, (config->pb2d_radius_alpha - 1.f) * .5f);
        radius_pb2d = fmaxf(radius_pb2d, 1e-7f); // numeric stability
        const float radius_pb2d_sqr = radius_pb2d * radius_pb2d;
        // PP3D
        float radius_pp3d =
            pp3d_radius_initial * powf(effective_iteration, (config->pp3d_radius_alpha - 1.f) * (1.f / 3.f));
        radius_pp3d = fmaxf(radius_pp3d, 1e-7f); // numeric stability
        const float radius_pp3d_cube = radius_pp3d * radius_pp3d * radius_pp3d;
        // SURF
        float radius_surf = surf_radius_initial * powf(effective_iteration, (config->surf_radius_alpha - 1.f) * .5f);
        radius_surf = fmaxf(radius_surf, 1e-7f); // numeric stability
        const float radius_surf_sqr = radius_surf * radius_surf;

        params.mis_factor_bb1d = .5f * radius_bb1d * params.subpath_count_bb1d;
        params.mis_factor_bp2d = (PI_F * radius_bp2d_sqr) * params.subpath_count_bp2d;
        params.mis_factor_pb2d = (PI_F * radius_pb2d_sqr) * params.subpath_count_pb2d;
        params.mis_factor_pp3d = (4.f / 3.f) * (PI_F * radius_pp3d_cube) * params.subpath_count_pp3d;
        params.mis_factor_surf = (PI_F * radius_surf_sqr) * params.subpath_count_surf;

        // factor used to normalize vertex merging contribution.
        // beam-based does not include kernel as this kernel is not constant
        params.bb1d_normalization = 1.f / bb1d_used_light_subpath_count;
        params.bp2d_normalization = 1.f / static_cast<float>(light_subpath_count);
        params.pb2d_normalization = 1.f / static_cast<float>(light_subpath_count);
        params.pp3d_normalization = 1.f / params.mis_factor_pp3d;
        params.surf_normalization = 1.f / params.mis_factor_surf;

        params.radius_bb1d = radius_bb1d;
        params.radius_pb2d = radius_pb2d;
        params.radius_surf = radius_surf;
        params.radius_pp3d = radius_pp3d;
        params.radius_bp2d = radius_bp2d;
        params.rad_point_med = 0.f;
        if (params.do_pp3d) {
            params.rad_point_med = std::max(params.rad_point_med, radius_pp3d);
        }
        if (params.do_pb2d) {
            params.rad_point_med = std::max(params.rad_point_med, radius_pb2d);
        }
        params.rad_beam = 0.f;
        if (params.do_bb1d) {
            params.rad_beam = std::max(params.rad_beam, radius_bb1d);
        }
        if (params.do_bp2d) {
            params.rad_beam = std::max(params.rad_beam, radius_bp2d);
        }
        params.radius_bb1d_2 = radius_bb1d * radius_bb1d;
        params.radius_bp2d_2 = radius_bp2d * radius_bp2d;
        params.radius_pb2d_2 = radius_pb2d * radius_pb2d;
        params.radius_pp3d_2 = radius_pp3d * radius_pp3d;
        params.radius_surf_2 = radius_surf * radius_surf;
        params.iteration = iteration;

        // reset counters
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(params.light_vertex_counter), 0, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(params.light_vertex_surface_counter), 0, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(params.light_vertex_medium_counter), 0, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(params.light_beam_counter), 0, sizeof(unsigned int)));

        // note: no need to memset memory to zero, as only the data
        // that is generated is copied and used

        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_params),
                                   &params,
                                   sizeof(LaunchParams),
                                   cudaMemcpyHostToDevice,
                                   optix::DeviceState::get_instance()->stream));
#ifdef DBG_CORE_UTIL
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(dTraceCount), 0, mLightSubPathCount * sizeof(uint64_t)));
        // measure precise time
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
#endif
        OPTIX_CHECK(optixLaunch(pipeline_trace_light,
                                optix::DeviceState::get_instance()->stream,
                                reinterpret_cast<CUdeviceptr>(d_params),
                                sizeof(LaunchParams),
                                &sbt_trace_light,
                                light_subpath_count,
                                1,
                                1));
        CUDA_SYNC_CHECK(); // make sure launch is done before copying back
#ifdef DBG_CORE_UTIL
        // record the milliseconds
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        light_ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&light_ms, start, stop));
        // get the count
        auto* trace_count = static_cast<uint64_t*>(malloc(sizeof(uint64_t) * mLightSubPathCount));
        CUDA_CHECK(cudaMemcpy(trace_count,
                              reinterpret_cast<void*>(dTraceCount),
                              sizeof(uint64_t) * mLightSubPathCount,
                              cudaMemcpyDeviceToHost));
        light_trace_count = 0;
        for (uint32_t i = 0; i < mLightSubPathCount; i++) {
            light_trace_count += trace_count[i];
        }
        free(trace_count);
#endif

        CUDA_CHECK(cudaMemcpy(&light_vertices_found,
                              reinterpret_cast<void*>(params.light_vertex_counter),
                              sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&light_vertices_found_surface,
                              reinterpret_cast<void*>(params.light_vertex_surface_counter),
                              sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&light_vertices_found_medium,
                              reinterpret_cast<void*>(params.light_vertex_medium_counter),
                              sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&light_beams_found,
                              reinterpret_cast<void*>(params.light_beam_counter),
                              sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));

        // report if more elements are found than memory allocated, but do not
        // exit or throw error
        if (light_vertices_found > params.light_vertex_count) {
            fprintf(stderr, "error: device finds more vertices than memory allocated");
        }
        if (light_beams_found > params.light_beam_count) {
            fprintf(stderr, "error: device finds more beams than memory allocated");
        }
    }

    // stop light trace time, start build time
    float time_light_trace = timer.lap();

    /** build data structures */
    CUdeviceptr d_tmp_buffer = 0;
    size_t tmp_buffer_size = 0;

    // counter-intuitively, this ias is too large to compress
    optix::accel_struct ias_beams{};
    if ((build_bb1d || build_bp2d) && light_beams_found > 0) {
        ias_beams.accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        ias_beams.accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        ias_beams.input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        ias_beams.input.instanceArray.instances = reinterpret_cast<CUdeviceptr>(params.instances_beam);
        ias_beams.input.instanceArray.numInstances = light_beams_found;

        optix::complete_creation(
            optix::DeviceState::get_instance(), &ias_beams, &d_tmp_buffer, &tmp_buffer_size, false, false);
    }

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tmp_buffer)));
    d_tmp_buffer = 0;
    tmp_buffer_size = 0;

    optix::accel_struct gas_point_surface{};
    if (build_surf && light_vertices_found_surface > 0) {
        gas_point_surface.accel_options.buildFlags =
            OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        gas_point_surface.accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        gas_point_surface.input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        gas_point_surface.input.customPrimitiveArray.aabbBuffers = reinterpret_cast<CUdeviceptr*>(&params.aabb_surface);
        gas_point_surface.input.customPrimitiveArray.numPrimitives = light_vertices_found_surface;
        uint32_t flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        gas_point_surface.input.customPrimitiveArray.flags = &flags;
        gas_point_surface.input.customPrimitiveArray.numSbtRecords = 1;

        optix::complete_creation(
            optix::DeviceState::get_instance(), &gas_point_surface, &d_tmp_buffer, &tmp_buffer_size, false, true);
    }

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tmp_buffer)));
    d_tmp_buffer = 0;
    tmp_buffer_size = 0;

    optix::accel_struct gas_point_medium{};
    if ((build_pb2d || build_pp3d) && light_vertices_found_medium > 0) {
        gas_point_medium.accel_options.buildFlags =
            OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        gas_point_medium.accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        gas_point_medium.input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        gas_point_medium.input.customPrimitiveArray.aabbBuffers = reinterpret_cast<CUdeviceptr*>(&params.aabb_medium);
        gas_point_medium.input.customPrimitiveArray.numPrimitives = light_vertices_found_medium;
        uint32_t flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        gas_point_medium.input.customPrimitiveArray.flags = &flags;
        gas_point_medium.input.customPrimitiveArray.numSbtRecords = 1u;

        optix::complete_creation(
            optix::DeviceState::get_instance(), &gas_point_medium, &d_tmp_buffer, &tmp_buffer_size, false, true);
    }

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tmp_buffer)));
    d_tmp_buffer = 0;
    tmp_buffer_size = 0;

    /** build top level gas */
    optix::Instance instance = {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}};
    OptixInstance instances[2];
    memset(instances, 0, sizeof(OptixInstance) * 2);
    uint32_t inst_idx = 0;

    if (build_surf && light_vertices_found_surface > 0) {
        uint32_t i = inst_idx++;
        instances[i].traversableHandle = gas_point_surface.handle;
        instances[i].flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
        instances[i].instanceId = 0; // not used
        instances[i].sbtOffset = static_cast<uint32_t>(scene_loader->record_count * RAY_TYPE_COUNT);
        instances[i].visibilityMask = GEOM_MASK_PP2D;
        memcpy(instances[i].transform, instance.transform, sizeof(float) * 12);
    }
    if ((build_pb2d || build_pp3d) && light_vertices_found_medium > 0) {
        uint32_t i = inst_idx++;
        instances[i].traversableHandle = gas_point_medium.handle;
        instances[i].flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
        instances[i].instanceId = 0; // not used
        instances[i].sbtOffset = static_cast<uint32_t>(scene_loader->record_count * RAY_TYPE_COUNT);
        instances[i].visibilityMask = GEOM_MASK_POINT_MEDIUM;
        memcpy(instances[i].transform, instance.transform, sizeof(float) * 12);
    }

    optix::accel_struct ias_points;
    if (inst_idx > 0) {
        CUdeviceptr dInstances = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dInstances), sizeof(OptixInstance) * inst_idx));
        ias_points.input = {};
        ias_points.input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        ias_points.input.instanceArray.instances = dInstances;
        ias_points.input.instanceArray.numInstances = inst_idx;
        ias_points.accel_options = {};
        ias_points.accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        ias_points.accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(dInstances), &instances, sizeof(OptixInstance) * inst_idx, cudaMemcpyHostToDevice));
        optix::complete_creation(
            optix::DeviceState::get_instance(), &ias_points, &d_tmp_buffer, &tmp_buffer_size, false, false);
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dInstances)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tmp_buffer)));
    }

    // stop build time, start camera trace time
    float time_build = timer.lap();

    // report memory at the point most memory is allocated
    double mem_used, mem_free, mem_total;
    optix::report_memory(mem_used, mem_free, mem_total);

    /** camera tracing */
    // Unless rendering with traditional light tracing
    if (params.trace_camera_paths) {
        params.handle_points = ias_points.handle;
        params.handle_beams = ias_beams.handle;
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_params),
                                   &params,
                                   sizeof(LaunchParams),
                                   cudaMemcpyHostToDevice,
                                   optix::DeviceState::get_instance()->stream));
#ifdef DBG_CORE_UTIL
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(dTraceCount), 0, mLightSubPathCount * sizeof(uint64_t)));
        // measure precise time
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
#endif
        OPTIX_CHECK(optixLaunch(pipeline_trace_camera,
                                optix::DeviceState::get_instance()->stream,
                                reinterpret_cast<CUdeviceptr>(d_params),
                                sizeof(LaunchParams),
                                &sbt_trace_camera,
                                params.screen_pixel_count,
                                1,
                                1));
        CUDA_SYNC_CHECK(); // make sure launch is done before copying back
#ifdef DBG_CORE_UTIL
        // record the milliseconds
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        camera_ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&camera_ms, start, stop));
        // get the count
        auto* trace_count = static_cast<uint64_t*>(malloc(sizeof(uint64_t) * mLightSubPathCount));
        CUDA_CHECK(cudaMemcpy(trace_count,
                              reinterpret_cast<void*>(dTraceCount),
                              sizeof(uint64_t) * mLightSubPathCount,
                              cudaMemcpyDeviceToHost));
        camera_trace_count = 0;
        for (uint32_t i = 0; i < mLightSubPathCount; i++) {
            camera_trace_count += trace_count[i];
        }
        free(trace_count);
#endif
    }

    /** cleanup data structures */
    if (((build_pb2d || build_pp3d) && light_vertices_found_medium > 0) ||
        (build_surf && light_vertices_found_surface > 0)) {
        optix::delete_accel_struct(&ias_points);
    }

    if ((build_bb1d || build_bp2d) && light_beams_found > 0) {
        optix::delete_accel_struct(&ias_beams);
    }

    if ((build_pb2d || build_pp3d) && light_vertices_found_medium > 0) {
        optix::delete_accel_struct(&gas_point_medium);
    }

    if (build_surf && light_vertices_found_surface > 0) {
        optix::delete_accel_struct(&gas_point_surface);
    }

    CUDA_CHECK(cudaMemcpy(colors,
                          reinterpret_cast<void*>(params.framebuffer.color),
                          screen_pixel_count * sizeof(float3),
                          cudaMemcpyDeviceToHost));

    // add this iteration's contribution to the framebuffer
    {
        struct Framebuffer tmp;
        tmp.resolution = framebuffer->resolution;
        tmp.color = colors;
        add_framebuffer(*framebuffer, tmp);
    }

    float time_camera_trace = timer.lap();

    float light_util = 0.f, camera_util = 0.f;
#ifdef DBG_CORE_UTIL
    // in Giga rays/sec:
    // / 1e3f for milliseconds to seconds, / 1e9f for giga
    light_util = static_cast<float>(light_trace_count) / 1e9f / (light_ms / 1e3f);
    camera_util = static_cast<float>(camera_trace_count) / 1e9f / (camera_ms / 1e3f);
#endif

    if (config->do_log) {
        Logger::get_instance().log_times(config->scene_obj_file,
                                         iteration,
                                         time_light_trace,
                                         time_build,
                                         time_camera_trace,
                                         static_cast<uint32_t>(scene_loader->scene.camera.resolution.x),
                                         static_cast<uint32_t>(scene_loader->scene.camera.resolution.y),
                                         mem_used,
                                         mem_free,
                                         mem_total,
                                         light_util,
                                         camera_util);
    }
}

void Renderer::cleanup()
{
#ifdef DBG_CORE_UTIL
    // no need to free dSumOut, it is included in dTraceCount
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(dTraceCount)));
#endif

    /** light pass data */
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(params.light_beams)));
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(params.light_vertices)));

    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(params.light_beam_counter)));
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(params.light_vertex_counter)));
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(params.light_vertex_medium_counter)));
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(params.light_vertex_surface_counter)));

    free(colors);
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(params.framebuffer.color)));

    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(params.instances_beam)));
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(params.aabb_medium)));
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(params.aabb_surface)));

    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(params.indices_medium)));
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(params.indices_surface)));

    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(params.light_subpath_starts)));

    /** scene data */
    delete_scene_device(params.scene);

    // pipeline general
    OPTIX_CHECK_NOTHROW(optixProgramGroupDestroy(hit_group_real));
    OPTIX_CHECK_NOTHROW(optixProgramGroupDestroy(hit_group_imag));
    OPTIX_CHECK_NOTHROW(optixProgramGroupDestroy(miss_group_real));
    OPTIX_CHECK_NOTHROW(optixProgramGroupDestroy(miss_group_imag));
    OPTIX_CHECK_NOTHROW(optixModuleDestroy(module_intersect));

    // estimators
    OPTIX_CHECK_NOTHROW(optixProgramGroupDestroy(hit_group_bp2d));
    OPTIX_CHECK_NOTHROW(optixProgramGroupDestroy(miss_group_bp2d));
    OPTIX_CHECK_NOTHROW(optixModuleDestroy(module_bp2d));

    OPTIX_CHECK_NOTHROW(optixProgramGroupDestroy(hit_group_pp3d));
    OPTIX_CHECK_NOTHROW(optixProgramGroupDestroy(miss_group_pp3d));
    OPTIX_CHECK_NOTHROW(optixModuleDestroy(module_pp3d));

    OPTIX_CHECK_NOTHROW(optixProgramGroupDestroy(hit_group_surf));
    OPTIX_CHECK_NOTHROW(optixProgramGroupDestroy(miss_group_surf));
    OPTIX_CHECK_NOTHROW(optixModuleDestroy(module_surf));

    OPTIX_CHECK_NOTHROW(optixProgramGroupDestroy(hit_group_pb2d));
    OPTIX_CHECK_NOTHROW(optixProgramGroupDestroy(miss_group_pb2d));
    OPTIX_CHECK_NOTHROW(optixModuleDestroy(module_pb2d));

    OPTIX_CHECK_NOTHROW(optixProgramGroupDestroy(hit_group_bb1d));
    OPTIX_CHECK_NOTHROW(optixProgramGroupDestroy(miss_group_bb1d));
    OPTIX_CHECK_NOTHROW(optixModuleDestroy(module_bb1d));

    // pipeline camera
    OPTIX_CHECK_NOTHROW(optixPipelineDestroy(pipeline_trace_camera));
    OPTIX_CHECK_NOTHROW(optixProgramGroupDestroy(raygen_group_trace_camera));
    OPTIX_CHECK_NOTHROW(optixModuleDestroy(module_trace_light));

    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(sbt_trace_light.hitgroupRecordBase)));
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(sbt_trace_light.missRecordBase)));
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(sbt_trace_light.raygenRecord)));

    // pipeline light
    OPTIX_CHECK_NOTHROW(optixPipelineDestroy(pipeline_trace_light));
    OPTIX_CHECK_NOTHROW(optixProgramGroupDestroy(raygen_group_trace_light));
    OPTIX_CHECK_NOTHROW(optixModuleDestroy(module_trace_camera));

    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(sbt_trace_camera.callablesRecordBase)));
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(sbt_trace_camera.missRecordBase)));
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(sbt_trace_camera.hitgroupRecordBase)));
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(sbt_trace_camera.raygenRecord)));

    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(d_params)));

    /** gases */
    // safe to delete regardless of whether they were initialized
    optix::delete_accel_struct(&scene_ias);
    optix::delete_accel_struct(&gas_real_spheres);
    optix::delete_accel_struct(&gas_real_triangles);
    optix::delete_accel_struct(&gas_imaginary_spheres);
    optix::delete_accel_struct(&gas_imaginary_triangles);
    optix::delete_accel_struct(&gas_cylinder);
}
