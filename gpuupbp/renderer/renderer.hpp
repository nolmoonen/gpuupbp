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

#ifndef RENDERER_RENDERER_HPP
#define RENDERER_RENDERER_HPP

#include "../host/framebuffer.hpp"
#include "../misc/optix_helper.hpp"
#include "../misc/scene_loader.hpp"
#include "../misc/timer.hpp"
#include "../shared/intersect_defs.h"
#include "../shared/launch_params.h"

#include <cmath>
#include <vector>

typedef optix::RecordData<RayGenData> raygen_record;
typedef optix::RecordData<MissData> miss_record;
typedef optix::RecordData<HitgroupData> hitgroup_record;
typedef optix::RecordData<ExceptionData> exception_record;

/// For all computations, we assume the balance heuristic and n_bpt == 1.
struct Renderer {
    /// Returns -1 on failure, 0 on success.
    int32_t create_pipeline(bool gpu_assert);

    /// Returns -1 on failure, 0 on success.
    int32_t init(const SceneLoader* sl, const Config* cfg, Framebuffer* fb);

    void run_iteration(int iteration);

    void cleanup();

    Framebuffer* framebuffer;
    const SceneLoader* scene_loader;
    const Config* config;

    optix::accel_struct gas_cylinder;

#ifdef DBG_CORE_UTIL
    /// For every thread (light subpath), an uint64_t counter.
    CUdeviceptr dTraceCount = 0;
#endif

    /// Host array of colors for copying from device to host.
    float3* colors = nullptr;

    /** Whether to build the photon maps. */
    bool build_bb1d;
    bool build_bp2d;
    bool build_pb2d;
    bool build_pp3d;
    bool build_surf;

    /** Initial merging radii */
    float bb1d_radius_initial;
    float bp2d_radius_initial;
    float pb2d_radius_initial;
    float pp3d_radius_initial;
    float surf_radius_initial;

    /// First bb1d_used_light_subpath_count out of light_subpath_count
    /// light paths will generate photon beams
    float bb1d_used_light_subpath_count;

    /// Number of pixels
    uint32_t screen_pixel_count;
    /// Number of light sub-paths
    uint32_t light_subpath_count;

    Timer timer;

    optix::accel_struct gas_real_triangles;
    optix::accel_struct gas_real_spheres;
    optix::accel_struct gas_imaginary_triangles;
    optix::accel_struct gas_imaginary_spheres;
    /// Contains all and only all geometry, used in light tracing.
    optix::accel_struct scene_ias;

    /// Contains ray generation program for light tracing.
    OptixModule module_trace_light;
    /// Contains ray generation program for camera tracing.
    OptixModule module_trace_camera;
    /// Contains hit and miss programs.
    OptixModule module_intersect;
    /// Contain intersection programs for all estimators.
    OptixModule module_bb1d;
    OptixModule module_bp2d;
    OptixModule module_pb2d;
    OptixModule module_pp3d;
    OptixModule module_surf;
    /// Contains the exception program.
    OptixModule module_exception;
    OptixProgramGroup raygen_group_trace_light;
    OptixProgramGroup raygen_group_trace_camera;
    /// The intersect function for real geometry.
    OptixProgramGroup miss_group_real;
    OptixProgramGroup hit_group_real;
    /// The intersect function for imaginary geometry.
    OptixProgramGroup miss_group_imag;
    OptixProgramGroup hit_group_imag;
    /// BB1D estimator.
    OptixProgramGroup miss_group_bb1d;
    OptixProgramGroup hit_group_bb1d;
    /// BP2D estimator.
    OptixProgramGroup miss_group_bp2d;
    OptixProgramGroup hit_group_bp2d;
    /// PB2D estimator.
    OptixProgramGroup miss_group_pb2d;
    OptixProgramGroup hit_group_pb2d;
    /// PP3D estimator.
    OptixProgramGroup miss_group_pp3d;
    OptixProgramGroup hit_group_pp3d;
    /// SURF estimator.
    OptixProgramGroup miss_group_surf;
    OptixProgramGroup hit_group_surf;
    /// Exception.
    OptixProgramGroup exception_group;
    OptixPipeline pipeline_trace_light;
    OptixPipeline pipeline_trace_camera;
    LaunchParams params;
    /// Launch params on device.
    LaunchParams* d_params;
    OptixShaderBindingTable sbt_trace_light;
    OptixShaderBindingTable sbt_trace_camera;
};

#endif // RENDERER_RENDERER_HPP
