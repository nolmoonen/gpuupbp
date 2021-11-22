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

#include "../host/exception.hpp"
#include "renderer.hpp"

/// Create a triangle geometry acceleration structure.
void create_triangle_gas(
    // in
    uint32_t count,
    uint32_t record_count,
    float4* h_vertices,
    float3* h_normals,
    uint32_t* h_records,
    CUdeviceptr* d_tmp_buffer,
    size_t* tmp_buffer_size,
    // out
    optix::accel_struct* map,
    float4** d_vertices,
    float3** d_normals)
{
    // device array of record indices
    CUdeviceptr d_mat_indices = 0;

    const size_t vert_size = 3 * sizeof(float4) * count;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(d_vertices), vert_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(*d_vertices), h_vertices, vert_size, cudaMemcpyHostToDevice));

    const size_t norm_size = 4 * sizeof(float3) * count;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(d_normals), norm_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(*d_normals), h_normals, norm_size, cudaMemcpyHostToDevice));

    const size_t rcrd_size = sizeof(uint32_t) * count;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat_indices), rcrd_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_mat_indices), h_records, rcrd_size, cudaMemcpyHostToDevice));

    auto* triang_input_flags = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * record_count));
    std::fill(triang_input_flags, triang_input_flags + record_count, OPTIX_GEOMETRY_FLAG_NONE);

    map->input = {};
    map->input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    map->input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    map->input.triangleArray.vertexStrideInBytes = sizeof(float4);
    map->input.triangleArray.numVertices = count * 3;
    map->input.triangleArray.vertexBuffers = reinterpret_cast<CUdeviceptr*>(d_vertices);
    map->input.triangleArray.flags = triang_input_flags;
    map->input.triangleArray.numSbtRecords = record_count;
    map->input.triangleArray.sbtIndexOffsetBuffer = d_mat_indices;
    map->input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    map->input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

    map->accel_options = {};
    map->accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    map->accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    optix::complete_creation(optix::DeviceState::get_instance(), map, d_tmp_buffer, tmp_buffer_size, false, true);

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_mat_indices)));
    free(triang_input_flags);
}

/// Create a sphere geometry acceleration structure.
void create_sphere_gas(
    // in
    uint32_t count,
    uint32_t record_count,
    OptixAabb* h_aabbs,
    uint32_t* h_records,
    CUdeviceptr* d_tmp_buffer,
    size_t* tmp_buffer_size,
    // out
    optix::accel_struct* map,
    OptixAabb** d_aabbs)
{
    // device array of record indices
    CUdeviceptr d_mat_indices = 0;

    const size_t aabb_size = sizeof(OptixAabb) * count;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(d_aabbs), aabb_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(*d_aabbs), h_aabbs, aabb_size, cudaMemcpyHostToDevice));

    const size_t rcrd_size = sizeof(uint32_t) * count;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat_indices), rcrd_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_mat_indices), h_records, rcrd_size, cudaMemcpyHostToDevice));

    auto* sphere_input_flags = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * record_count));
    std::fill(sphere_input_flags, sphere_input_flags + record_count, OPTIX_GEOMETRY_FLAG_NONE);

    map->input = {};
    map->input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    map->input.customPrimitiveArray.aabbBuffers = reinterpret_cast<CUdeviceptr*>(d_aabbs);
    map->input.customPrimitiveArray.flags = sphere_input_flags;
    map->input.customPrimitiveArray.numSbtRecords = record_count;
    map->input.customPrimitiveArray.numPrimitives = count;
    map->input.customPrimitiveArray.sbtIndexOffsetBuffer = d_mat_indices;
    map->input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    map->input.customPrimitiveArray.primitiveIndexOffset = 0;

    map->accel_options = {};
    map->accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    map->accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    optix::complete_creation(optix::DeviceState::get_instance(), map, d_tmp_buffer, tmp_buffer_size, false, true);

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_mat_indices)));
    free(sphere_input_flags);
}

/// Creates instance acceleration structure with (at most) four
/// geometry acceleration structures.
void create_gases(const SceneLoader& scene,
                  LaunchParams& params,
                  optix::accel_struct& triang_gas_real,
                  optix::accel_struct& sphere_gas_real,
                  optix::accel_struct& triang_gas_imag,
                  optix::accel_struct& sphere_gas_imag,
                  optix::accel_struct& instan_gas)
{
    // use the identity matrix for the instance transform
    optix::Instance instance = {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}};

    // four instances, initialized to zero
    OptixInstance optix_instances[4];
    memset(optix_instances, 0, sizeof(OptixInstance) * 4);

    // index of next instance to instantiate
    uint32_t inst_idx = 0;

    // stratch buffer
    CUdeviceptr d_tmp_buffer = 0;
    size_t tmp_buffer_size = 0;

    if (scene.scene.triangles_real.count > 0) {
        triang_gas_real = {};
        create_triangle_gas(scene.scene.triangles_real.count,
                            scene.record_count,
                            scene.scene.triangles_real.verts,
                            scene.scene.triangles_real.norms,
                            scene.scene.triangles_real.records,
                            &d_tmp_buffer,
                            &tmp_buffer_size,
                            &triang_gas_real,
                            &params.scene.triangles_real.verts,
                            &params.scene.triangles_real.norms);

        uint32_t i = inst_idx++;
        optix_instances[i].traversableHandle = triang_gas_real.handle;
        optix_instances[i].flags = OPTIX_INSTANCE_FLAG_NONE;
        optix_instances[i].instanceId = i; // not used
        optix_instances[i].sbtOffset = 0;
        optix_instances[i].visibilityMask = GEOM_MASK_REAL;
        memcpy(optix_instances[i].transform, instance.transform, sizeof(float) * 12);
    } else {
        // to have a valid pointer on which we can call free later
        params.scene.triangles_real.verts = nullptr;
        params.scene.triangles_real.norms = nullptr;
        params.scene.triangles_real.records = nullptr;
    }

    if (scene.scene.spheres_real.count > 0) {
        sphere_gas_real = {};
        create_sphere_gas(scene.scene.spheres_real.count,
                          scene.record_count,
                          scene.scene.spheres_real.aabbs,
                          scene.scene.spheres_real.records,
                          &d_tmp_buffer,
                          &tmp_buffer_size,
                          &sphere_gas_real,
                          &params.scene.spheres_real.aabbs);

        uint32_t i = inst_idx++;
        optix_instances[i].traversableHandle = sphere_gas_real.handle;
        optix_instances[i].flags = OPTIX_INSTANCE_FLAG_NONE;
        optix_instances[i].instanceId = i; // not used
        optix_instances[i].sbtOffset = 0;
        optix_instances[i].visibilityMask = GEOM_MASK_REAL;
        memcpy(optix_instances[i].transform, instance.transform, sizeof(float) * 12);
    } else {
        // to have a valid pointer on which we can call free later
        params.scene.spheres_real.aabbs = nullptr;
        params.scene.spheres_real.records = nullptr;
    }

    if (scene.scene.triangles_imag.count > 0) {
        triang_gas_imag = {};
        create_triangle_gas(scene.scene.triangles_imag.count,
                            scene.record_count,
                            scene.scene.triangles_imag.verts,
                            scene.scene.triangles_imag.norms,
                            scene.scene.triangles_imag.records,
                            &d_tmp_buffer,
                            &tmp_buffer_size,
                            &triang_gas_imag,
                            &params.scene.triangles_imag.verts,
                            &params.scene.triangles_imag.norms);

        uint32_t i = inst_idx++;
        optix_instances[i].traversableHandle = triang_gas_imag.handle;
        optix_instances[i].flags = OPTIX_INSTANCE_FLAG_NONE;
        optix_instances[i].instanceId = i; // not used
        optix_instances[i].sbtOffset = 0;
        optix_instances[i].visibilityMask = GEOM_MASK_IMAG;
        memcpy(optix_instances[i].transform, instance.transform, sizeof(float) * 12);
    } else {
        // to have a valid pointer on which we can call free later
        params.scene.triangles_imag.verts = nullptr;
        params.scene.triangles_imag.norms = nullptr;
        params.scene.triangles_imag.records = nullptr;
    }

    if (scene.scene.spheres_imag.count > 0) {
        sphere_gas_imag = {};
        create_sphere_gas(scene.scene.spheres_imag.count,
                          scene.record_count,
                          scene.scene.spheres_imag.aabbs,
                          scene.scene.spheres_imag.records,
                          &d_tmp_buffer,
                          &tmp_buffer_size,
                          &sphere_gas_imag,
                          &params.scene.spheres_imag.aabbs);

        uint32_t i = inst_idx++;
        optix_instances[i].traversableHandle = sphere_gas_imag.handle;
        optix_instances[i].flags = OPTIX_INSTANCE_FLAG_NONE;
        optix_instances[i].instanceId = i; // not used
        optix_instances[i].sbtOffset = 0;
        optix_instances[i].visibilityMask = GEOM_MASK_IMAG;
        memcpy(optix_instances[i].transform, instance.transform, sizeof(float) * 12);
    } else {
        // to have a valid pointer on which we can call free later
        params.scene.spheres_imag.aabbs = nullptr;
        params.scene.spheres_imag.records = nullptr;
    }

    // if at least one category of geometry is found, create instance AS
    // if no geometry exists in the scene, the handle remains null
    // this results in correct behavior later, calling optixTrace with a
    // null handle will only invoke the miss program
    if (inst_idx > 0) {
        instan_gas = {};

        CUdeviceptr d_instances = 0;
        size_t instance_size_in_bytes = sizeof(OptixInstance) * inst_idx;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances), instance_size_in_bytes));

        instan_gas.input = {};
        instan_gas.input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        instan_gas.input.instanceArray.instances = d_instances;
        instan_gas.input.instanceArray.numInstances = inst_idx;

        instan_gas.accel_options = {};
        instan_gas.accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
        instan_gas.accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_instances), &optix_instances, instance_size_in_bytes, cudaMemcpyHostToDevice));

        optix::complete_creation(
            optix::DeviceState::get_instance(), &instan_gas, &d_tmp_buffer, &tmp_buffer_size, false, false);
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_instances)));
    }

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tmp_buffer)));
    // do not free device vertices, normals, and aabbs here,
    // as they are used in the shader programs
}

/// Initializes the arrays of the scene on device.
void init_scene(const SceneLoader& sl, LaunchParams& params)
{
    /** material data */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.scene.materials), sl.scene.mat_count * sizeof(Material)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(params.scene.materials),
                          sl.scene.materials,
                          sl.scene.mat_count * sizeof(Material),
                          cudaMemcpyHostToDevice));
    /** media data */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.scene.media), sl.scene.med_count * sizeof(Medium)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(params.scene.media),
                          sl.scene.media,
                          sl.scene.med_count * sizeof(Medium),
                          cudaMemcpyHostToDevice));
    /** light data */
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&params.scene.lights), sl.scene.light_count * sizeof(AbstractLight)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(params.scene.lights),
                          sl.scene.lights,
                          sl.scene.light_count * sizeof(AbstractLight),
                          cudaMemcpyHostToDevice));
    /** environment map */
    // if the scene has a background light and an environment map,
    // and the background light uses the environment map, copy the env map
    if (sl.scene.background_light_idx != UINT32_MAX &&
        sl.scene.lights[sl.scene.background_light_idx].background.uses_env_map && sl.scene.has_env_map) {
        // allocate memory and copy the distributions
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.scene.env_map.distribution.conditional_v),
                              sl.scene.env_map.img.height * sizeof(Distribution1D)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(params.scene.env_map.distribution.conditional_v),
                              sl.scene.env_map.distribution.conditional_v,
                              sl.scene.env_map.img.height * sizeof(Distribution1D),
                              cudaMemcpyHostToDevice));
        // allocate memory and copy the data
        size_t dist_size = sizeof(float) * (sl.scene.env_map.img.height + 1) * (2 * sl.scene.env_map.img.width + 1);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.scene.env_map.distribution.data), dist_size));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(params.scene.env_map.distribution.data),
                              sl.scene.env_map.distribution.data,
                              dist_size,
                              cudaMemcpyHostToDevice));
        // allocate memory and copy the image
        size_t image_size = sizeof(float3) * sl.scene.env_map.img.width * sl.scene.env_map.img.height;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.scene.env_map.img.data), image_size));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(params.scene.env_map.img.data),
                              sl.scene.env_map.img.data,
                              image_size,
                              cudaMemcpyHostToDevice));
    } else {
        // set pointers to zero so we can call free later
        params.scene.env_map.img.data = nullptr;
        params.scene.env_map.distribution.data = nullptr;
        params.scene.env_map.distribution.conditional_v = nullptr;
    }
}

int32_t Renderer::create_pipeline(bool gpu_assert)
{
    /** create pipeline */
    OptixPipelineCompileOptions pipeline_compile_options = {};
    optix::GasType type = optix::SINGLE_LEVEL_INSTANCING;
    // primitive type always includes custom primitives due to photon maps
    optix::PrimType prim_type = optix::ANY_;
    // sphere.cu uses 3 attributes
    optix::init_pipeline_compile_options(&pipeline_compile_options, 2, 3, type, prim_type, gpu_assert);

    OptixModuleCompileOptions module_compile_options = {};
    optix::init_module_compile_options(&module_compile_options);
    if (create_module(optix::DeviceState::get_instance(),
                      &pipeline_compile_options,
                      &module_compile_options,
                      "kernel/trace_light.cu",
                      &module_trace_light)) {
        return -1;
    }
    if (create_module(optix::DeviceState::get_instance(),
                      &pipeline_compile_options,
                      &module_compile_options,
                      "kernel/trace_camera.cu",
                      &module_trace_camera)) {
        return -1;
    }
    if (create_module(optix::DeviceState::get_instance(),
                      &pipeline_compile_options,
                      &module_compile_options,
                      "kernel/intersector.cu",
                      &module_intersect)) {
        return -1;
    }
    if (create_module(optix::DeviceState::get_instance(),
                      &pipeline_compile_options,
                      &module_compile_options,
                      "kernel/pde/program_bb1d.cu",
                      &module_bb1d)) {
        return -1;
    }
    if (create_module(optix::DeviceState::get_instance(),
                      &pipeline_compile_options,
                      &module_compile_options,
                      "kernel/pde/program_pb2d.cu",
                      &module_pb2d)) {
        return -1;
    }
    if (create_module(optix::DeviceState::get_instance(),
                      &pipeline_compile_options,
                      &module_compile_options,
                      "kernel/pde/program_surf.cu",
                      &module_surf)) {
        return -1;
    }
    if (create_module(optix::DeviceState::get_instance(),
                      &pipeline_compile_options,
                      &module_compile_options,
                      "kernel/pde/program_pp3d.cu",
                      &module_pp3d)) {
        return -1;
    }
    if (create_module(optix::DeviceState::get_instance(),
                      &pipeline_compile_options,
                      &module_compile_options,
                      "kernel/pde/program_bp2d.cu",
                      &module_bp2d)) {
        return -1;
    }
    if (create_module(optix::DeviceState::get_instance(),
                      &pipeline_compile_options,
                      &module_compile_options,
                      "kernel/exception.cu",
                      &module_exception)) {
        return -1;
    }

    // program groups
    OptixProgramGroupOptions program_group_options = {};
    optix::init_program_group_options(&program_group_options);

    optix::create_raygen_program_group(optix::DeviceState::get_instance(),
                                       &program_group_options,
                                       &raygen_group_trace_light,
                                       "__raygen__trace_light",
                                       module_trace_light);
    optix::create_raygen_program_group(optix::DeviceState::get_instance(),
                                       &program_group_options,
                                       &raygen_group_trace_camera,
                                       "__raygen__trace_camera",
                                       module_trace_camera);

    optix::create_miss_program_group(optix::DeviceState::get_instance(),
                                     &program_group_options,
                                     &miss_group_real,
                                     nullptr,
                                     nullptr); // no miss program
    optix::create_hitgroup_program_group(optix::DeviceState::get_instance(),
                                         &program_group_options,
                                         &hit_group_real,
                                         nullptr,
                                         nullptr,
                                         "__closesthit__closest",
                                         module_intersect,
                                         "__intersection__sphere_real",
                                         module_intersect);

    optix::create_miss_program_group(optix::DeviceState::get_instance(),
                                     &program_group_options,
                                     &miss_group_imag,
                                     nullptr,
                                     nullptr); // no miss program
    optix::create_hitgroup_program_group(optix::DeviceState::get_instance(),
                                         &program_group_options,
                                         &hit_group_imag,
                                         "__anyhit__all",
                                         module_intersect,
                                         nullptr,
                                         nullptr,
                                         "__intersection__sphere_imag",
                                         module_intersect);

    optix::create_miss_program_group(optix::DeviceState::get_instance(),
                                     &program_group_options,
                                     &miss_group_bb1d,
                                     nullptr,
                                     nullptr); // no miss program for estimators
    optix::create_hitgroup_program_group(optix::DeviceState::get_instance(),
                                         &program_group_options,
                                         &hit_group_bb1d,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         "__intersection__bb1d",
                                         module_bb1d);

    optix::create_miss_program_group(optix::DeviceState::get_instance(),
                                     &program_group_options,
                                     &miss_group_pb2d,
                                     nullptr,
                                     nullptr); // no miss program for estimators
    optix::create_hitgroup_program_group(optix::DeviceState::get_instance(),
                                         &program_group_options,
                                         &hit_group_pb2d,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         "__intersection__pb2d",
                                         module_pb2d);

    optix::create_miss_program_group(optix::DeviceState::get_instance(),
                                     &program_group_options,
                                     &miss_group_surf,
                                     nullptr,
                                     nullptr); // no miss program for estimators
    optix::create_hitgroup_program_group(optix::DeviceState::get_instance(),
                                         &program_group_options,
                                         &hit_group_surf,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         "__intersection__surf",
                                         module_surf);

    optix::create_miss_program_group(optix::DeviceState::get_instance(),
                                     &program_group_options,
                                     &miss_group_pp3d,
                                     nullptr,
                                     nullptr); // no miss program for estimators
    optix::create_hitgroup_program_group(optix::DeviceState::get_instance(),
                                         &program_group_options,
                                         &hit_group_pp3d,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         "__intersection__pp3d",
                                         module_pp3d);

    optix::create_miss_program_group(optix::DeviceState::get_instance(),
                                     &program_group_options,
                                     &miss_group_bp2d,
                                     nullptr,
                                     nullptr); // no miss program for estimators
    optix::create_hitgroup_program_group(optix::DeviceState::get_instance(),
                                         &program_group_options,
                                         &hit_group_bp2d,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         "__intersection__bp2d",
                                         module_bp2d);

    optix::create_exception_program_group(optix::DeviceState::get_instance(),
                                          &program_group_options,
                                          &exception_group,
                                          "__exception__upbp",
                                          module_exception);

    // pipeline trace light (no estimators)
    OptixProgramGroup program_groups_trace_light[] = {
        raygen_group_trace_light, miss_group_real, miss_group_imag, hit_group_real, hit_group_imag, exception_group};
    optix::create_pipeline(optix::DeviceState::get_instance(),
                           &pipeline_trace_light,
                           &pipeline_compile_options,
                           program_groups_trace_light,
                           sizeof(program_groups_trace_light) / sizeof(program_groups_trace_light[0]),
                           1,
                           2);

    // pipeline trace camera (includes estimators)
    OptixProgramGroup program_groups_trace_camera[] = {raygen_group_trace_camera,
                                                       miss_group_real,
                                                       hit_group_real,
                                                       miss_group_imag,
                                                       hit_group_imag,
                                                       miss_group_bb1d,
                                                       hit_group_bb1d,
                                                       miss_group_pb2d,
                                                       hit_group_pb2d,
                                                       miss_group_surf,
                                                       hit_group_surf,
                                                       miss_group_pp3d,
                                                       hit_group_pp3d,
                                                       miss_group_bp2d,
                                                       hit_group_bp2d,
                                                       exception_group};
    optix::create_pipeline(optix::DeviceState::get_instance(),
                           &pipeline_trace_camera,
                           &pipeline_compile_options,
                           program_groups_trace_camera,
                           sizeof(program_groups_trace_camera) / sizeof(program_groups_trace_camera[0]),
                           1,
                           2);

    // sbt

    // raygen
    const size_t raygen_record_size = sizeof(raygen_record);
    raygen_record rg_sbt = {};
    sbt_trace_light = {};
    optix::create_sbt_raygen_record(&sbt_trace_light, &rg_sbt, raygen_record_size, raygen_group_trace_light);
    sbt_trace_camera = {};
    optix::create_sbt_raygen_record(&sbt_trace_camera, &rg_sbt, raygen_record_size, raygen_group_trace_camera);

    // miss
    const size_t miss_record_size = sizeof(miss_record);
    auto* ms_sbt = new miss_record[RAY_TYPE_COUNT + RAY_TYPE2_COUNT];
    OptixProgramGroup ms_groups[] = {miss_group_real,
                                     miss_group_imag,
                                     miss_group_bb1d,
                                     miss_group_pb2d,
                                     miss_group_surf,
                                     miss_group_pp3d,
                                     miss_group_bp2d};
    optix::create_sbt_miss_records(&sbt_trace_light,
                                   ms_sbt,
                                   miss_record_size,
                                   ms_groups,
                                   sizeof(ms_groups) / sizeof(ms_groups[0]),
                                   RAY_TYPE_COUNT + RAY_TYPE2_COUNT);
    optix::create_sbt_miss_records(&sbt_trace_camera,
                                   ms_sbt,
                                   miss_record_size,
                                   ms_groups,
                                   sizeof(ms_groups) / sizeof(ms_groups[0]),
                                   RAY_TYPE_COUNT + RAY_TYPE2_COUNT);
    delete[] ms_sbt;

    // hit
    auto entries = static_cast<uint32_t>(scene_loader->record_count * RAY_TYPE_COUNT + RAY_TYPE2_COUNT);
    auto* hg_sbt = static_cast<hitgroup_record*>(malloc(sizeof(hitgroup_record) * entries));
    // triangle and sphere, real and imaginary (all use index 0)
    for (uint32_t i = 0; i < scene_loader->record_count; i++) {
        {
            const uint32_t sbt_index = i * RAY_TYPE_COUNT + 0;
            OPTIX_CHECK(optixSbtRecordPackHeader(hit_group_real, &hg_sbt[sbt_index]));
            hg_sbt[sbt_index].data.gpu_mat = scene_loader->records[i];
        }
        {
            const uint32_t sbt_index = i * RAY_TYPE_COUNT + 1;
            OPTIX_CHECK(optixSbtRecordPackHeader(hit_group_imag, &hg_sbt[sbt_index]));
            hg_sbt[sbt_index].data.gpu_mat = scene_loader->records[i];
        }
    }
    OptixProgramGroup type2_groups[] = {hit_group_bb1d, hit_group_pb2d, hit_group_surf, hit_group_pp3d, hit_group_bp2d};
    for (uint32_t i = 0; i < RAY_TYPE2_COUNT; i++) {
        const auto sbt_index = static_cast<uint32_t>((scene_loader->record_count * RAY_TYPE_COUNT) + i);
        OPTIX_CHECK(optixSbtRecordPackHeader(type2_groups[i], &hg_sbt[sbt_index]));
    }

    const size_t hitgroup_record_size = sizeof(hitgroup_record);

    // light
    CUdeviceptr d_hitgroup_records_light = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_records_light), hitgroup_record_size * entries));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_records_light),
                          hg_sbt,
                          hitgroup_record_size * entries,
                          cudaMemcpyHostToDevice));
    sbt_trace_light.hitgroupRecordBase = d_hitgroup_records_light;
    sbt_trace_light.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
    sbt_trace_light.hitgroupRecordCount = entries;

    // camera
    CUdeviceptr d_hitgroup_records_camera = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_records_camera), hitgroup_record_size * entries));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_records_camera),
                          hg_sbt,
                          hitgroup_record_size * entries,
                          cudaMemcpyHostToDevice));
    sbt_trace_camera.hitgroupRecordBase = d_hitgroup_records_camera;
    sbt_trace_camera.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
    sbt_trace_camera.hitgroupRecordCount = entries;

    free(hg_sbt);

    const size_t exception_record_size = sizeof(raygen_record);
    exception_record exc_sbt = {};
    optix::create_sbt_exception_record(&sbt_trace_light, &exc_sbt, exception_record_size, exception_group);
    optix::create_sbt_exception_record(&sbt_trace_camera, &exc_sbt, exception_record_size, exception_group);

    return 0;
}

int32_t Renderer::init(const struct SceneLoader* sl, const struct Config* cfg, struct Framebuffer* fb)
{
    scene_loader = sl;
    config = cfg;
    framebuffer = fb;

    // initialize initial radii using scene radius if relative
    const float scene_radius = scene_loader->scene.scene_sphere.radius;
    bb1d_radius_initial = cfg->bb1d_radius_initial;
    if (bb1d_radius_initial < 0.f) {
        bb1d_radius_initial = -bb1d_radius_initial * scene_radius;
    }
    assert(bb1d_radius_initial > 0.f);

    bp2d_radius_initial = cfg->bp2d_radius_initial;
    if (bp2d_radius_initial < 0.f) {
        bp2d_radius_initial = -bp2d_radius_initial * scene_radius;
    }
    assert(bp2d_radius_initial > 0.f);

    pb2d_radius_initial = cfg->pb2d_radius_initial;
    if (pb2d_radius_initial < 0.f) {
        pb2d_radius_initial = -pb2d_radius_initial * scene_radius;
    }
    assert(pb2d_radius_initial > 0.f);

    pp3d_radius_initial = cfg->pp3d_radius_initial;
    if (pp3d_radius_initial < 0.f) {
        pp3d_radius_initial = -pp3d_radius_initial * scene_radius;
    }
    assert(pp3d_radius_initial > 0.f);

    surf_radius_initial = cfg->surf_radius_initial;
    if (surf_radius_initial < 0.f) {
        surf_radius_initial = -surf_radius_initial * scene_radius;
    }
    assert(surf_radius_initial > 0.f);

    // we trace light and camera if at least one technique is enabled
    params.trace_light_paths = cfg->algorithm_flags != 0;
    params.trace_camera_paths = cfg->algorithm_flags != 0;
    params.do_bpt = cfg->algorithm_flags & BPT;
    params.do_bb1d = cfg->algorithm_flags & BB1D;
    params.do_bp2d = cfg->algorithm_flags & BP2D;
    params.do_pb2d = cfg->algorithm_flags & PB2D;
    params.do_pp3d = cfg->algorithm_flags & PP3D;
    params.do_surf = cfg->algorithm_flags & SURF;

    if (cfg->algorithm_flags & SPECULAR_ONLY) {
        // only trace fully specular camera paths
        params.trace_light_paths = false;
        params.trace_camera_paths = true;
        params.do_bpt = false;
        params.do_bb1d = false;
        params.do_bp2d = false;
        params.do_pb2d = false;
        params.do_pp3d = false;
        params.do_surf = false;
    }

    // only connect from surf if both PREVIOUS and COMPATIBLE are disabled
    params.connect_to_camera_from_surf = (cfg->algorithm_flags & (PREVIOUS | COMPATIBLE)) == 0;

    screen_pixel_count = static_cast<uint32_t>(scene_loader->scene.camera.resolution.x) *
                         static_cast<uint32_t>(scene_loader->scene.camera.resolution.y);
    // we don't have the same number of pixels (camera paths) and light paths
    light_subpath_count = static_cast<uint32_t>(cfg->path_count_per_iter);

    bb1d_used_light_subpath_count = cfg->bb1d_used_light_subpath_count;
    if (bb1d_used_light_subpath_count < 0.f) {
        // if used bb1d sub paths is negative, it denotes a number relative
        // to the number of light sub paths
        bb1d_used_light_subpath_count =
            std::floor(-bb1d_used_light_subpath_count * static_cast<float>(light_subpath_count));
    }

    // copy the scene to params
    params.scene = scene_loader->scene;
    // handle the creation of acceleration structures
    create_gases(*scene_loader,
                 params,
                 gas_real_triangles,
                 gas_real_spheres,
                 gas_imaginary_triangles,
                 gas_imaginary_spheres,
                 scene_ias);
    // create the pipeline
    if (create_pipeline(cfg->gpu_assert)) return -1;
    // handle the arrays of the scene
    init_scene(*scene_loader, params);

    // create unit cylinder gas
    {
        gas_cylinder = {};
        CUdeviceptr d_tmp_buffer = 0;
        size_t tmp_buffer_size = 0;
        OptixAabb aabb;
        CUdeviceptr d_aabb;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb), sizeof(OptixAabb)));

        optix::unit_cylinder_bound(&aabb);
        optix::create_gas_map_custom_primitives(optix::DeviceState::get_instance(),
                                                &gas_cylinder,
                                                &aabb,
                                                &d_aabb,
                                                false,
                                                1,
                                                OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
                                                &d_tmp_buffer,
                                                &tmp_buffer_size);

        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_aabb)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tmp_buffer)));
    }

#ifdef DBG_CORE_UTIL
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dTraceCount), sizeof(uint64_t) * mLightSubPathCount));
#endif

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&params.light_subpath_starts), light_subpath_count * sizeof(unsigned int)));

    // allocate counters
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.light_vertex_counter), sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.light_beam_counter), sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.light_vertex_surface_counter), sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.light_vertex_medium_counter), sizeof(unsigned int)));

    // maximum number of vertices that can be on a light path.
    // there are at most max_path_length - 1 vertices as the first vertex is
    // not stored and light vertices are not stored on the camera object
    uint32_t max_light_vertices_per_path = cfg->max_path_length - 1;
    // this is a heuristic to reduce the beam storage
    const uint32_t avg_volume_segments_per_path_segment = 1u;
    // maximum number of beams that can be on a light path.
    // beams exist in-between vertices, hence - 1
    uint32_t max_light_beams_per_path = (max_light_vertices_per_path - 1) * avg_volume_segments_per_path_segment;

    // maximum number of light vertices that can be generated in one iteration.
    // determines the size of some allocations
    uint32_t light_vertex_max_count = light_subpath_count * max_light_vertices_per_path;
    uint32_t beam_paths = 0;
    if (params.do_bb1d) {
        beam_paths = max(beam_paths, static_cast<uint32_t>(bb1d_used_light_subpath_count));
    }
    if (params.do_bp2d) {
        beam_paths = max(beam_paths, light_subpath_count);
    }
    // maximum number of light beams that can be generated in one iteration.
    uint32_t light_beam_max_count = beam_paths * max_light_beams_per_path;

    // init framebuffer
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.framebuffer.color), screen_pixel_count * sizeof(float3)));
    colors = static_cast<float3*>(malloc(sizeof(float3) * screen_pixel_count));

    // allocate launch parameters on device
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(LaunchParams)));

    /** all fixed size device allocations are done at this point */

    size_t free_bytes, total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    // bytes for storing a single vertex
    // as we do not know beforehand whether it will be on surface or in media
    // we must allocate enough space for both arrays.
    const size_t vertex_bytes = sizeof(LightVertex) + sizeof(OptixAabb) * 2ull + sizeof(unsigned int) * 2ull;

    // bytes for storing a single beam
    const size_t beam_bytes = sizeof(LightBeam) + sizeof(OptixInstance);

    // make a decision on how many vertices and beams to allocate, based
    // on the remaining device memory
    const size_t desired_bytes = light_vertex_max_count * vertex_bytes + light_beam_max_count * beam_bytes;

    // heuristic: enough memory should be left for API to operate
    // this incorporates the memory needed to build and keep the ASes in memory
    const size_t api_bytes = 2000ull * 1024ull * 1024ull; // 2000 MB
    assert(free_bytes > api_bytes);
    const size_t usable_bytes = free_bytes - api_bytes;

    if (usable_bytes >= desired_bytes) {
        // device has enough memory left, allocate the maximum amount
        params.light_vertex_count = light_vertex_max_count;
        params.light_beam_count = light_beam_max_count;
    } else {
        // device does not have enough memory left over, allocate as much as
        // possible, relative to how much is needed for vertices and beams
        const size_t bytes_path_vertices = vertex_bytes * max_light_vertices_per_path;
        const size_t bytes_path_beams = beam_bytes * max_light_beams_per_path;
        auto bytes_vertices = static_cast<size_t>(
            (static_cast<float>(bytes_path_vertices) / static_cast<float>(bytes_path_vertices + bytes_path_beams)) *
            static_cast<float>(usable_bytes));
        auto bytes_beams = static_cast<size_t>(
            (static_cast<float>(bytes_path_beams) / static_cast<float>(bytes_path_vertices + bytes_path_beams)) *
            static_cast<float>(usable_bytes));

        params.light_vertex_count = static_cast<uint32_t>(bytes_vertices / vertex_bytes);
        params.light_beam_count = static_cast<uint32_t>(bytes_beams / beam_bytes);
    }

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&params.light_vertices), params.light_vertex_count * sizeof(LightVertex)));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&params.aabb_surface), params.light_vertex_count * sizeof(OptixAabb)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.indices_surface),
                          params.light_vertex_count * sizeof(unsigned int)));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&params.aabb_medium), params.light_vertex_count * sizeof(OptixAabb)));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&params.indices_medium), params.light_vertex_count * sizeof(unsigned int)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.light_beams), params.light_beam_count * sizeof(LightBeam)));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&params.instances_beam), params.light_beam_count * sizeof(OptixInstance)));

    const bool do_merging = cfg->max_path_length > 1;
    build_bb1d = params.trace_light_paths && do_merging && params.do_bb1d;
    build_bp2d = params.trace_light_paths && do_merging && params.do_bp2d;
    build_pb2d = params.trace_light_paths && do_merging && params.do_pb2d;
    build_pp3d = params.trace_light_paths && do_merging && params.do_pp3d;
    build_surf = params.trace_light_paths && do_merging && params.do_surf;

    params.rng_offset = cfg->iteration_offset * light_subpath_count;
    params.min_path_length = cfg->min_path_length;
    params.max_path_length = cfg->max_path_length;
    params.photon_beam_type = cfg->photon_beam_type;
    params.camera_beam_type = cfg->query_beam_type;
    params.subpath_count_bpt = float(params.do_bpt); // * 1.f;
    params.subpath_count_bb1d = float(params.do_bb1d) * bb1d_used_light_subpath_count;
    params.subpath_count_bp2d = float(params.do_bp2d) * light_subpath_count;
    params.subpath_count_pb2d = float(params.do_pb2d) * light_subpath_count;
    params.subpath_count_pp3d = float(params.do_pp3d) * light_subpath_count;
    params.subpath_count_surf = float(params.do_surf) * light_subpath_count;
    params.build_bb1d = build_bb1d;
    params.build_bp2d = build_bp2d;
    params.build_pb2d = build_pb2d;
    params.build_pp3d = build_pp3d;
    params.build_surf = build_surf;
    params.ignore_fully_spec_paths = cfg->ignore_fully_spec_paths;
    params.screen_pixel_count = screen_pixel_count;
    params.light_subpath_count = light_subpath_count;
    params.estimator_techniques = cfg->algorithm_flags;
    params.bb1d_used_light_sub_path_count = static_cast<unsigned int>(bb1d_used_light_subpath_count);
    params.framebuffer.resolution = make_uint2(static_cast<unsigned int>(scene_loader->scene.camera.resolution.x),
                                               static_cast<unsigned int>(scene_loader->scene.camera.resolution.y));
    params.use_shading_normal = cfg->use_shading_normal;
    params.handle_gas_beam = gas_cylinder.handle;
    params.sbt_offset = static_cast<uint32_t>(scene_loader->record_count * RAY_TYPE_COUNT);
    params.handle = scene_ias.handle;
#ifdef DBG_CORE_UTIL
    params.trace_count = reinterpret_cast<unsigned long long*>(dTraceCount);
#endif
    return 0;
}
