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

#include "optix_helper.hpp"
#include "../host/exception.hpp"
#include "../shared/matrix.h"
#include "../shared/vec_math.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>

#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>

void optix::context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(16) << tag << "]: " << message << "\n";
}

void optix::DeviceState::create_context(DeviceState* state)
{
    // initialize CUDA
    CUDA_CHECK(cudaFree(0));

    // initialize OptiX
    OptixDeviceContext context;
    CUcontext cu_ctx = 0; // zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    // hide status and progress messages in release builds
#ifdef NDEBUG
    options.logCallbackLevel = 3;
#else
    options.logCallbackLevel = 4;
#endif
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
    //    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));
    state->context = context;

    // initialize stream
    CUDA_CHECK(cudaStreamCreate(&state->stream));
}

void optix::DeviceState::delete_context(DeviceState* state) { OPTIX_CHECK(optixDeviceContextDestroy(state->context)); }

void optix::init_pipeline_compile_options(OptixPipelineCompileOptions* optix_pipeline_compile_options,
                                          uint32_t num_payload_val,
                                          uint32_t num_attribute_val,
                                          GasType gas_type,
                                          PrimType prim_type,
                                          bool use_exceptions)
{
    optix_pipeline_compile_options->usesMotionBlur = false;
    switch (gas_type) {
    case ANY:
        optix_pipeline_compile_options->traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        break;
    case SINGLE_GAS:
        optix_pipeline_compile_options->traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        break;
    case SINGLE_LEVEL_INSTANCING:
        optix_pipeline_compile_options->traversableGraphFlags =
            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        break;
    }
    optix_pipeline_compile_options->numPayloadValues = static_cast<int>(num_payload_val);
    optix_pipeline_compile_options->numAttributeValues = static_cast<int>(num_attribute_val);
    if (use_exceptions) {
        optix_pipeline_compile_options->exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                                         OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
                                                         OPTIX_EXCEPTION_FLAG_USER;
    } else {
        optix_pipeline_compile_options->exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    }
    optix_pipeline_compile_options->pipelineLaunchParamsVariableName = "params";
    switch (prim_type) {
    case ANY_:
        optix_pipeline_compile_options->usesPrimitiveTypeFlags =
            OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM | OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        break;
    case CUSTOM:
        optix_pipeline_compile_options->usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
        break;
    case TRIANGLE:
        optix_pipeline_compile_options->usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        break;
    case QUADRATIC_BSPLINE:
        optix_pipeline_compile_options->usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE;
        break;
    }
}

void optix::init_module_compile_options(OptixModuleCompileOptions* module_compile_options)
{
    module_compile_options->maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options->optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options->debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
}

static bool read_source_file(std::string& str, const std::string& filename)
{
    // try to open file
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (file.good()) {
        // found usable source file
        std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
        str.assign(buffer.begin(), buffer.end());
        return true;
    }
    return false;
}

int32_t optix::create_module(DeviceState* s,
                             OptixPipelineCompileOptions* optix_pipeline_compile_options,
                             OptixModuleCompileOptions* module_compile_options,
                             const char* filename,
                             OptixModule* module)
{
    std::string ptx;
    std::string source_file_path(filename);
    source_file_path += ".ptx";
    if (!read_source_file(ptx, source_file_path)) {
        fprintf(stderr, "Couldn't open source file \"%s\"\n", source_file_path.c_str());
        return -1;
    }

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(s->context,
                                             module_compile_options,
                                             optix_pipeline_compile_options,
                                             ptx.c_str(),
                                             ptx.size(),
                                             log,
                                             &sizeof_log,
                                             module));

    return 0;
}

void optix::init_program_group_options(OptixProgramGroupOptions* program_group_options)
{
    // intentionally empty
    // note: changing this has effect on ALL programs.
}

void optix::create_exception_program_group(DeviceState* s,
                                           OptixProgramGroupOptions* program_group_options,
                                           OptixProgramGroup* group,
                                           const char* function_name,
                                           OptixModule module)
{
    char log[2048];
    size_t sizeof_log = sizeof(log);

    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
    raygen_prog_group_desc.exception.module = module;
    raygen_prog_group_desc.exception.entryFunctionName = function_name;

    OPTIX_CHECK_LOG(optixProgramGroupCreate(s->context,
                                            &raygen_prog_group_desc,
                                            1, // num program groups
                                            program_group_options,
                                            log,
                                            &sizeof_log,
                                            group));
}

void optix::create_raygen_program_group(DeviceState* s,
                                        OptixProgramGroupOptions* program_group_options,
                                        OptixProgramGroup* group,
                                        const char* function_name,
                                        OptixModule module)
{
    char log[2048];
    size_t sizeof_log = sizeof(log);

    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = module;
    raygen_prog_group_desc.raygen.entryFunctionName = function_name;

    OPTIX_CHECK_LOG(optixProgramGroupCreate(s->context,
                                            &raygen_prog_group_desc,
                                            1, // num program groups
                                            program_group_options,
                                            log,
                                            &sizeof_log,
                                            group));
}

void optix::create_miss_program_group(DeviceState* s,
                                      OptixProgramGroupOptions* program_group_options,
                                      OptixProgramGroup* group,
                                      const char* function_name,
                                      OptixModule module)
{
    char log[2048];
    size_t sizeof_log = sizeof(log);

    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = module;
    miss_prog_group_desc.miss.entryFunctionName = function_name;

    OPTIX_CHECK_LOG(optixProgramGroupCreate(s->context,
                                            &miss_prog_group_desc,
                                            1, // num program groups
                                            program_group_options,
                                            log,
                                            &sizeof_log,
                                            group));
}

void optix::create_hitgroup_program_group(DeviceState* s,
                                          OptixProgramGroupOptions* program_group_options,
                                          OptixProgramGroup* group,
                                          const char* function_name_ah,
                                          OptixModule module_ah,
                                          const char* function_name_ch,
                                          OptixModule module_ch,
                                          const char* function_name_is,
                                          OptixModule module_is)
{
    char log[2048];
    size_t sizeof_log = sizeof(log);

    OptixProgramGroupDesc hit_prog_group_desc = {};
    hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleAH = module_ah;
    hit_prog_group_desc.hitgroup.entryFunctionNameAH = function_name_ah;
    hit_prog_group_desc.hitgroup.moduleCH = module_ch;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = function_name_ch;
    hit_prog_group_desc.hitgroup.moduleIS = module_is;
    hit_prog_group_desc.hitgroup.entryFunctionNameIS = function_name_is;

    OPTIX_CHECK_LOG(optixProgramGroupCreate(s->context,
                                            &hit_prog_group_desc,
                                            1, // num program groups
                                            program_group_options,
                                            log,
                                            &sizeof_log,
                                            group));
}

void optix::create_continuation_callable_program_group(DeviceState* s,
                                                       OptixProgramGroupOptions* program_group_options,
                                                       OptixProgramGroup* group,
                                                       const char* function_name_cc,
                                                       OptixModule module)
{
    char log[2048];
    size_t sizeof_log = sizeof(log);

    OptixProgramGroupDesc prog_group_desc = {};
    prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc.callables.moduleDC = nullptr;
    prog_group_desc.callables.entryFunctionNameDC = nullptr;
    prog_group_desc.callables.moduleCC = module;
    prog_group_desc.callables.entryFunctionNameCC = function_name_cc;

    OPTIX_CHECK_LOG(optixProgramGroupCreate(s->context,
                                            &prog_group_desc,
                                            1, // num program groups
                                            program_group_options,
                                            log,
                                            &sizeof_log,
                                            group));
}

void optix::create_pipeline(DeviceState* s,
                            OptixPipeline* pipeline,
                            OptixPipelineCompileOptions* pipeline_compile_options,
                            OptixProgramGroup* program_groups,
                            uint32_t program_group_count,
                            uint32_t max_trace_depth,
                            uint32_t max_traversable_graph_depth)
{
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(s->context,
                                        pipeline_compile_options,
                                        &pipeline_link_options,
                                        program_groups,
                                        program_group_count,
                                        log,
                                        &sizeof_log,
                                        pipeline));

    // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stack_sizes = {};
    for (uint32_t i = 0; i < program_group_count; i++) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(program_groups[i], &stack_sizes));
    }

    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes,
                                           max_trace_depth, // Maximum depth of optixTrace() calls.
                                           max_cc_depth,
                                           max_dc_depth,
                                           &direct_callable_stack_size_from_traversal,
                                           &direct_callable_stack_size_from_state,
                                           &continuation_stack_size));

    OPTIX_CHECK(optixPipelineSetStackSize(*pipeline,
                                          direct_callable_stack_size_from_traversal,
                                          direct_callable_stack_size_from_state,
                                          continuation_stack_size,
                                          max_traversable_graph_depth));
}

void optix::create_sbt_raygen_record(OptixShaderBindingTable* sbt,
                                     void* rg_sbt,
                                     size_t rg_size,
                                     OptixProgramGroup rg_group)
{
    // raygen program record
    CUdeviceptr d_raygen_record;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), rg_size));

    // pack header
    OPTIX_CHECK(optixSbtRecordPackHeader(rg_group, rg_sbt));

    // copy to device
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_raygen_record), rg_sbt, rg_size, cudaMemcpyHostToDevice));

    // set sbt variables
    sbt->raygenRecord = d_raygen_record;
}

void optix::create_sbt_miss_records(OptixShaderBindingTable* sbt,
                                    void* ms_sbt,
                                    size_t ms_size,
                                    OptixProgramGroup ms_group[],
                                    uint32_t ms_group_count,
                                    uint32_t ray_type_count)
{
    // miss program records
    CUdeviceptr d_miss_records;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_records), ms_size * ray_type_count));

    // pack headers
    auto p = static_cast<char*>(ms_sbt);
    for (uint32_t i = 0; i < ms_group_count; i++) {
        OPTIX_CHECK(optixSbtRecordPackHeader(ms_group[i], p + i * ms_size));
    }

    // copy to device
    CUDA_CHECK(
        cudaMemcpy(reinterpret_cast<void*>(d_miss_records), ms_sbt, ms_size * ray_type_count, cudaMemcpyHostToDevice));

    // set sbt variables
    sbt->missRecordBase = d_miss_records;
    sbt->missRecordStrideInBytes = static_cast<uint32_t>(ms_size);
    sbt->missRecordCount = ray_type_count;
}

void optix::create_sbt_hitgroup_records(OptixShaderBindingTable* sbt,
                                        void* hg_sbt,
                                        size_t hg_size,
                                        OptixProgramGroup hg_group[],
                                        uint32_t hg_group_count,
                                        uint32_t ray_type_count,
                                        uint32_t material_count)
{
    // hitgroup program records
    CUdeviceptr d_hitgroup_records;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_records), hg_size * ray_type_count * material_count));

    // pack headers
    auto p = static_cast<char*>(hg_sbt);
    for (uint32_t i = 0; i < material_count; i++) {
        for (uint32_t j = 0; j < hg_group_count; j++) {
            OPTIX_CHECK(optixSbtRecordPackHeader(hg_group[j], p + (i * ray_type_count + j) * hg_size));
        }
    }

    // copy to device
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_records),
                          hg_sbt,
                          hg_size * ray_type_count * material_count,
                          cudaMemcpyHostToDevice));

    // set sbt variables
    sbt->hitgroupRecordBase = d_hitgroup_records;
    sbt->hitgroupRecordStrideInBytes = static_cast<uint32_t>(hg_size);
    sbt->hitgroupRecordCount = ray_type_count * material_count;
}

void optix::create_sbt_exception_record(OptixShaderBindingTable* sbt,
                                        void* exc_sbt,
                                        size_t exc_size,
                                        OptixProgramGroup exc_group)
{
    // exception program record
    CUdeviceptr d_exception_record;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_exception_record), exc_size));

    // pack header
    OPTIX_CHECK(optixSbtRecordPackHeader(exc_group, exc_sbt));

    // copy to device
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_exception_record), exc_sbt, exc_size, cudaMemcpyHostToDevice));

    // set sbt variables
    sbt->exceptionRecord = d_exception_record;
}

void optix::enlarge_buffer(CUdeviceptr* buffer, size_t* curr_size, size_t new_size)
{
    if (*curr_size < new_size) {
        // NB: first time this will be nullptr, but that's okay
        CUDA_CHECK(cudaFree((void*)*buffer));
        *curr_size = new_size;
        CUDA_CHECK(cudaMalloc((void**)buffer, *curr_size));
    }
}

void optix::complete_creation(DeviceState* device_state,
                              accel_struct* gas_map,
                              CUdeviceptr* temp_buffer,
                              size_t* temp_buffer_size,
                              bool updatable,
                              bool compact)
{
    // flags in gas map should equal actual parameter
    assert(compact == static_cast<bool>(gas_map->accel_options.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION));
    assert(updatable == static_cast<bool>(gas_map->accel_options.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE));

    // get non-compacted required sizes
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        device_state->context, &gas_map->accel_options, &gas_map->input, 1, &gas_buffer_sizes));

    // enlarge temp buffer if need be
    size_t tmp_size_in_bytes = gas_buffer_sizes.tempSizeInBytes;
    if (updatable) {
        // if gas is updatable, tmp buffer should at least also accommodate that requirement
        tmp_size_in_bytes = std::max(tmp_size_in_bytes, gas_buffer_sizes.tempUpdateSizeInBytes);
    }
    enlarge_buffer(temp_buffer, temp_buffer_size, tmp_size_in_bytes);

    // enlarge output buffer if need be
    size_t out_size_in_bytes = gas_buffer_sizes.outputSizeInBytes;
    enlarge_buffer(&gas_map->d_output_buffer,
                   &gas_map->output_buffer_size, // this is zero at this point
                   out_size_in_bytes);

    OptixAccelEmitDesc* emit_properties = nullptr;
    uint32_t emit_property_count = 0;

    // prevent scope destroying these objects
    OptixAccelEmitDesc emitProperty = {};
    CUdeviceptr d_compacted_size = 0;
    if (compact) {
        // if gas should be compacted, acquire size of compacted gas when building
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_compacted_size), sizeof(size_t)));

        emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result = d_compacted_size;

        emit_properties = &emitProperty;
        emit_property_count = 1;
    }

    OPTIX_CHECK(optixAccelBuild(device_state->context,
                                nullptr,
                                &gas_map->accel_options,
                                &gas_map->input,
                                1,
                                *temp_buffer,
                                gas_buffer_sizes.tempSizeInBytes,
                                gas_map->d_output_buffer,
                                gas_buffer_sizes.outputSizeInBytes,
                                &gas_map->handle,
                                emit_properties,
                                emit_property_count));

    if (compact) {
        // get size of compacted gas into host memory, free on device memory
        size_t compacted_gas_size;
        CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree((void*)d_compacted_size));

        // shrink allocated output memory if possible
        if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
            // allocate a new buffer of the correct size
            CUdeviceptr tmp_out_buffer = 0;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tmp_out_buffer), compacted_gas_size));

            // compact the old out buffer in the new out buffer
            OPTIX_CHECK(optixAccelCompact(
                device_state->context, nullptr, gas_map->handle, tmp_out_buffer, compacted_gas_size, &gas_map->handle));

            // free the old buffer and assign new buffer
            CUDA_CHECK(cudaFree((void*)gas_map->d_output_buffer));
            gas_map->d_output_buffer = tmp_out_buffer;
        }
    }
}

void optix::delete_accel_struct(accel_struct* gas_map)
{
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(gas_map->d_output_buffer)));
    gas_map->output_buffer_size = 0;
    // necessary so cudaFree can be called without any issues
    gas_map->d_output_buffer = 0;
}

void optix::create_gas_map_custom_primitives(DeviceState* device_state,
                                             accel_struct* gas_map,
                                             OptixAabb* aabb_host,
                                             CUdeviceptr* aabb_device,
                                             bool updatable,
                                             uint32_t num_primitives,
                                             unsigned int flags,
                                             CUdeviceptr* temp_buffer,
                                             size_t* temp_buffer_size)
{
    // copy host memory to device
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(*aabb_device), aabb_host, sizeof(OptixAabb) * num_primitives, cudaMemcpyHostToDevice));

    gas_map->accel_options = {};
    gas_map->accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    if (updatable) {
        gas_map->accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    }
    gas_map->accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    gas_map->input = {};
    gas_map->input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    gas_map->input.customPrimitiveArray.aabbBuffers = aabb_device;
    gas_map->input.customPrimitiveArray.numPrimitives = num_primitives;
    gas_map->input.customPrimitiveArray.flags = &flags;
    gas_map->input.customPrimitiveArray.numSbtRecords = 1;

    complete_creation(device_state, gas_map, temp_buffer, temp_buffer_size, updatable, false);
}

void optix::report_memory(double& used, double& free, double& total)
{
    size_t free_byte, total_byte;
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));
    free = static_cast<double>(free_byte) / 1024.0 / 1024.0;
    total = static_cast<double>(total_byte) / 1024.0 / 1024.0;
    used = total - free;
}
