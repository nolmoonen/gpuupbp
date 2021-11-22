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

#ifndef MISC_OPTIX_HELPER_HPP
#define MISC_OPTIX_HELPER_HPP

#include <cstdint>
#include <optix_stubs.h>
#include <string>
#include <vector>
#include <vector_types.h>

/** This file contains OptiX helper code. */

namespace optix {
/// Type to align the address to a multiple of {OPTIX_SBT_RECORD_ALIGNMENT}.
template <typename T>
struct RecordData {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

/// Static function passed to {optixDeviceContextCreate} as a log callback.
void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */);

/// Global device state.
struct DeviceState {
    /// Set by {create_context}.
    OptixDeviceContext context = nullptr;

    /// Set by {create_stream}.
    CUstream stream = nullptr;

    /// https://stackoverflow.com/questions/1008019/c-singleton-design-pattern
    static DeviceState* get_instance()
    {
        static DeviceState instance;
        return &instance;
    }

    DeviceState() { create_context(this); }

    ~DeviceState() { delete_context(this); }

    /// Initializes global device state (CUDA and OptiX)
    /// Independent of all programs, geometry, etc.
    static void create_context(DeviceState* state);

    /// Deletes global device state.
    static void delete_context(DeviceState* state);

  public:
    DeviceState(DeviceState const&) = delete;

    void operator=(DeviceState const&) = delete;
};

struct Instance {
    float transform[12];
};

enum GasType { ANY, SINGLE_GAS, SINGLE_LEVEL_INSTANCING };

enum PrimType { ANY_, CUSTOM, TRIANGLE, QUADRATIC_BSPLINE };

/// Initializes the pipeline compile options.
void init_pipeline_compile_options(OptixPipelineCompileOptions* optix_pipeline_compile_options,
                                   uint32_t num_payload_val,
                                   uint32_t num_attribute_val,
                                   GasType gas_type,
                                   PrimType prim_type,
                                   bool use_exceptions);

/// Initializes the module compile options.
void init_module_compile_options(OptixModuleCompileOptions* module_compile_options);

/// Creates an OptiX module, given options and a filename.
/// Returns -1 on failure, 0 on success.
int32_t create_module(DeviceState* s,
                      OptixPipelineCompileOptions* optix_pipeline_compile_options,
                      OptixModuleCompileOptions* module_compile_options,
                      const char* filename,
                      OptixModule* module);

/// Initializes the program group options.
void init_program_group_options(OptixProgramGroupOptions* program_group_options);

/// Creates a exception program group.
void create_exception_program_group(DeviceState* s,
                                    OptixProgramGroupOptions* program_group_options,
                                    OptixProgramGroup* group,
                                    const char* function_name,
                                    OptixModule module);

/// Creates a single ray generation program group.
void create_raygen_program_group(DeviceState* s,
                                 OptixProgramGroupOptions* program_group_options,
                                 OptixProgramGroup* group,
                                 const char* function_name,
                                 OptixModule module);

/// Creates a single miss program group.
void create_miss_program_group(DeviceState* s,
                               OptixProgramGroupOptions* program_group_options,
                               OptixProgramGroup* group,
                               const char* function_name,
                               OptixModule module);

/// Creates a single hitgroup program group.
void create_hitgroup_program_group(DeviceState* s,
                                   OptixProgramGroupOptions* program_group_options,
                                   OptixProgramGroup* group,
                                   const char* function_name_ah,
                                   OptixModule module_ah,
                                   const char* function_name_ch,
                                   OptixModule module_ch,
                                   const char* function_name_is,
                                   OptixModule module_is);

/// Creates a single callable program group.
void create_continuation_callable_program_group(DeviceState* s,
                                                OptixProgramGroupOptions* program_group_options,
                                                OptixProgramGroup* group,
                                                const char* function_name_cc,
                                                OptixModule module);

/// Creates pipeline given the program groups.
/// @param max_trace_depth Maximum trace recursion depth. 0 means a ray
/// generation program can be launched, but can't trace any rays.
void create_pipeline(DeviceState* s,
                     OptixPipeline* pipeline,
                     OptixPipelineCompileOptions* pipeline_compile_options,
                     OptixProgramGroup* program_groups,
                     uint32_t program_group_count,
                     uint32_t max_trace_depth,
                     // The maximum depth of a traversable graph passed to trace.
                     uint32_t max_traversable_graph_depth);

/// Creates the raygen sbt record.
void create_sbt_raygen_record(OptixShaderBindingTable* sbt, void* rg_sbt, size_t rg_size, OptixProgramGroup rg_group);

/// Creates the miss sbt records.
void create_sbt_miss_records(OptixShaderBindingTable* sbt,
                             void* ms_sbt,
                             size_t ms_size,
                             OptixProgramGroup ms_group[],
                             uint32_t ms_group_count,
                             uint32_t ray_type_count);

/// Creates the hitgroup sbt records.
void create_sbt_hitgroup_records(OptixShaderBindingTable* sbt,
                                 void* hg_sbt,
                                 size_t hg_size,
                                 OptixProgramGroup hg_group[],
                                 uint32_t hg_group_count,
                                 uint32_t ray_type_count,
                                 uint32_t material_count);

/// Creates the exception sbt record.
void create_sbt_exception_record(OptixShaderBindingTable* sbt,
                                 void* exc_sbt,
                                 size_t exc_size,
                                 OptixProgramGroup exc_group);

struct accel_struct {
    OptixAccelBuildOptions accel_options{};
    OptixBuildInput input{};
    OptixTraversableHandle handle = 0;
    CUdeviceptr d_output_buffer = 0;
    size_t output_buffer_size = 0;
};

/// enlargens buffer, updates {curr_size} */
void enlarge_buffer(CUdeviceptr* buffer, size_t* curr_size, size_t new_size);

void complete_creation(DeviceState* device_state,
                       accel_struct* gas_map,
                       CUdeviceptr* temp_buffer,
                       size_t* temp_buffer_size,
                       bool updatable,
                       bool compact);

void delete_accel_struct(accel_struct* gas_map);

/// note: takes (CUdeviceptr *) since build input requests an array
/// can also be called to rebuild the gas, as memory allocation will only
/// grow
void create_gas_map_custom_primitives(DeviceState* device_state,
                                      accel_struct* gas_map,
                                      OptixAabb* aabb_host,
                                      CUdeviceptr* aabb_device,
                                      bool updatable,
                                      uint32_t num_primitives,
                                      unsigned int flags,
                                      CUdeviceptr* temp_buffer,
                                      size_t* temp_buffer_size);

/// Returns AABB for a sphere.
void inline sphere_bound(float center_x, float center_y, float center_z, float radius, float result[6])
{
    auto* aabb = reinterpret_cast<OptixAabb*>(result);
    *aabb = {center_x - radius,
             center_y - radius,
             center_z - radius,
             center_x + radius,
             center_y + radius,
             center_z + radius};
}

/// Sets a bounding box for a cylinder of radius 1 with origin at (0,0,0)
/// and direction (0,1,0) and length 1.
void inline unit_cylinder_bound(OptixAabb* result)
{
    result->minX = -1.f;
    result->minY = 0.f;
    result->minZ = -1.f;
    result->maxX = 1.f;
    result->maxY = 1.f;
    result->maxZ = 1.f;
}

/// Report GPU memory usage in MB.
void report_memory(double& used, double& free, double& total);
} // namespace optix

#endif // MISC_OPTIX_HELPER_HPP
