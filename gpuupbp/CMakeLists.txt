# Copyright (C) 2021, Nol Moonen
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom
# the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# (The above is MIT License: http://en.wikipedia.origin/wiki/MIT_License)

# Takes a list of files and "wraps" the .cu files in a custom command that calls NVCC to generate PTX.
# Input is a list of variadic arguments after generated_files, which is the output.
function(wrap_sources generated_files)
    set(_generated_files "")
    set(_files ${ARGN})
    set(nvcc_options --use_fast_math --machine=64 -rdc=true)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        list(APPEND nvcc_options -G)
    endif ()
    foreach (FILE ${_files})
        if (${FILE} MATCHES "\\.cu$")
            get_filename_component(FILE_NAME ${FILE} NAME_WE)
            set(OUTPUT_FILENAME "${FILE}.ptx")
            add_custom_command(
                    OUTPUT ${OUTPUT_FILENAME}
                    MAIN_DEPENDENCY ${FILE}
                    DEPENDS ${_files}
                    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -ptx ${nvcc_options} -I ${OptiX_INCLUDE} "${CMAKE_CURRENT_SOURCE_DIR}/${FILE}" -o ${OUTPUT_FILENAME})
            list(APPEND _generated_files ${OUTPUT_FILENAME})
        endif ()
    endforeach ()
    set(${generated_files} ${_generated_files})
endfunction()

set(SOURCES
        host/framebuffer.cpp

        kernel/pde/functs_bb1d.cuh
        kernel/pde/functs_bp2d.cuh
        kernel/pde/functs_pb2d.cuh
        kernel/pde/functs_pp.cuh
        kernel/pde/functs_surf.cuh
        kernel/pde/functs_pp3d.cuh
        kernel/pde/program_bb1d.cu
        kernel/pde/program_pb2d.cu
        kernel/pde/program_surf.cu
        kernel/pde/program_pp3d.cu
        kernel/pde/program_bp2d.cu
        kernel/bsdf.cuh
        kernel/camera.cuh
        kernel/defs.cuh
        kernel/distribution.cuh
        kernel/env_map.cuh
        kernel/exception.cu
        kernel/frame_buffer.cuh
        kernel/functs_camera.cuh
        kernel/functs_light.cuh
        kernel/functs_shared.cuh
        kernel/intersection.cuh
        kernel/intersector.cu
        kernel/intersector.cuh
        kernel/light.cuh
        kernel/medium.cuh
        kernel/optix_util.cuh
        kernel/params_def.cuh
        kernel/path_weight.cuh
        kernel/phase_function.cuh
        kernel/pstack.cuh
        kernel/rng.cuh
        kernel/sample.cuh
        kernel/sarray.cuh
        kernel/scene.cuh
        kernel/trace_camera.cu
        kernel/trace_light.cu
        kernel/types.cuh

        misc/config.cpp
        misc/logger.cpp
        misc/optix_helper.cpp
        misc/scene_loader_obj.cpp
        misc/scene_loader_predef.cpp
        misc/timer.cpp
        misc/tinyexr_wrapper.cpp

        renderer/renderer.cpp
        renderer/renderer_init.cpp

        shared/bsdf.h
        shared/camera.h
        shared/env_map.h
        host/exception.hpp
        shared/frame.h
        shared/intersect_defs.h
        shared/launch_params.h
        shared/light.h
        shared/light_vertex.h
        shared/material.h
        shared/matrix.h
        shared/medium.h
        shared/preprocessor.h
        shared/record.h
        shared/scene.h
        shared/shared_defs.h
        shared/shared_enums.h
        shared/vec_math_ext.h
        shared/vec_math.h

        gpuupbp.cpp)

# wrap the .cu sources to generate ptx
wrap_sources(generated_files ${SOURCES})

add_executable(gpuupbp ${SOURCES} ${generated_files})
target_link_libraries(gpuupbp PRIVATE CUDA::cudart_static OptiX tinyexr)
