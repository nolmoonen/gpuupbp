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

project(gpuupbp-project)

# minimum requirement for CUDAToolkit
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set a default build type if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

# find CUDAToolkit package for access to NVCC
find_package(CUDAToolkit REQUIRED)

# find OptiX and set OptiX_INCLUDE
if (NOT OptiX_ROOT)
    message(FATAL_ERROR "Specify OptiX installed location by setting \"OptiX_ROOT\".")
endif ()

set(OptiX_PATH ${OptiX_ROOT} CACHE PATH "Path to OptiX installed location.")

find_path(OptiX_INCLUDE NAMES optix.h PATHS "${OptiX_PATH}/include" NO_DEFAULT_PATH)
if (NOT OptiX_INCLUDE)
    message(FATAL_ERROR "OptiX not found in location \"${OptiX_PATH}\".")
else ()
    message(STATUS "OptiX found in location \"${OptiX_PATH}\".")
endif ()

add_library(OptiX INTERFACE)
target_include_directories(OptiX INTERFACE ${OptiX_INCLUDE})

add_subdirectory(benchmark)
add_subdirectory(deps)
add_subdirectory(gpuupbp)
add_subdirectory(tools)
