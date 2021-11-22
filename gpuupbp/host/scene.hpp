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

#ifndef HOST_SCENE_HPP
#define HOST_SCENE_HPP

#include "../shared/scene.h"
#include "exception.hpp"

inline void delete_scene_host(const Scene& scene)
{
    free(scene.materials);
    free(scene.media);
    free(scene.lights);
    free(scene.triangles_real.verts);
    free(scene.triangles_real.norms);
    free(scene.triangles_real.records);
    free(scene.triangles_imag.verts);
    free(scene.triangles_imag.norms);
    free(scene.triangles_imag.records);
    free(scene.spheres_real.aabbs);
    free(scene.spheres_real.records);
    free(scene.spheres_imag.aabbs);
    free(scene.spheres_imag.records);
    free(scene.env_map.img.data);
    free(scene.env_map.distribution.data);
    free(scene.env_map.distribution.conditional_v);
}

inline void delete_scene_device(const Scene& scene)
{
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(scene.materials)));
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(scene.media)));
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(scene.lights)));
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(scene.triangles_real.verts)));
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(scene.triangles_real.norms)));
    // scene.triangles_real.records not allocated on device
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(scene.triangles_imag.verts)));
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(scene.triangles_imag.norms)));
    // scene.triangles_imag.records not allocated on device
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(scene.spheres_real.aabbs)));
    // scene.spheres_real.records not allocated on device
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(scene.spheres_imag.aabbs)));
    // scene.spheres_imag.records not allocated on device
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(scene.env_map.img.data)));
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(scene.env_map.distribution.data)));
    CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(scene.env_map.distribution.conditional_v)));
}

#endif // HOST_SCENE_HPP
