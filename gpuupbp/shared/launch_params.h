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

#ifndef SHARED_LAUNCH_PARAMS_H
#define SHARED_LAUNCH_PARAMS_H

#include "frame.h"
#include "framebuffer.h"
#include "light_vertex.h"
#include "scene.h"
#include "shared_enums.h"

#include <optix.h>
#include <vector_types.h>

/// Uncommenting will measure and print RT Core util.
/// Preprocessor macro as to not cause overhead in default case.
// #define DBG_CORE_UTIL

/// All parameters placed in constant GPU memory when the pipeline is launched.
struct LaunchParams {
    /**
     * parameters that remain constant during lifetime of renderer.
     * and thus are set in constructor of renderer */

    /// Constant scene data.
    Scene scene;
    /// Minimum path length in # of segments.
    unsigned int min_path_length;
    /// Minimum path length in # of segments.
    unsigned int max_path_length;
    BeamType photon_beam_type;
    BeamType camera_beam_type;
    /// n_bpt:  number of light subpaths each camera subpath uses for BPT
    float subpath_count_bpt;
    /// n_bb:   number of light subpaths each camera subpath uses for BB1D
    float subpath_count_bb1d;
    /// n_bp:   number of light subpaths each camera subpath uses for BP2D
    float subpath_count_bp2d;
    /// n_pb:   number of light subpaths each camera subpath uses for PB2D
    float subpath_count_pb2d;
    /// n_pp:   number of light subpaths each camera subpath uses for PP3D
    float subpath_count_pp3d;
    /// n_surf: number of light subpaths each camera subpath uses for SURF
    float subpath_count_surf;
    bool trace_light_paths;
    bool trace_camera_paths;
    /// Whether to evaluate BPT.
    bool do_bpt;
    /// Whether to evaluate BB1D.
    bool do_bb1d;
    /// Whether to evaluate BP2D.
    bool do_bp2d;
    /// Whether to evaluate PB2D.
    bool do_pb2d;
    /// Whether to evaluate PP3D.
    bool do_pp3d;
    /// Whether to evaluate SURF.
    bool do_surf;
    /// Whether to build a photon map for BB1D.
    bool build_bb1d;
    /// Whether to build a photon map for BP2D.
    bool build_bp2d;
    /// Whether to build a photon map for PB2D.
    bool build_pb2d;
    /// Whether to build a photon map for PP3D.
    bool build_pp3d;
    /// Whether to build a photon map for SURF.
    bool build_surf;
    /// Whether camera connections may be made from surfaces.
    bool connect_to_camera_from_surf;
    /// Whether to ignore fully specular paths from camera.
    bool ignore_fully_spec_paths;
    unsigned int screen_pixel_count;
    unsigned int light_subpath_count;
    unsigned int estimator_techniques;
    unsigned int bb1d_used_light_sub_path_count;
    Framebuffer framebuffer;
    /// Whether to use the shading normal in intersections.
    bool use_shading_normal;
    /// The geometry acceleration handle for the beam used for the BB1D and BP2D
    /// photon maps.
    OptixTraversableHandle handle_gas_beam;
    /// The SBT offset used for photon maps.
    unsigned int sbt_offset;
    /// Array of AABBs for surface photon points.
    OptixAabb* aabb_surface;
    /// Array of AABBs for medium photon points.
    OptixAabb* aabb_medium;
    /// An index for each surface AABB to its corresponding point in the
    /// array of light vertices.
    unsigned int* indices_surface;
    /// An index for each medium AABB to its corresponding point in the
    /// array of light vertices.
    unsigned int* indices_medium;
    /// Array of instance for photon beams.
    OptixInstance* instances_beam;
    /// Traversable for the scene representation.
    OptixTraversableHandle handle;
    /// The number of iterations to offset the random number generator by.
    unsigned long long rng_offset;

    /**
     * Parameters that are changed by the iteration number. */

    /** Iteration-dependent contribution normalization. */
    /// 1 / (n_bb1d)
    float bb1d_normalization;
    /// 1 / (n_bp2d)
    float bp2d_normalization;
    /// 1 / (n_pb2d)
    float pb2d_normalization;
    /// 1 / (K3^{-1} * n_pp3d)
    float pp3d_normalization;
    /// 1 / (K2^{-1} * n_surf)
    float surf_normalization;
    /** Iteration-dependent MIS weights. */
    /// (n_\bb1d) / (K1)
    float mis_factor_bb1d;
    /// (n_\bp2d) / (K2)
    float mis_factor_bp2d;
    /// (n_\pb2d) / (K2)
    float mis_factor_pb2d;
    /// (n_\pp3d) / (K3)
    float mis_factor_pp3d;
    /// (n_\surf) / (K2)
    float mis_factor_surf;
    /** radii */
    float radius_bb1d;
    float radius_bp2d;
    float radius_pb2d;
    float radius_pp3d;
    float radius_surf;
    /// Radius for medium points.
    float rad_point_med;
    /// Radius for beams in media.
    float rad_beam;
    /** squared radii */
    float radius_bb1d_2;
    float radius_bp2d_2;
    float radius_pb2d_2;
    float radius_pp3d_2;
    float radius_surf_2;
    /// Iteration index, needed as seed for random number generator.
    unsigned int iteration;
    /// The index of the first vertex of the light subpaths. UINT32_MAX if none.
    unsigned int* light_subpath_starts;
    /// Counter of surface vertices.
    unsigned int* light_vertex_surface_counter;
    /// Counter of medium vertices.
    unsigned int* light_vertex_medium_counter;
    /// Counter for vertices.
    unsigned int* light_vertex_counter;
    /// The capacity of {light_vertices}.
    unsigned int light_vertex_count;
    /// Array of light vertices.
    LightVertex* light_vertices;
    /// Counter for beams.
    unsigned int* light_beam_counter;
    /// The capacity of {light_beams}.
    unsigned int light_beam_count;
    /// Array of light beams.
    LightBeam* light_beams;
    /**
     * parameters that are changed by the results of the light trace pass
     */
    /// Traversable for the point-based photon map.
    OptixTraversableHandle handle_points;
    /// Traversable for the beam-based photon map.
    OptixTraversableHandle handle_beams;
#ifdef DBG_CORE_UTIL
    /// In the launch dimension, a counter measuring the amount of trace calls.
    unsigned long long* trace_count;
#endif
};

#endif // SHARED_LAUNCH_PARAMS_H
