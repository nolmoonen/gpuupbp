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

#ifndef MISC_CONFIG_HPP
#define MISC_CONFIG_HPP

#include "../host/framebuffer.hpp"
#include "../shared/shared_enums.h"
#include "scene_config.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

/// Renderer configuration, holds algorithm, scene, and all other settings.
struct Config {
    int scene_id;
    std::string scene_obj_file;

    /// Flags of the algorithm used for rendering. Can contain
    /// {estimator_technique} and {other_settings} values. Controls information
    /// display, the name of the output image file and settings of renderer.
    uint32_t algorithm_flags;

    /// Number of rendering iterations.
    int iteration_count;
    /// Maximum time the rendering can take.
    float max_time;
    /// Name of the output image file.
    // todo split this in name and extension
    std::string output_name;
    /// Resolution of the rendered image.
    uint2 resolution;

    /// Maximum length of constructed paths.
    uint32_t max_path_length;
    /// Minimum length of constructed paths.
    uint32_t min_path_length;

    /// Number of paths traced from lights per iteration.
    float path_count_per_iter;

    /// Used type of query beams.
    BeamType query_beam_type;
    /// Used type of photon beams.
    BeamType photon_beam_type;

    /// Initial radius for BB1D.
    float bb1d_radius_initial;
    /// Radius reduction factor for BB1D.
    float bb1d_radius_alpha;
    /// Number of light paths used to generate photon beams.
    float bb1d_used_light_subpath_count;

    /// Initial radius for BP2D.
    float bp2d_radius_initial;
    /// Radius reduction factor for BP2D.
    float bp2d_radius_alpha;

    /// Initial radius for PB2D.
    float pb2d_radius_initial;
    /// Radius reduction factor for PB2D.
    float pb2d_radius_alpha;

    /// Initial radius for PP3D.
    float pp3d_radius_initial;
    /// Radius reduction factor for PP3D.
    float pp3d_radius_alpha;

    /// Initial radius for surface photon mapping.
    float surf_radius_initial;
    /// Radius reduction factor for surface photon mapping.
    float surf_radius_alpha;

    /// String appended to the name of the output image file that contains
    /// additional arguments specified on the command line.
    std::string additional_args;
    /// Value x > 0 means generating one image per x iterations.
    int continuous_output;

    /// Set to ignore fully specular paths from camera.
    bool ignore_fully_spec_paths;

    /// Whether to enable GPU assertions.
    bool gpu_assert;
    /// Number of iterations to offset random number generation with.
    unsigned long long iteration_offset;
    /// Whether to output a log containing times and sizes.
    bool do_log;
    /// Whether to use shading normals.
    bool use_shading_normal;
};

/// Gets a description of the given configuration.
///
/// Description is a multi line string that summarizes all settings in the given
/// configuration. It is displayed at the beginning of the rendering.
/// Leading spaces may be specified to indent the description.
std::string get_description(const Config& config);

/// Initializes scene configurations.
/// This is where the predefined scenes listed in the help are defined.
std::vector<SceneConfig> init_scene_configs();

/// Creates a default name of the output image file that contains no scene
/// identifier.
/// If user does not specify their own name of the output image file, default
/// one is created based on parameters used for rendering. This one lacks any
/// information about rendered scene and is used by default_filename methods
/// that add proper scene identifier based on whether predefined or user scene
/// is rendered.
std::string default_filename_without_scene(const Config& config);

/// Creates a default name of the output image file for a predefined scene.
/// If user does not specify his own name of the output image file, default one
/// is created based on parameters used for rendering. This one contains
/// identifier of a predefined scene.
std::string default_filename(int scene_id, const Config& config);

/// Creates a default name of the output image file for a user scene.
/// If user does not specify their own name of the output image file, default
/// one is created based on parameters used for rendering. This one contains
/// filename of a user scene.
std::string default_filename(const std::string& scene_file_path, const Config& config);

/// Prints a full help.
/// Full help contains all available options and lists all algorithms and
/// predefined scenes.
void print_help(const char* argv[]);

/// Prints a short help.
/// Short help lists only basic options, scenes and algorithms.
void print_short_help(const char* argv[]);

/// Parses command line and sets up the Config according to it.
/// Returns -1 on failure, 0 on success and rendering should happen,
/// 1 on success and no rendering should happen (help is printed).
int32_t parse_commandline(int argc, const char* argv[], Config& config);

/// Configurations of the predefined scenes.
extern std::vector<SceneConfig> g_scene_configs;

#endif // MISC_CONFIG_HPP
