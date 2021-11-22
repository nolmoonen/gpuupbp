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

#include "misc/config.hpp"
#include "misc/logger.hpp"
#include "misc/scene_loader.hpp"
#include "renderer/renderer.hpp"

/// Output image if continuous outputting is enabled.
void continuous_output(const Config& config,
                       int iter,
                       Framebuffer& accum_frame_buffer,
                       Framebuffer& output_frame_buffer,
                       Renderer& renderer,
                       const std::string& name,
                       const std::string& ext)
{
#define FILENAME_BUFFER_SIZE 1024
    char filename[FILENAME_BUFFER_SIZE];
#undef FILENAME_BUFFER_SIZE
    if (config.continuous_output > 0) {
        if (iter % config.continuous_output == 0) {
            clear(output_frame_buffer);
            add_scaled(output_frame_buffer, accum_frame_buffer, 1.f / static_cast<float>(iter));

            sprintf(filename, "%s-%d.%s", name.c_str(), iter, ext.c_str());

            // save the image
            save(output_frame_buffer, filename);
        }
    }
}

int main(int argc, const char* argv[])
{
    // read in configuration based on the command line arguments
    Config config{};
    // exit if parsing error or not rendering should happen
    if (parse_commandline(argc, argv, config)) return 0;

    // load scene based on configuration
    SceneLoader scene_loader{};
    scene_loader.load_scene(config);

    // prints what we are doing
    printf("Scene:    %s\n", config.scene_obj_file.c_str());
    if (config.max_time > 0) {
        printf("Target:   %g seconds render time\n", config.max_time);
    } else {
        printf("Target:   %d iteration(s)\n", config.iteration_count);
    }
    std::string desc = get_description(config);
    printf("Running:  %s", desc.c_str());

    // buffer that contains all contributions
    Framebuffer accum_frame_buffer{};
    init(accum_frame_buffer, config.resolution);
    // buffer that is used to stage the output image, which can be continuous
    Framebuffer output_frame_buffer{};
    init(output_frame_buffer, config.resolution);

    Renderer renderer{};
    if (renderer.init(&scene_loader, &config, &accum_frame_buffer)) return 0;

    clock_t start_t = clock();
    int32_t iter = 0; // number of executed iterations

    // todo this assumes extension is three characters
    // the name of the file
    std::string name = config.output_name.substr(0, config.output_name.length() - 4);
    // the extension of the file
    std::string ext = config.output_name.substr(config.output_name.length() - 3, 3);

    // rendering loop:
    // when we have any time limit, use time-based loop,
    // otherwise go with required iterations

    if (config.max_time > 0) {
        // time based loop
        auto end_t = start_t + static_cast<clock_t>(config.max_time * CLOCKS_PER_SEC);
        while (clock() < end_t) {
            renderer.run_iteration(iter);
            iter++; // counts number of iterations
            continuous_output(config, iter, accum_frame_buffer, output_frame_buffer, renderer, name, ext);
        }
    } else {
        // iterations based loop
        int32_t cnt = 0;
        int32_t percent = -1;
        for (iter = 0; iter < config.iteration_count; iter++) {
            renderer.run_iteration(iter);
            ++cnt;
            auto curr_percent = static_cast<int32_t>((static_cast<float>(cnt) / config.iteration_count) * 100.f);
            if (curr_percent != percent) {
                percent = curr_percent;
                std::cout << curr_percent << "%" << std::endl;
            }
            continuous_output(config, cnt, accum_frame_buffer, output_frame_buffer, renderer, name, ext);
        }
        iter = config.iteration_count;
    }

    clock_t end_t = clock();

    // deallocate buffer for continuous output
    cleanup(output_frame_buffer);

    // scale the accumulated color by the number of iterations
    scale(accum_frame_buffer, 1.f / static_cast<float>(iter));

    // clean up renderer
    renderer.cleanup();

    float time = float(end_t - start_t) / CLOCKS_PER_SEC;
    printf("done in %.2f s (%i iterations)\n", time, iter);

    // save the image and deallocate buffer
    save(accum_frame_buffer, config.output_name, 2.2f);
    cleanup(accum_frame_buffer);

    // scene cleanup
    scene_loader.delete_scene_loader();

    return 0;
}
