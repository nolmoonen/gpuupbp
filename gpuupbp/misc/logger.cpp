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

#include "logger.hpp"

#include <fstream>
#include <iomanip>

void Logger::log_times(const std::string& scene_name,
                       uint32_t iteration,
                       double light_trace,
                       double build,
                       double camera_trace,
                       uint32_t res_x,
                       uint32_t res_y,
                       double mem_used,
                       double mem_free,
                       double mem_total,
                       float light_util,
                       float camera_util)
{
    if (!printed_header) print_header();
    std::ofstream file(file_name, std::ios_base::app);
    file << std::setw(40) << scene_name << "," << std::setw(6) << iteration << "," << std::setw(12) << light_trace
         << "," << std::setw(12) << build << "," << std::setw(12) << camera_trace << "," << std::setw(12) << res_x
         << "," << std::setw(12) << res_y << "," << std::setw(12) << mem_used << "," << std::setw(12) << mem_free << ","
         << std::setw(12) << mem_total << "," << std::setw(12) << light_util << "," << std::setw(12) << camera_util
         << std::endl;
}

Logger::Logger()
{
    // create filename based on timestamp
    time_t rawtime;
    tm* time_info;
    char buffer[80];
    time(&rawtime);

    time_info = localtime(&rawtime);

    strftime(buffer, sizeof(buffer), "%Y-%m-%d-%H-%M-%S", time_info);
    std::string time(buffer);
    file_name = time;
    file_name.append("_log.txt");
}

void Logger::print_header()
{
    printed_header = true;
    std::ofstream file(file_name, std::ios_base::out);
    file << std::setw(40) << "scene"
         << "," << std::setw(6) << "iter"
         << "," << std::setw(12) << "lighttime"
         << "," << std::setw(12) << "buildtime"
         << "," << std::setw(12) << "cameratime"
         << "," << std::setw(12) << "res_x"
         << "," << std::setw(12) << "res_y"
         << "," << std::setw(12) << "mem_used"
         << "," << std::setw(12) << "mem_free"
         << "," << std::setw(12) << "mem_total"
         << "," << std::setw(12) << "light_util"
         << "," << std::setw(12) << "camera_util" << std::endl;
}
