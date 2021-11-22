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

#ifndef MISC_LOGGER_HPP
#define MISC_LOGGER_HPP

#include "../shared/shared_enums.h"

#include <chrono>
#include <string>

// https://stackoverflow.com/questions/1008019/c-singleton-design-pattern
struct Logger {
    static Logger& get_instance()
    {
        static Logger instance;
        return instance;
    }

    void log_times(const std::string& scene_name,
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
                   float camera_util);

    Logger();

    void print_header();

    bool printed_header = false;
    std::string file_name;

    Logger(Logger const&) = delete;

    void operator=(Logger const&) = delete;
};

#endif // MISC_LOGGER_HPP
