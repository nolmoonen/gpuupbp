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

#ifndef SHARED_SHARED_ENUMS_H
#define SHARED_SHARED_ENUMS_H

enum BeamType { SHORT_BEAM = (1u << 0), LONG_BEAM = (1u << 1) };

///	Estimators available to combine in the UPBP renderer.
enum EstimatorTechnique {
    /// Bidirectional path tracer.
    BPT = (1u << 2),
    /// Surface photon mapping.
    SURF = (1u << 3),
    /// Medium photon mapping (point vs points).
    PP3D = (1u << 4),
    /// BRE (beam vs points).
    PB2D = (1u << 5),
    /// Photon beams (beam vs beams).
    BB1D = (1u << 6),
    /// Photon beams vs camera points.
    BP2D = (1u << 7)
};

/// Misc flags.
enum OtherSettings {
    /// Run all used estimators in a previous mode.
    PREVIOUS = (1u << 9),
    /// Run all used estimators in a compatible mode.
    COMPATIBLE = (1u << 10),
    /// Render only specular paths.
    SPECULAR_ONLY = (1u << 12),
};

#endif // SHARED_SHARED_ENUMS_H
