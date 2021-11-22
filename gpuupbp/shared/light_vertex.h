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

#ifndef SHARED_LIGHT_VERTEX_H
#define SHARED_LIGHT_VERTEX_H

#include "bsdf.h"
#include "frame.h"

#include <cuda_runtime.h>

struct Ray {
    float3 origin;
    float3 direction;

    SHARED_INLINE SHARED_HOSTDEVICE Ray(){};

    SHARED_INLINE SHARED_HOSTDEVICE Ray(const float3& origin, const float3& direction)
        : origin(origin), direction(direction)
    {
    }
};

/// MIS data used to compute the MIS weight for vertex i.
struct VertexWeights {
    float d_shared;
    float d_bpt;
    float d_pde;

    /// 1         / pr_{T,i  }
    float ray_sample_for_pdf_inv;
    /// 1         / pl_{T,i-1}
    float ray_sample_rev_pdf_inv;
    /// prob_r{i} / pr_{T,i}
    float ray_sample_for_pdfs_ratio;
    /// prob_l{i} / pl_{T,i}
    float ray_sample_rev_pdfs_ratio;
    /// Whether i-1 is in a medium.
    bool prev_in_medium;
    /// Whether i-1 is specular.
    bool prev_specular;
};

struct StateWeights {
    VertexWeights weights;
    float last_sin_theta;
    /// dBPT = dBPTa * f_i + dBPTb
    float d_bpt_a;
    float d_bpt_b;
    /// dPDE = dPDEa * f_i + dPDEb
    float d_pde_a;
    float d_pde_b;
};

/// Light vertex, used for upbp
struct LightVertex {
    /// Position of the vertex.
    float3 hitpoint;
    /// Path throughput (including emission).
    float3 throughput;
    /// Number of segments between source and vertex.
    unsigned char path_length;
    /// Stores all required local information, including incoming direction.
    BSDF bsdf;
    /// Index of next vertex in path.
    unsigned int next_idx;
    /// Data needed for MIS weights computation.
    VertexWeights weights;
    /// Maintains three flags, see below.
    unsigned char flags;

  private:
    enum LightVertexFlags {
        /// Vertex in medium.
        IN_MEDIUM = (1u << 0),
        /// Vertex in participating medium behind real (not imaginary) surface.
        BEHIND_SURF = (1u << 1)
    };

    /// General flag set.
    __device__ __forceinline__ void set_flag(unsigned char flag) { flags |= flag; }

    /// General flag clear.
    __device__ __forceinline__ void clear_flag(unsigned char flag) { flags &= ~flag; }

  public:
    /** in medium flag */

    __device__ __forceinline__ bool is_in_medium() const { return flags & IN_MEDIUM; }

    __device__ __forceinline__ void set_in_medium(bool val) { val ? set_flag(IN_MEDIUM) : clear_flag(IN_MEDIUM); }

    /** behind surf flag */

    __device__ __forceinline__ bool is_behind_surf() const { return flags & BEHIND_SURF; }

    __device__ __forceinline__ void set_behind_surf(bool val) { val ? set_flag(BEHIND_SURF) : clear_flag(BEHIND_SURF); }
};

struct LightBeam {
    /// Origin and direction of the photon beam.
    Ray ray;
    /// Path throughput (including emission and division by path PDF) at
    /// beam origin.
    float3 throughput_at_origin;
    /// Medium in which beam resides.
    Medium* medium;
    /// Beam length.
    float beam_length;
    /// PDF of sampling through previous media on the generating ray.
    float ray_sample_for_pdf;
    /// Reverse PDF of sampling through previous media on the generating ray.
    float ray_sample_rev_pdf;
    /// Flags for sampling inside the beam.
    unsigned int ray_sampling_flags;
    /// Used to determine whether BB1D should use this beam if BP2D is also
    /// enabled.
    unsigned int path_idx;
    StateWeights weights;
    // todo put below in state_weights?

    /// Number of segments in the path, including this beam.
    unsigned int path_length;
    bool is_infinite_light;
};

#endif // SHARED_LIGHT_VERTEX_H
