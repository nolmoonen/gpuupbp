# Changes with respect to SmallUPBP

This file list a complete overview of functionality changes of GPUUPBP compared
to SmallUPBP.

## Removed command line options

* Removed the option to set number of threads and OpenMP support (`-th`).
  GPUUPBP always operates on a single host thread.
* Removed the option to set the maximum amount of memory per thread.
* Removed the option to set the grid resolution (`-gridres`), maximum beams per
  grid cell (`-gridmax`), and grid reduction type (`-gridred`) for BB1D grid
  data structure. This data structure is removed completely and replaced by an
  RTX-BVH-based implementation.
* Removed the option to store beams only in some media depending on the mean
  free path (`-beamstore`).
* Removed the option to set environment map from the command line (`-em`).
* Removed the option to use k nearest neighbors for photon density
  estimation (`-r_initial_pb2d_knn` and `-r_initial_bb1d_knn`).
* Removed the option to set a seed for random number generation (`-seed`). This
  is replaced by a different mechanism, see below for details.
* Removed the option to append duration of rendering to the filename (`-time`).
* Removed the option to configure the minimum distance from camera to
  medium (`-min_dist2med`).
* Removed the option to not evaluate the sine in BB1D MIS weights (`-nosin`).
* Removed the option to only evaluate the BB1D estimator using previous
  mode (`-previous_bb1d`).
* Removed the option to set the number of reference paths per
  iteration (`-rpcpi`). The number of reference paths is now always equal to the
  number of light paths.

## Added command line options

* Added the option to enable GPU assertions. If enabled, an error will be
  printed and the causing thread terminates (`-assert`).
* Added the option to offset the random number generation by a specified number
  of iterations (`-ioff`).
* Added the option to generate a log file (`-log`).

## Renderers

All subsets of the UPBP renderer are removed (`-a` options `upbp_lt`, `upbp_ptd`
, `upbp_ptls`, `upbp_ptmis`, `upbp_bpt`,`upbp_ppm`,`upbp_bpm`, and`upbp_vcm`).
Manual subsets of the techniques used in UPBP can still be run by
using `-a upbp<tech>[+<tech>]*` where `<tech>` is one
of `{bpt,bb1d,bp2d,pb2d,pp3d,surf}`. The following renderers are removed:

* Removed `EyeLight` (`-a` option `el`).
* Removed `PathTracer` (`-a` option `pt`).
* Removed `VertexCM` (`-a` options `ppm`, `bpm`, `lt`, `bpt`, and `vcm`).
* Removed `VolPathTracer` (`-a` options `vptd`, `vpts`, `vptls`, `vptmis`, `vlt`
  , `pb2d`, and `bb1d`)
* Removed `VolBidirPT` (`-a` options `vbpt_lt`, `vbpt_ptd`, `vbpt_ptls`
  , `vbpt_ptmis`, and `vbpt`)

## Behavior

Errors are handled more graciously:

* If the static data structure size is exceeded, the kernel thread terminates
  without making any further modifications.
* If the memory for light vertex or light beam storage is exceeded, the program
  continues as usual with the vertices and beams that have been stored.

## Miscellaneous

* Removed `BeamDensity`, which aggregated statistics for the BB1D grid data
  structure (`-beamdens`).
* Removed `DebugImages`, which aggregated debug information (`-debugimg_option`
  , `-debugimg_multbyweight`, and `-debugimg_output_weights`).
* Removed printing average camera tracing time.
* No longer add an acronym for predefined scenes to the output filename if not
  output filename is specified.
* Removed seeded randomness, replaced by device RNG that can be offset.
* No longer append number of iterations to filename.