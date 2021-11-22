![image](doc/bunnyrow.png)

GPUUPBP is an OptiX-based GPU implementation of the [unified, points, beams, and paths](https://cs.dartmouth.edu/wjarosz/publications/krivanek14upbp.html) (UPBP) algorithm for robust light simulation in participating media and is the code for the paper "Efficient Hardware Acceleration of Robust Volumetric Light Transport Simulation". GPUUPBP is based on [SmallUPBP](http://www.smallupbp.com/) and makes two main improvements:

* The algorithm for computing the multiple importance sampling (MIS) weights is replaced with a new algorithm. Instead of iterating over all vertices of each full path, this new algorithm computes the path weight in constant time by only accessing the vertex data of the subpath ends. This is achieved by formulating the subpath weights in a recursive manner similar to how is done by [SmallVCM](http://www.smallvcm.com/) and as described in its accompanying [technical paper](https://www.iliyan.com/publications/ImplementingVCM).
* The photon map for photon density estimation of the volumetric B-B1D, P-B2D, and P-P3D estimators, as well as for surface photon mapping are implemented using the RTX bounding volume hierarchy. The entire algorithm is implemented on the GPU with OptiX and the scene intersection routine is accelerated using RTX.

If you find this code useful in your research, please consider citing:

```
@article{moonen-jalba-2023,
  author   = {Moonen, Nol and Jalba, Andrei},
  title    = {Efficient Hardware Acceleration of Robust Volumetric Light Transport Simulation},
  journal  = {Computer Graphics Forum},
  volume   = {},
  number   = {},
  pages    = {},
  keywords = {},
  doi      = {},
  url      = {},
  abstract = {Efficiently simulating the full range of light effects in arbitrary input scenes that contain participating media is a difficult task. Unified points, beams and paths (UPBP) is an algorithm capable of capturing a wide range of media effects, by combining bidirectional path tracing (BPT) and photon density estimation (PDE) with multiple importance sampling (MIS). A computationally expensive task of UPBP is the MIS weight computation, performed each time a light path is formed. We derive an efficient algorithm to compute the MIS weights for UPBP, which improves over previous work by eliminating the need to iterate over the path vertices. We achieve this by maintaining recursive quantities as subpaths are generated, from which the subpath weights can be computed. In this way, the full path weight can be computed by only using the data cached at the two vertices at the ends of the subpaths. Furthermore, a costly part of PDE is the search for nearby photon points and beams. Previous work has shown that a spatial data structure for photon mapping can be implemented using the hardware-accelerated bounding volume hierarchy of NVIDIA’s RTX GPUs. We show that the same technique can be applied to different types of volumetric PDE and compare the performance of these data structures with the state of the art. Finally, using our new algorithm and data structures we fully implement UPBP on the GPU which we, to the best of our knowledge, are the first to do so.},
  year     = {2023}
}
```

## System requirements and dependencies

GPUUPBP requires CMake 3.17 or later, NVIDIA OptiX 7.6 and CUDA 11.8. [tinyexr](https://github.com/syoyo/tinyexr) is required for reading and writing EXR files and a version is supplied in `deps/tinyexr`. An OptiX-capable
GPU is required for running the project. Optionally, Python is required for running the benchmark script.

## License

In continuation of SmallUPBP, the code of GPUUPBP is released under the [MIT license](http://en.wikipedia.org/wiki/MIT_License). tinyexr is licensed with a [modified BSD license](http://www.openexr.com/license.html). The licenses of the scenes can be found in their respective directories.

## Using

Run `gpuupbp -h` for a short command line parameter description
and `gpuupbp -hf` for a long one.

After building, run `benchmark/benchmark.py` to run 60 iterations of each scene
used in the thesis and report the average time per iteration. The scene `.obj`
files must be extracted first.

## Building

```shell
git clone https://github.com/nolmoonen/gpuupbp.git
cd gpuupbp
cmake -B build -S . -D OptiX_ROOT=<path_to_optix>
cmake --build build
```

Optionally, specify `-D CUDAToolkit_ROOT` if the CUDA toolkit could not be found.

## Acknowledgements

The author would like to thank the authors of SmallUPBP and SmallVCM for making
their code publicly available.

## Additional information

An additional volumetric photon density estimator is implemented, the BP2D estimator, which uses photon beams and camera points. Further details can be found in the author's master's thesis:

> [Hardware Acceleration of Robust Light Transport Simulation in Participating Media  
> N.A.H. Moonen, Master's thesis, Eindhoven University of Technology, November 2021](https://research.tue.nl/en/studentTheses/hardware-acceleration-of-robust-light-transport-simulation-in-par)

The code contains references to Equations in the thesis, indicated by `Eq x.x`. For a full list of changes between SmallUPBP and GPUUPBP, see [`doc/changes`](doc/changes.md). For a more complete structure of the project see [`doc/structure`](doc/structure.md).
