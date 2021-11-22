The top-level structure of GPUUPBP is the following:

* `deps` contains a single third-party dependency, tinyexr, for IO of EXR files.
* `doc` contains additional documentation.
* `gpuupbp` contains the project's source code. The code for the renderer itself
  is spread over `host`, `device`, and `shared` to avoid mixing device and host
  code. The other code is mostly for parsing command line parameters, loading
  scene files, and converting the scene to a format that OptiX supports. `.hpp`
  files are header files included in host code only. `.cuh` files are header
  files included in device code only. `.h` files are included in both host and
  device code and thus can not contain any host- or device-specific language
  constructs.
    * `host` contains host renderer code files.
    * `kernel` contains device renderer code files. The most important files
      are `trace_light.cu` and `trace_camera.cu` which are responsible for
      generating light and camera subpaths, respectively.
    * `misc` contains several host cod files with helper functionality.
    * `renderer` contains host code files to start rendering on the device.
    * `shared` contains definitions that are included both by the host and the
      device code.
* `scenes` contain a number of scenes defined with OBJ files.
* `sutil` is part of the OptiX sample framework and contains a utility library.
  Most of the sutil library is removed, except for the code for compiling code
  with NVRTC and the vector and matrix structs. The library is changed to be
  static.
* `tools` contains separate code functionality to support the main projects'
  code.
