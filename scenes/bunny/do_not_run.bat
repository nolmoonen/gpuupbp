"..\..\cmake-build-release\bin\gpuupbp.exe" ^
-s -1 "..\..\scenes\bunny\bunny.obj" ^
-a %ALG% ^
-l 12 ^
-t 1800 ^
-r 400x800 ^
-pbc 4000 ^
%MODE% ^
-r_initial_pp3d -0.0001 ^
-r_alpha_pp3d 1 ^
-r_initial_pb2d -0.0001 ^
-r_alpha_pb2d 1 ^
-r_initial_surf -0.0001 ^
-r_alpha_surf 0.75 ^
-r_initial_bb1d -0.0001 ^
-r_alpha_bb1d 1 ^
-ignorespec 1 ^
-o %NAME%