# Sun Temple

## Credit

Scene acquired
from [Open Research Content Archive (ORCA)](https://developer.nvidia.com/ue4-sun-temple)
. Acquired EXR environment map from [Poly Haven](https://polyhaven.com/).

## License

[Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
.

## Modifications

The following modifications are made, in order:

* Import `SunTemple.fbx` in Blender 2.90.1.
* Remove all rocks and trees, as well as the separating wall in the corridor.
* Export as `suntemple.obj`, which also creates `suntemple.mtl`, using the
  following settings:
  ```
  Include
  Limit to   [ ] Selection Only
  Objects as [x] OBJ Objects
             [ ] OBJ Groups 
             [ ] Material Groups
    
             [ ] Animation
    
  Transform
  Scale      100.00
  Path Mode  Auto
  Forward    Y Forward
  Up         Z Up
    
  Geometry
             [x] Apply Modifiers
             [ ] Smooth Groups
             [ ] Bitflag Smooth Groups
             [x] Write Normals
             [x] Include UVs
             [x] Write Materials
             [x] Triangulate Faces
             [ ] Curves as NURBS
             [ ] Polygroups
             [ ] Keep Vertex Order
  ```
* Replace all `Kd` constants by the average of the `map_Kd` image, if present.
* Replace all `Ks` constants by the average of the blue channel of the `map_Ks`
  image, if present. The blue channel represents the "metalness" constant.
* If the `Ks` is `1 1 1` and the `Kd` is not `0 0 0`, replace the `Ks` by
  the `Kd` and the `Kd` by `0 0 0`.
* Change `M_Statue_Inst` in `suntemple.mtl` to:
  ```
  newmtl M_Statue_Inst
  Kd 0 0 0
  Ns 2.02139
  ```
  which is the statue medium.
* Append to `suntemple.mtl`:
  ```
  newmtl mat_fog
  ```