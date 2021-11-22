# Bistro

Benchmark scene. Execute `run.bat` to generate the image included in the thesis.

## Credit

Scene acquired
from [Open Research Content Archive (ORCA)](https://developer.nvidia.com/orca/amazon-lumberyard-bistro)
. Acquired (a lower resolution) EXR environment map
from [Poly Haven](https://polyhaven.com/a/san_giuseppe_bridge).

## License

[Creative Commons CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Modifications

The following modifications are made, in order:

* Import `BistroInterior_Wine.fbx` in Blender 2.90.1.
* Unhide `Bistro_Research_Interior_paris_building_01_interior_2140_Mesh.281`.
* Add four square light emitters with material `light` in front of the three
  open windows and the door, all having the same angle.
* Export as `bistro.obj`, which also creates `bistro.mtl`, using the following
  settings:
  ```
  Include
  Limit to   [ ] Selection Only
  Objects as [x] OBJ Objects
             [ ] OBJ Groups 
             [ ] Material Groups
    
             [ ] Animation
    
  Transform
  Scale      350.00
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
* Change `TransparentGlass` in `bistro.mtl` to:
  ```
  newmtl TransparentGlass
  Kd 0 0 0
  Ns 18.5613
  ```
  which is the glass medium.
* Change `TransparentGlassWine` in `bistro.mtl` to:
  ```
  newmtl TransparentGlassWine
  Kd 0 0 0
  Ns 18.5613
  ```
  which is the glass medium.
* Change `Red_Wine` in `bistro.mtl` to:
  ```
  newmtl Red_Wine
  Kd 0 0 0
  Ns 18.5613
  ```
  which is the red wine.
* Change `White_Wine` in `bistro.mtl` to:
  ```
  newmtl White_Wine
  Kd 0 0 0
  Ns 18.5613
  ```
  which is the white wine.
* Change `Water` in `bistro.mtl` to:
  ```
  newmtl Water
  Kd 0 0 0
  Ns 18.5613
  ```
  which is the water.
* Change `Ice` in `bistro.mtl` to:
  ```
  newmtl Ice
  Kd 0 0 0
  Ns 18.5613
  ```
  which is the ice.
* Change `Beer` in `bistro.mtl` to:
  ```
  newmtl Beer
  Kd 0 0 0
  Ns 18.5613
  ```
  which is the beer.
* In terms of lights: set the `Ke` of `MASTER_Interior_01_Paris_Lantern1`
  , `Paris_CeilingFan`, `Paris_Ceiling_Lamp`, and `Paris_Wall_Light_Interior`
  to `0 0 0`, as these should have been controlled by a `map_Ke` but we cannot
  load textures. Set the `Ke` of `Lightbulb` to `0 0 0`, to prevent half of all
  light samples going to the small light.

## Notes

* The glasses have volume, but do in places intersect with the table.
* The windows have no glass in them.