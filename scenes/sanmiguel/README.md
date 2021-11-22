# San Miguel

## Credit

Scene acquired
from [McGuire Computer Graphics Archive](https://casual-effects.com/data/).

## License

[Creative Commons CC-BY 3.0](https://creativecommons.org/licenses/by/3.0/).

## Modifications

The following modifications are made, in order:

* Import `san-miguel.obj` in Blender 2.90.1.
* Remove all faces of the light bulbs of the single-light chandeliers that point
  inwards or upwards.
* Remove the plant and chair in view of the camera, as well as the large
  structural parts only visible through the two corridors.
* Export as `sanmiguel.obj`, which also creates `sanmiguel.mtl`, using the
  following settings:
  ```
  Include
  Limit to   [ ] Selection Only
  Objects as [x] OBJ Objects
             [ ] OBJ Groups 
             [ ] Material Groups
    
             [ ] Animation
    
  Transform
  Scale      100.0
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
* Replace all `Ks` constants by the average of the `map_Ks` image, if present.
* Set `Ke` of `material_83` in `sanmiguel.mtl` to `0 0 0`.
* Change `material_041` in `sanmiguel.mtl` to:
  ```
  newmtl material_041
  Kd 0 0 0
  Ns 18.5613
  ```
  which is the glass medium.
* Change `material_035` in `sanmiguel.mtl` to:
  ```
  newmtl material_035
  Kd 0 0 0
  Ns 2.02139
  ```
  which is the wax medium.
* Change `materialo` in `sanmiguel.mtl` to:
  ```
  newmtl materialo
  Kd 0 0 0
  Ns 18.5613
  ```
  which is the fountain water.
* Change `material_033` in `sanmiguel.mtl` to:
  ```
  newmtl material_033
  Kd 0 0 0
  Ns 18.5613
  ```
  which is the ashtray.
* Change `materialn` in `sanmiguel.mtl` to:
  ```
  newmtl materialn
  Kd 0 0 0
  Ns 18.5613
  ```
  which is the lantern glass.
* Change `material_042` in `sanmiguel.mtl` to:
  ```
  newmtl material_042
  Kd 0 0 0
  Ns 18.5613
  ```
  which is the salt shaker container.
* Change `Ks` of `material_0` to `0 0 0`, which is the glass in the doors.

## Notes

* Without the modification, the light bulb glass has a volume and thus also have
  a surface inside the bulb that emits light rays, which never leave the bulb
  itself.