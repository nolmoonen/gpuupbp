# Scenes

Each scene is defined by three files (please see Petr Vévoda's thesis for more
information about them):

* OBJ: A file with geometry of the scene in the Wavefront OBJ file format.
* MTL: A file with materials of the scene geometry in the MTL file format.
* AUX: A file with definition of the camera, media, additional material
  properties and lights. It is a manually created ASCII file in our own format.

Each scene contains a `run.bat` which is used to render the scenes in Nol
Moonen's thesis. These batch files expect `gpuupbp.exe`
at `gpuupbp\cmake-build-release\bin\gpuupbp.exe`.