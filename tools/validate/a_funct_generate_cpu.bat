"..\..\tools\validate\SmallUPBP.exe" ^
-s %SCENE% ^
-a upbp_%TECH% ^
-i %ITER% ^
-pbc 10000 ^
-o %SCENE%_predef_%TECH%%GPU%.bmp ^
-continuous_output %CONT_OUT% ^
-r 512x512 ^
-th 1