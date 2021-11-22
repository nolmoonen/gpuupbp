@set NAME_A=%SCENE%_predef_%TECH%
@set NAME_B=%SCENE%_predef_%TECH%-gpu

FOR /L %%Z in (%CONT_OUT%,%CONT_OUT%,%ITER%) DO (
    echo on
    ..\..\cmake-build-release\bin\compare.exe !NAME_A!-%%Z.bmp !NAME_B!-%%Z.bmp 0
    echo off
)
@REM empty line
echo(