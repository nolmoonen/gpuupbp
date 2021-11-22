@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

for /f "delims== tokens=1,2" %%G in (a_params.txt) do set %%G=%%H

@set GPU=""

FOR %%A in (%SCENES%) DO (
    @set SCENE=%%A

    @set TECH=all
    call a_funct_generate_cpu.bat

    @set TECH=bb1d
    call a_funct_generate_cpu.bat

    @set TECH=bpt
    call a_funct_generate_cpu.bat

    @set TECH=pb2d
    call a_funct_generate_cpu.bat

    @set TECH=pp3d
    call a_funct_generate_cpu.bat

    @set TECH=surf
    call a_funct_generate_cpu.bat
)
PAUSE