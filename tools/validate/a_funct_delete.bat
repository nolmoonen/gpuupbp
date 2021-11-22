@REM delete the continuous output images
FOR /L %%Z in (%CONT_OUT%,%CONT_OUT%,%ITER%) DO (
    del "%SCENE%_predef_%TECH%%GPU%-%%Z.bmp" /q
)
@REM delete the final image
del "%SCENE%_predef_%TECH%%GPU%.bmp" /q