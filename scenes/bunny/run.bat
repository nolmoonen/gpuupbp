@SET MODE=-compatible
@SET ALG=upbp_bpt
@SET NAME=bunny_compatible_bpt.bmp
call do_not_run.bat

@SET MODE=-previous
@SET ALG=upbp_pp3d
@SET NAME=bunny_previous_pp3d.bmp
call do_not_run.bat

@SET MODE=-previous
@SET ALG=upbp_pb2d
@SET NAME=bunny_previous_pb2d.bmp
call do_not_run.bat

@SET MODE=-previous
@SET ALG=upbp_bb1d
@SET NAME=bunny_previous_bb1d.bmp
call do_not_run.bat

@SET MODE=-compatible
@SET ALG=upbp_bb1d+pb2d+pp3d+bpt
@SET NAME=bunny_compatible_all.bmp
call do_not_run.bat

PAUSE