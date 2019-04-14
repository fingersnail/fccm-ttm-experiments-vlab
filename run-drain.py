import os
import sys

emul = 0

if (emul):
  aoc_opt = "-march=emulator"
else:
  aoc_opt = "-time time.out -time-passes -regtest_mode -v -g -fpc -fp-relaxed -report"

# Note: II has to be 1 else the compute buffer cannot be implemented as rotating register

if (emul):
  I=16, J=16 ,II=8, JJ=8, KX=3, KY=3, SI=1, SJ=1

else:
  I=16, J=16 ,II=8, JJ=8, KX=3, KY=3, SI=1, SJ=1

aocx_name = "linebuffer"

$(FILE).cl -o $(FILE).aocx
command = "qsub-aoc {} {}.cl -o {}.aocx 2>&1 | tee -a run.log".format(aoc_opt, aocx_name, aocx_name)

os.system("echo "+command)
os.system(command)
