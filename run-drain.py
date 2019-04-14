import os
import sys

emul = 0

if (emul):
  aoc_opt = "-march=emulator"
else:
  aoc_opt = "-time time.out -time-passes -regtest_mode -v -g -fpc -fp-relaxed -report"

# Note: II has to be 1 else the compute buffer cannot be implemented as rotating register

if (emul):
  I,J,II,JJ,KX,KY,SI,SJ=16,16,8,8,3,3,1,1

else:
  I,J,II,JJ,KX,KY,SI,SJ=16,16,8,8,3,3,1,1

aocx_name = "linebuffer"

command = "qsub-aoc {} {}.cl -o {}.aocx 2>&1 | tee -a run.log".format(aoc_opt, aocx_name, aocx_name)

os.system("echo "+command)
os.system(command)
