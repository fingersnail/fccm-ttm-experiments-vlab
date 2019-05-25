import os
import sys

emul = 0 

if (emul):
  aoc_opt = "-march=emulator"
else:
  aoc_opt = "-time time.out -time-passes -regtest_mode -v -g -fpc -fp-relaxed -report"

# Note: II has to be 1 else the compute buffer cannot be implemented as rotating register

if (emul):
  BATCH,NOX,NOY,NIF,NOF,POX,POY,POF,KX,KY,SX,SY=16,128,128,16,64,8,8,8,3,3,1,1
else:
  BATCH,NOX,NOY,NIF,NOF,POX,POY,POF,KX,KY,SX,SY=16,128,128,16,64,8,8,8,3,3,1,1

aocx_name = "conv3-64"

command = "qsub-aoc {} -DBATCH={} -DNOX={} -DNOY={} -DNIF={} -DNOF={} -DPOX={} -DPOY={} -DPOF={} -DKX={} -DKY={} -DSX={} -DSY={} {}.cl -o {}.aocx 2>&1 | tee -a run.log".format(aoc_opt,BATCH,NOX,NOY,NIF,NOF,POX,POY,POF,KX,KY,SX,SY,aocx_name,aocx_name)

os.system("echo "+command)
os.system(command)
