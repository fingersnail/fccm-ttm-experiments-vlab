import os
import sys


emul = 0

if emul:
  BATCH,NOX,NOY,NIF,NOF,POX,POY,POF,KX,KY,SX,SY=16,128,128,16,64,8,8,8,3,3,1,1
else:
  BATCH,NOX,NOY,NIF,NOF,POX,POY,POF,KX,KY,SX,SY=16,128,128,16,64,8,8,8,3,3,1,1

AOCX_FILE = "conv{}-{}.aocx".format(NIF,NOF)

command = 'g++ -g -O0 -D__USE_XOPEN2K8 -Wall -IALSDK/include -I/usr/local/include -DHAVE_CONFIG_H -DTESTB -g -LALSDK/lib  -L/usr/local/lib  -fPIC -I../common/inc -I../extlibs/inc -I/export/quartus_pro/17.1.1/hld/host/include  conv3-64-host.cpp -L/export/quartus_pro/17.1.1/hld/linux64/lib -L/export/fpga/release/a10_gx_pac_ias_1_1_pv/opencl/opencl_bsp/linux64/lib  -L/export/quartus_pro/17.1.1/hld/host/linux64/lib -Wl,--no-as-needed -lalteracl  -lintel_opae_mmd -lrt -lelf -L/opt/aalsdk/sdk502/lib -Wl,-rpath=/opt/aalsdk/sdk502/lib -DBATCH={} -DNOX={} -DNOY={} -DNIF={} -DNOF={} -DPOX={} -DPOY={} -DPOF={} -DKX={} -DKY={} -DSX={} -DSY={} -DAOCX_FILE=\\"{}\\"'.format(BATCH,NOX,NOY,NIF,NOF,POX,POY,POF,KX,KY,SX,SY,AOCX_FILE) + " -o conv 2>&1 | tee -a host.log"

print command

os.system(command)
