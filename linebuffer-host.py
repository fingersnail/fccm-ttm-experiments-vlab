import os
import sys


emul = 0

if emul:
  I,J,II,JJ,KX,KY,SI,SJ=128,128,32,32,3,3,1,1

else:
  I,J,II,JJ,KX,KY,SI,SJ=128,128,32,32,3,3,1,1

AOCX_FILE = "linebuffer.aocx"

command = 'g++ -g -O0 -D__USE_XOPEN2K8 -Wall -IALSDK/include -I/usr/local/include -DHAVE_CONFIG_H -DTESTB -g -LALSDK/lib  -L/usr/local/lib  -fPIC -I../common/inc -I../extlibs/inc -I/export/quartus_pro/17.1.1/hld/host/include  linebuffer-host.cpp -L/export/quartus_pro/17.1.1/hld/linux64/lib -L/export/fpga/release/a10_gx_pac_ias_1_1_pv/opencl/opencl_bsp/linux64/lib  -L/export/quartus_pro/17.1.1/hld/host/linux64/lib -Wl,--no-as-needed -lalteracl  -lintel_opae_mmd -lrt -lelf -L/opt/aalsdk/sdk502/lib -Wl,-rpath=/opt/aalsdk/sdk502/lib -DI={} -DJ={} -DKX={} -DKY={} -DII={} -DJJ={} -DSI={} -DSJ={} -DAOCX_FILE=\\"{}\\"'.format(I,J,KX,KY,II,JJ,SI,SJ,AOCX_FILE) + " -o host-emul 2>&1 | tee -a host.log"

print command

# print "num ops:"
# print 2*I*II*III*J*JJ*JJJ*K*KK*KKK*L*LL*LLL

os.system(command)
