import os
import sys


emul = 0

# Note: II has to be 1 else the compute buffer cannot be implemented as rotating register

if emul:
  I,J,K,L = 1,1,1,2
  II, JJ, KK, LL = 1,2,2,2
  III, JJJ, KKK, LLL = 2,2,2,2

else:
  I,J,K,L = 8,4,4,16
  II, JJ, KK, LL = 1,8,11,4
  III, JJJ, KKK, LLL = 8,8,16,16

AOCX_FILE = "ttm-drain_{}x{}.aocx".format(JJ,KK)

command = 'g++ -g -O0 -D__USE_XOPEN2K8 -Wall -IALSDK/include -I/usr/local/include -DHAVE_CONFIG_H -DTESTB -g -LALSDK/lib  -L/usr/local/lib  -fPIC -I../common/inc -I../extlibs/inc -I/export/quartus_pro/17.1.1/hld/host/include  ttm-host.cpp -L/export/quartus_pro/17.1.1/hld/linux64/lib -L/export/fpga/release/a10_gx_pac_ias_1_1_pv/opencl/opencl_bsp/linux64/lib  -L/export/quartus_pro/17.1.1/hld/host/linux64/lib -Wl,--no-as-needed -lalteracl  -lintel_opae_mmd -lrt -lelf -L/opt/aalsdk/sdk502/lib -Wl,-rpath=/opt/aalsdk/sdk502/lib -DI={} -DJ={} -DK={} -DL={} -DII={} -DJJ={} -DKK={} -DLL={} -DIII={} -DJJJ={} -DKKK={} -DLLL={} -DAOCX_FILE=\\"{}\\"'.format(I,J,K,L,II,JJ,KK,LL,III,JJJ,KKK,LLL,AOCX_FILE) + " -o device-ttm 2>&1 | tee -a host.log"

print command

print "num ops:"
print 2*I*II*III*J*JJ*JJJ*K*KK*KKK*L*LL*LLL

os.system(command)
