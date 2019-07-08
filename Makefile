FILE=conv3-64

CONV_MACROS=-DFLOAT_VEC=float8 -DBATCH=1 -DNOX=64 -DNOY=64 -DNIF=32 -DNOF=64 -DPIF=4 -DPOX=2 -DPOY=2 -DPOF=2 -DKX=5 -DKY=5 -DKIF=3 -DSX=1 -DSY=1 -DSIF=3

compile-device:
	aoc -march=emulator -g -v $(CONV_MACROS) -emulator-channel-depth-model=strict $(FILE).cl -o $(FILE).aocx

compile-host:
	g++ $(FILE)-host.cpp -g $(CONV_MACROS) -DLINUX -DALTERA_CL -fPIC -Iinc -IALSDK/include -I/usr/local/include -DHAVE_CONFIG_H -DTESTB -g -LALSDK/lib  -L/usr/local/lib  -fPIC -I../common/inc -I../extlibs/inc -I/export/quartus_pro/17.1.1/hld/host/include -L/export/quartus_pro/17.1.1/hld/linux64/lib -L/export/fpga/release/a10_gx_pac_ias_1_1_pv/opencl/opencl_bsp/linux64/lib  -L/export/quartus_pro/17.1.1/hld/host/linux64/lib -Wl,--no-as-needed -lalteracl -lintel_opae_mmd -L/opt/aalsdk/sdk502/lib -Wl,--no-as-needed -lalteracl -o host-emul 2>&1 | tee -a host.log

run-emulation:
	CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./host-emul 2>&1 | tee trace.log

clean:
	rm -rf host-emul *.log $(FILE).aoc* $(FILE).*.temp $(FILE) *.o*


.PHONY: compile-device compile-host run-emulation clean
