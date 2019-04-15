FILE=linebuffer

HOST_MACROS=-DI=16 -DJ=16 -DII=8 -DJJ=8 -DKX=3 -DKY=3 -DSI=1 -DSJ=1

compile-device:
	aoc -march=emulator -g -v -emulator-channel-depth-model=strict $(DEVICE_MACROS) $(FILE).cl -o $(FILE).aocx

compile-host:
	g++ $(FILE)-host.cpp -g $(HOST_MACROS) -DLINUX -DALTERA_CL -fPIC -Iinc -IALSDK/include -I/usr/local/include -DHAVE_CONFIG_H -DTESTB -g -LALSDK/lib  -L/usr/local/lib  -fPIC -I../common/inc -I../extlibs/inc -I/export/quartus_pro/17.1.1/hld/host/include -L/export/quartus_pro/17.1.1/hld/linux64/lib -L/export/fpga/release/a10_gx_pac_ias_1_1_pv/opencl/opencl_bsp/linux64/lib  -L/export/quartus_pro/17.1.1/hld/host/linux64/lib -Wl,--no-as-needed -lalteracl -lintel_opae_mmd -L/opt/aalsdk/sdk502/lib -Wl,--no-as-needed -lalteracl -o host-emul 2>&1 | tee -a host.log

run-emulation:
	CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./host-emul 2>&1 | tee trace.log

clean:
	rm -rf host-emul *.log $(FILE).aoc* $(FILE).*.temp $(FILE)


.PHONY: compile-device compile-host run-emulation clean
