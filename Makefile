FILE=linebuffer

HOST_MACROS=-DI=16 -DJ=16 -DII=8 -DJJ=8 -DKX=3 -DKY=3 -DSI=1 -DSJ=1

compile-device:
	aoc -march=emulator -g -v -emulator-channel-depth-model=strict $(DEVICE_MACROS) $(FILE).cl -o $(FILE).aocx

compile-host:
	g++ $(FILE)-host.cpp -g $(HOST_MACROS) -DLINUX -DALTERA_CL -fPIC -Iinc  -I/opt/AMDAPPSDK-3.0/include -I$(ALTERAOCLSDKROOT)/host/include -L$(ALTERAOCLSDKROOT)/host/linux64/lib -L$(AOCL_BOARD_PACKAGE_ROOT)/linux64/lib -Wl,--no-as-needed  -lalteracl -l$(AOCL_BOARD_LIB) -lelf -o host-emul 2>&1 | tee -a host.log

run-emulation:
	CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./host-emul 2>&1 | tee trace.log

clean:
	rm -rf host-emul *.log $(FILE).aoc* $(FILE).*.temp $(FILE)


.PHONY: compile-device compile-host run-emulation clean
