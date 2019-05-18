FILE=conv3-64

CONV_MACROS=-DBATCH=4 -DNOX=64 -DNOY=64 -DNIF=3 -DNOF=64 -DPOX=2 -DPOY=2 -DPOF=4 -DKX=3 -DKY=3 -DKIF=3 -DSX=1 -DSY=1 -DSIF=3

compile-device:
	aoc -march=emulator -g -v $(CONV_MACROS) -emulator-channel-depth-model=strict $(FILE).cl -o $(FILE).aocx

compile-host:
	g++ $(FILE)-host.cpp -g $(CONV_MACROS) -DLINUX -DALTERA_CL -fPIC -Iinc  -I/opt/AMDAPPSDK-3.0/include -I$(ALTERAOCLSDKROOT)/host/include -L$(ALTERAOCLSDKROOT)/host/linux64/lib -L$(AOCL_BOARD_PACKAGE_ROOT)/linux64/lib -Wl,--no-as-needed  -lalteracl -l$(AOCL_BOARD_LIB) -lelf -o host-emul 2>&1 | tee -a host.log

run-emulation:
	CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./host-emul 2>&1 | tee trace.log

clean:
	rm -rf host-emul *.log $(FILE).aoc* $(FILE).*.temp $(FILE)


.PHONY: compile-device compile-host run-emulation clean
