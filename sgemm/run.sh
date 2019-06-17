#!/bin/bash

CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./device-ttm 2>&1 | tee run.log
