Execute following commands:

source vlab.sh

python run-drain.py

python linebuffer-host.py

Didn't work. The error information is in build.log

Or use Makefile (Worked):
(use aoc -march=emulator instead of qsub-aoc)

make compile-device

make compile-host

make run-emulation
