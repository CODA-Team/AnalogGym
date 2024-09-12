#!/bin/bash
./clear.sh
hspice64 -i tran_27corner.sp -o chgp >chgp.info
# $DE_PY_HOME/Dependence/glibc/lib/ld-2.22.so --library-path $DE_PY_HOME/Dependence/glibc/lib:$LD_LIBRARY_PATH `which python` -u measure.py 27 de 1>run_sim.info
python -u measure.py 27 de 1>run_sim.info
