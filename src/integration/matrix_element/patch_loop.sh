#!/bin/bash
cp fortran_apis/loop_api.f $1/api.f
cd $1
cat >> makefile <<"EOF"

api.so: api.o $(PROCESS) makefile $(LIBS)
	$(FC) $(FFLAGS) -shared -fPIC -o api.so api.o $(PROCESS) $(LINKLIBS)
EOF
make api.so
