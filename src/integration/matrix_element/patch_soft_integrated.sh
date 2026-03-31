#!/bin/bash
cp fortran_apis/api_soft_integrated.f $1/api_soft_integrated.f
cd $1
cat >> makefile <<"EOF" 
api_soft_integrated.so: api_soft_integrated.o $(PROCESS) makefile $(LIBS)
	ar rcs ../../libmadgraph.a $(shell find . -name "*.o")
	$(FC) $(FFLAGS) -shared -fPIC -o api_soft_integrated.so api_soft_integrated.o $(PROCESS) ../../libmadgraph.a $(LINKLIBS) $(NLOLIBS) $(APPLLIBS) $(LINKLIBS) $(FJLIBS) $(FO_EXTRAPATHS) $(FO_EXTRALIBS) $(LDFLAGS)
EOF
VERBOSE=1 make api_soft_integrated.so
