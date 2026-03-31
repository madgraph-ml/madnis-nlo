#!/bin/bash
cp fortran_apis/api_collinear_integrated.f $1/api_collinear_integrated.f
cd $1
cat >> makefile <<"EOF" 
api_collinear_integrated.so: api_collinear_integrated.o $(PROCESS) makefile $(LIBS)
	ar rcs ../../libmadgraph.a $(shell find . -name "*.o")
	$(FC) $(FFLAGS) -shared -fPIC -o api_collinear_integrated.so api_collinear_integrated.o $(PROCESS) ../../libmadgraph.a $(LINKLIBS) $(NLOLIBS) $(APPLLIBS) $(LINKLIBS) $(FJLIBS) $(FO_EXTRAPATHS) $(FO_EXTRALIBS) $(LDFLAGS)
EOF
VERBOSE=1 make api_collinear_integrated.so
