#!/bin/bash
cp fortran_apis/api_sreal.f $1/api_sreal.f
cd $1
echo $(LINKLIBS)
echo $(NLOLIBS)
echo $(APPLLIBS)
echo $(LINKLIBS    )
cat >> makefile <<"EOF" 
api_sreal.so: api_sreal.o handling_lhe_events.o $(PROCESS) makefile $(LIBS)
	ar rcs ../../libmadgraph.a $(shell find . -name "*.o")
	$(FC) $(FFLAGS) -shared -fPIC -o api_sreal.so api_sreal.o $(PROCESS) ../../libmadgraph.a $(LINKLIBS) $(NLOLIBS) $(APPLLIBS) $(LINKLIBS) $(FJLIBS) $(FO_EXTRAPATHS) $(FO_EXTRALIBS) $(LDFLAGS)
EOF
VERBOSE=1 
make api_sreal.so
