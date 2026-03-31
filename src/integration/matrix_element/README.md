## Matrix element API

### Run everything automatically
The following example shows how to set up the Python API for a process, at NLO. The automnatic way to do this, especially if one already has a mg5amcnlo installation, is to run the automatic script that generates the desired ee->3j and ee->4j processes and patches the APIs. This is done via:

```bash
python src/integration/matrix_element/generate_and_patch.py --mg5-dir /path/to/mg5amcnlo
```

If this does not work, or if you don't have mg5amcnlo installed, you can follow the instructions below to do everything manually.

### Install MadGraph5_aMC@NLO at a specific commit.
Clone the madgraph repository ONLY if you don't have it already, and checkout the specific commit. This is important to ensure that the API is compatible with the code in this repository. Define after that the path to parx.
```bash
git clone https://github.com/mg5amcnlo/mg5amcnlo && cd mg5amcnlo && git checkout fb1c2f2d067f63cebe6799f582f81cb89157e291
```
Define the path to parx:
```bash
export PARX_PATH=/path/to/parx
```

### Run everything automatically
#### Born matrix element

First generate a MadGraph standalone folder (while you are in this folder):
```madgraph
generate e+ e- > u u~ g
output standalone /path/to/parx/src/matrix_element/process_api_storage/ee_3j/born
launch
```

Before patching, set the following in `param_card.dat`:
```
3 0.11901442560187821 # aS
```
and in `run_card.dat`:
```
  1.0  = jetalgo   ! FastJet jet algorithm (1=kT, 0=C/A, -1=anti-kT)
  0.4  = jetradius ! The radius parameter for the jet algorithm
 20.0  = ptj       ! Min jet transverse momentum
 5.0  = etaj      ! Max jet abs(pseudo-rap) (a value .lt.0 means no cut)
```

Then run **from within `matrix_element/`**
```bash
cd /path/to/parx/src/matrix_element
./patch_born.sh process_api_storage/ee_3j/born/SubProcesses/P1_epem_uuxg
```

Now you can use the `BornMatrixElement` class from `matrix_element.py` to call it.
```python
api = BornMatrixElement("process_api_storage/ee_3j/born/SubProcesses/P1_epem_uuxg/api.so")
matrix_element = api(momenta)
```

#### Loop matrix element

First generate a MadLoop standalone folder (while you are in this folder):
```madgraph
generate e+ e- > u u~ g [virt=QCD]
output /path/to/parx/src/matrix_element/process_api_storage/ee_3j/loop
launch
```

Before patching, set the following in `param_card.dat`:
```
3 0.11901442560187821 # aS
```
and in `run_card.dat`:
```
  1.0  = jetalgo   ! FastJet jet algorithm (1=kT, 0=C/A, -1=anti-kT)
  0.4  = jetradius ! The radius parameter for the jet algorithm
 20.0  = ptj       ! Min jet transverse momentum
 5.0  = etaj      ! Max jet abs(pseudo-rap) (a value .lt.0 means no cut)
```

Then run
```bash
cd /path/to/parx/src/matrix_element/
./patch_loop.sh process_api_storage/ee_3j/loop/SubProcesses/P0_epem_uuxg
```

Now you can use the `LoopMatrixElement` class from `matrix_element.py` to call it.
```python
api = LoopMatrixElement("process_api_storage/ee_3j/loop/SubProcesses/P0_epem_uuxg/api.so")
matrix_element = api(momenta)
```

#### Soft integrated counterterm

There is some difference with respect to the standard standalone. In this case you need to 'run' a full Madgraph integration, or at the very least compile all the different .f files from MadGraph. This means in practice
```madgraph
generate e+ e- > j j j [QCD] 
output /path/to/parx/src/matrix_element/process_api_storage/ee_3j/counterterms
launch
```
The compilation takes the same amount of time; if one wants to wait for the run to finish it is better to have some mock parameters which would give a useless result in practice, as a modifiable parameter from *terminal*:
```shell
fixed_order = ON 
```
and from the *runcard*:
```madgraph
 -1   = req_acc_FO       ! Required accuracy (-1=ignored, and use the  
                           ! number of points and iter. below)
# These numbers are ignored except if req_acc_FO is equal to -1
 1   = npoints_FO_grid  ! number of points to setup grids
 1      = niters_FO_grid   ! number of iter. to setup grids
 1  = npoints_FO       ! number of points to compute Xsec
 1      = niters_FO        ! number of iter. to compute Xsec
```
An additional piece of information is that the computations to be compared with are performed with fixed renormalization and factorization scales, so also in the *runcard*:
```madgraph
 True    = fixed_ren_scale  ! if .true. use fixed ren scale
 True    = fixed_fac_scale  ! if .true. use fixed fac scale
 ```
**DISCLAIMER**! The MadGraph run is going to fail, since not enough points are thrown. Nonetheless the important files have been compiled.

Before patching, set the following in `param_card.dat` (note the difference with respect to born and loop):
```
3 0.119 # aS
```
and in `run_card.dat`:
```
  1.0  = jetalgo   ! FastJet jet algorithm (1=kT, 0=C/A, -1=anti-kT)
  0.4  = jetradius ! The radius parameter for the jet algorithm
 20.0  = ptj       ! Min jet transverse momentum
 5.0  = etaj      ! Max jet abs(pseudo-rap) (a value .lt.0 means no cut)
```

Then run
```bash
./patch_soft_integrated.sh process_api_storage/counterterms/ee_3j/SubProcesses/P0_epem_uuxg
```
Now you can use the `SoftIntegratedCounterterm` class from `matrix_element.py` to call it.
```python
api = SoftIntegratedCounterterm("process_api_storage/counterterms/ee_3j/SubProcesses/P0_epem_uuxg/api_soft_integrated.so")
matrix_element = api(momenta)
```

**IMPORTANT**: if this worked, you can jump straight to the collinear counterterm. Otherwise we address here a possible failure mode of the previous code. This can fail if the MadGraph files were compiled for requiring *at least* an older version of MacOS of what you have. For me, what worked was compiling `handling_lhe_events.f` as an object file and then linking it manually. To do so, run:
```bash
cd matrix_element/process_api_storage/counterterms/ee_3j/SubProcesses/P0_epem_uuxg
gfortran -O -ffixed-line-length-132 -fno-automatic -c \
  -I. -I../../lib \
  ../handling_lhe_events.f
```
Now just insert `handling_lhe_events.o` after `api_soft_integrated.o`:
```bash
gfortran -O -ffixed-line-length-132 -fno-automatic -shared -fPIC \
  -o api_soft_integrated.so api_soft_integrated.o handling_lhe_events.o ../../libmadgraph.a \
  -L../../lib/ -ldhelas -lgeneric -lmodel -lpdf -lcernlib  \
  -L../../lib/ libMadLoop.a -lcts -liregi \
  -L/path/to/MadGraph/MG5_aMC_v3_5_8/HEPTools/lib/ -lninja \
  -L/path/to/MadGraph/MG5_aMC_v3_5_8/HEPTools/lib/ -lavh_olo \
  -L/path/to/MadGraph/MG5_aMC_v3_5_8/HEPTools/lib/ -lcollier \
  -L../../lib/ -ldhelas -lgeneric -lmodel -lpdf -lcernlib  \
  -lc++ -lc++ -mmacosx-version-min=10.8
```
Check that `api_soft_integrated.so` has been created, seeing that it has been dynamically linked to the correct libraries:
```bash
ls -lh api_soft_integrated.so
file api_soft_integrated.so
```


#### Collinear integrated counterterm
The construction is the same as the soft integrated counterterm

```bash
./patch_collinear_integrated.sh process_api_storage/counterterms/ee_3j/SubProcesses/P0_epem_uuxg
```
And the corresponding class is `CollinearIntegratedCounterterm`
```python
api = CollinearIntegratedCounterterm("process_api_storage/counterterms/ee_3j/SubProcesses/P0_epem_uuxg/api_collinear_integrated.so")
matrix_element = api(momenta)
```

**DISCLAIMERS**
- all that is currently in the soft and collinear apis works just for rather specific situations, in particular:
- colliding e+ e-, fixed energy to: 500 GeVs (standard MG input)
- final states: massless quarks/gluons. Only QCD radiation, no QED

#### Real emission: Sigma

The construction is the same as the collinear and soft integrated counterterms

```bash
./patch_real.sh process_api_storage/counterterms/ee_3j/SubProcesses/P0_epem_uuxg
```
And the corresponding class is `CollinearIntegratedCounterterm`
```python
api = RealEmission("process_api_storage/counterterms/ee_3j/SubProcesses/P0_epem_uuxg/api_collinear_integrated.so")
matrix_element = api(momenta)
```