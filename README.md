<h1 align="center"> MadNIS at NLO </h1>
<p align="center">
<a href="https://pytorch.org"><img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.2-DC583A.svg?style=flat&logo=pytorch"></a>
<a href="https://hydra.cc/"><img alt="Config style: Hydra" src="https://img.shields.io/badge/Hydra-1.3-78a9c2"></a>
<a href="https://github.com/psf/black"><img alt="Code style: Black" src="https://img.shields.io/badge/Black-22.3-000000.svg"></a>
<a href="https://github.com/MadGraphTeam/MadGraph7/tree/main/madspace"><img alt="GitHub: MadGraphTeam/MadGraph7/madspace" src="https://img.shields.io/badge/MadGraphTeam-madspace-181717.svg?logo=github"></a>
<a href="https://github.com/madgraph-ml/torchspace"><img alt="GitHub: madgraph-ml/torchspace" src="https://img.shields.io/badge/madgraph--ml-torchspace-181717.svg?logo=github"></a>
</a>
<a href="https://github.com/MadGraphTeam/madnis"><img alt="GitHub: MadGraphTeam/madnis" src="https://img.shields.io/badge/MadGraphTeam-madnis-181717.svg?logo=github"></a>
</p>

## Getting started
Create a new conda environment and install with the following command:

```bash
git clone git@github.com:madgraph-ml/madnis-nlo.git
```
`Python>=3.11` is in principle required to install all the necessary packages, which are available on PyPI for Linux and MacOS X (with Apple Silicon). For details see https://github.com/madgraph-ml/madevent7.

```bash
cd madnis-nlo/
python3.11 -m venv venv
source venv/bin/activate
pip install -e .
```

## Generate matrix element APIs
To generate the matrix element APIs, it is recommended to have the mg5amcnlo github repository cloned. You can follow the README.md in `src/integration/matrix_element/` to generate the APIs either automatically with the provided script, or manually by following the instructions. The script that automatically generates the APIs is located in `src/integration/matrix_element/generate_and_patch.py`, and can be run as follows:

```bash
python src/integration/matrix_element/generate_and_patch.py --mg5-dir /path/to/mg5amcnlo
```


## Standard training
Once the above has run successfully. From the parent folder run the following:
```bash
python run.py -cp config -cn integrator_run run.name=run_name process=ee_3j
```

This will generate a folder results inside `ee_3j/` or `ee_4j/` depending on the `process`, and inside it, a folder with the timestamp and the run name. Inside the folder, there will be the run `config.yaml` and the trained MadNIS model as `integrator.pth`. By default, the training will run a small vegas pretraining for 5 iterations, and then train MadNIS for 1000 iterations. To change these settings, simply modify the `vegas_pretraining`, `vegas_iterations`, and `train_iterations` fields in the `config/integrator/multichannel.yaml` file. To enable pure Vegas training, set `vegas: true`, and adjust the number of iterations with `vegas_iterations`.

## Use a pretrained integrator

To re-run a saved model, simply run the same command with `-cp` pointing to the folder containing the `integrator.pth` and `config.yaml` files, and `-cn` pointing to the config name inside:

```bash
python run.py -cp results/ee_{3_or_4}j/run_name -cn config integrator.sample_count=N
```

## Citation

    @article{DeCrescenzo:2026tsp,
    author = "De Crescenzo, Giovanni and Villadamigo, Javier Mari{\~n}o and Elmer, Nina and Heimel, Theo and Plehn, Tilman and Winterhalder, Ramon and Zaro, Marco",
    title = "{MadNIS at NLO}",
    eprint = "2603.22407",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "TIF-UNIMI-2026-2, IRMP-CP3-26-06, MCNET-26-04",
    month = "3",
    year = "2026"
    }
