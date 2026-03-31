# Integration

## Getting started
Create a new conda environment and install parx with the following command:

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
Once the above has run successfully. From parx main folder run the following:
```bash
python run.py -cp config -cn integrator_run run.name=run_name process=ee_3j
```

This will generate a folder results inside `ee_3j/` or `ee_4j/` depending on the `process`, and inside it, a folder with the timestamp and the run name. Inside the folder, there will be the run `config.yaml` and the trained MadNIS model as `integrator.pth`. By default, the training will run a small vegas pretraining for 5 iterations, and then train MadNIS for 1000 iterations. To change these settings, simply modify the `vegas_pretraining`, `vegas_iterations`, and `train_iterations` fields in the `config/integrator/multichannel.yaml` file. To enable pure Vegas training, set `vegas: true`, and adjust the number of iterations with `vegas_iterations`.

## Use a pretrained integrator

To re-run a saved model, simply run the same command with `-cp` pointing to the folder containing the `integrator.pth` and `config.yaml` files, and `-cn` pointing to the config name inside:

```bash
python run.py -cp results/ee_{3_or_4}j/run_name -cn config integrator.sample_count=N
```