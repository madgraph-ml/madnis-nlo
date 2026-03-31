from typing import Literal, TypedDict


class ParamsDict(TypedDict, total=False):
    # Dataset
    run_name: str
    file_path: str
    split_fractions: list[float]
    batch_size: int
    no_stoch: bool
    dtype: Literal["float32", "float64"]

    # Training
    bayesian: bool
    repulsive: bool
    prior_width: int
    transforms: dict[str, dict]
    input_dim: int
    output_dim: int
    log: bool
    lr: float
    betas: list[float]  # Tuple[float, float] could also work
    weight_decay: float
    use_scheduler: bool
    lr_scheduler: Literal[
        "step",
        "reduce_on_plateau",
        "one_cycle_lr",
        "cycle_lr",
        "multi_step_lr",
        "cosine_annealing",
        "cosine_annealing_warm_restart",
        "constant",
    ]
    max_lr: float
    dropout: float
    n_epochs: int
    cycle_epochs: int

    # Cut-off for outliers
    cut_off: float

    # Model
    model: str
    network: str
    hidden_dim: list[int]
    bayesian_layers: list[bool]
    bayesian_output_layer: bool
    nmfa_bayesian: bool
    bayesian_bias: bool
    activation: str

    # Validation
    val_interval: int
    predict: bool
    n_sample: int
