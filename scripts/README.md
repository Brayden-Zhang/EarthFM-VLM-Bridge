# OlmoEarth scripts

This folder contains scripts run during the development of the OlmoEarth family of models.
The majority of the scripts are [`archived`](archived) - this means they may not run against current versions of the codebase.

The non archived scripts contain:
- [`2025_10_02_phase2`](2025_10_02_phase2): This folder contains our final training runs (which swept learning rates and weight decays) and our final ablations.
- Utility scripts to manage runs. These include:
    - [`get_and_run_evals_from_project`](get_and_run_evals_from_project.py) which (as the name suggests) pulls a training run from wandb and kicks off a full set of evals against it.
    - [`get_max_eval_metrics_from_wandb`](get_max_eval_metrics_from_wandb.py) which (as the name also suggests) gets the best evaluation metrics from wandb, once the evals have been run. We pull the highest metrics because some sweeping happens at eval time (learning rate, normalizations). We pull the validation metrics and optionallty the test metrics, indexed by the highest validation results.
- Scripts to manipulate / prepare / compute evals. These include:
    - [`compute_rslearn_band_stats`](compute_rslearn_dataset_band_stats.py) which was used to compute normalization statistics from our partner tasks, so that they could be run within this codebase.
    - [`20250924_pastis128`](20250924_pastis128.py), which uses the original 128x128 window sizes for pastis (compared to the 64x64 window sizes, following [Galileo](https://arxiv.org/abs/2502.09356)).
