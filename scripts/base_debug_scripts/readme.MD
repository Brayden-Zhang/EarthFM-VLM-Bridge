# How to Use these scripts

These scripts contain a reccomended set up to get relatively fast start up and training.
These settings should do about 20 minute per epoch on 1 GPU with batch size 128 (Eval not timed)

For faster Training use more gpus bigger batch size potentially more workers until you are saturated

IF You want to share the runs with others please set the wandb project use override `--trainer.wandb.project=helios-train`

To alternate between different debug scripts, please delete other test run checkpoints in `local_output` or change `LOAD_STRATEGY` to `"never"` in `build_trainer_config.py`.
