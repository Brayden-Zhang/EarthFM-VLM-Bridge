#!/bin/bash

#python scripts/v0_sweep/galileo.py launch test_sweep_gal ai2/titan-cirrascale --model.decoder_config.depth=4 --common.launch.num_gpus=8
#python scripts/v0_sweep/latent_mim.py launch test_sweep_gal ai2/titan-cirrascale --model.decoder_config.depth=4 --common.launch.num_gpus=8
#python scripts/v0_sweep/contrastive_latent_mim.py launch test_sweep_gal ai2/titan-cirrascale --model.decoder_config.depth=4 --common.launch.num_gpus=8

#w/ and w/o cacheing
python scripts/v0_sweep/galileo.py launch test_sweep_gal_no_mae ai2/jupiter-cirrascale --model.reconstructor_config=null --train_module.mae_loss_config=null --model.decoder_config.depth=4 --common.launch.num_gpus=8
python scripts/v0_sweep/galileo.py launch test_sweep_gal_no_mae ai2/jupiter-cirrascale --model.reconstructor_config=null --train_module.mae_loss_config=null --model.decoder_config.depth=4 --common.launch.num_gpus=8 --dataset.dataset_configs[0].cache_dir=null --dataset.dataset_configs[1].cache_dir=null
