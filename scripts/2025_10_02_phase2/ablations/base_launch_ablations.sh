# no contrastive loss ablation
python scripts/2025_10_02_phase2/base.py launch phase2.0_base_no_contrastive ai2/ceres-cirrascale  --train_module.contrastive_config.loss_config.weight=0.0 --launch.clusters='[ai2/jupiter-cirrascale-2]' --launch.priority=high
# random masking
python scripts/2025_10_02_phase2/base.py launch phase2.0_base_random_masking ai2/ceres-cirrascale  --train_module.masking_config.strategy_config="{'type': 'random', 'encode_ratio': 0.5, 'decode_ratio': 0.5}" --launch.clusters='[ai2/jupiter-cirrascale-2]' --launch.priority=high
# MAE
python scripts/2025_10_02_phase2/ablations/base_mae.py phase2.0_base_mae ai2/ceres-cirrascale  --launch.clusters='[ai2/jupiter-cirrascale-2]' --launch.priority=high
