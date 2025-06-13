#!/bin/bash

python scripts/joe/latent_mim_st.py launch latent_mim_large_st_contrastive_random ai2/jupiter-cirrascale-2 --launch.priority=urgent --common.launch.num_gpus=8
