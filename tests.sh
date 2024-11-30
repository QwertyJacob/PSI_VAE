#!/bin/bash

for latent_dim in 32 64 128 256 512 1024
do
  cmd="CUDA_VISIBLE_DEVICES=0 python3 train.py \
    name=latent_dim_${latent_dim} \
    override=dista \
    latent_dim=${latent_dim}"
  echo Issuing command: $cmd
  eval $cmd
done

