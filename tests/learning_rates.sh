#!/bin/bash

for lr in 0.00001 0.000050.0001 0.0005 0.001
do
  cmd="CUDA_VISIBLE_DEVICES=1 python3 train.py \
    name=lr_${lr} \
    override=dista \
    learning_rate=${lr}"
  echo Issuing command: $cmd
  eval $cmd
done

