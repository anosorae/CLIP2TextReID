#!/bin/bash

CUDA_VISIBLE_DEVICES=$count \
python train.py \
--name 'CLIP2TextReID' \
--batch_size 128 \
--sampler 'identity' \
--num_instance 1 \
--img_aug \
--tcmpm 'on'