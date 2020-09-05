#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-8.0/bin/:$PATH
export CUDNN_PATH=/usr/local/cuda-8.0/lib64/libcudnn.so.5
source /mnt/data/Rohit/VideoCapsNet/env/mask-rcnn/bin/activate

CUDA_VISIBLE_DEVICES=0 python3 train.py train --dataset=/mnt/data/Rohit/VideoCapsNet/data/publaynet --weights=coco
# CUDA_VISIBLE_DEVICES=0 python3 train.py train --dataset=/mnt/data/Rohit/VideoCapsNet/data/publaynet --weights=last

