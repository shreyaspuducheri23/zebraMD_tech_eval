#!/bin/bash -l
# batch script to submit job on SCC
#$ -P vkolagrp
#$ -N zebraMD_nn_sweep_           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=6
#$ -m bes
#$ -l h_rt=3:00:00

module load miniconda
conda activate mri_radiology

python /usr4/ugrad/spuduch/zebraMD/Model/nn.py

