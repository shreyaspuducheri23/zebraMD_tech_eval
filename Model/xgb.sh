#!/bin/bash -l
# batch script to submit job on SCC
#$ -P vkolagrp
#$ -N zebraMD_sweep           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -pe omp 8
#$ -m bes
#$ -l h_rt=1:00:00

module load miniconda
conda activate mri_radiology

python /usr4/ugrad/spuduch/zebraMD/Model/xgb.py

