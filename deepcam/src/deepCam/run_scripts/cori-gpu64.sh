#!/bin/bash
#SBATCH -A m844
#SBATCH -J train_cam5
#SBATCH -t 02:00:00
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --exclusive
#SBATCH -G 64
#SBATCH --nodes=8
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 10

# The MIT License (MIT)
#
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#load stuff
#conda activate mlperf_deepcam
#module load pytorch/v1.4.0
export PROJ_LIB=/global/homes/k/kwf5687/.conda/envs/mlperf_deepcam/share/basemap
export PYTHONPATH=/global/homes/k/kwf5687/.conda/envs/mlperf_deepcam/lib/python3.8/site-packages:${PYTHONPATH}

#ranks per node
rankspernode=8
totalranks=$(( ${SLURM_NNODES} * ${rankspernode} ))

#parameters
run_tag="deepcam_prediction_run64_c2d_1-cori"
data_c_dir_prefix="/global/cscratch1/sd/kwf5687/deepcam/All-Hist_c"
data_dir_prefix="/global/cscratch1/sd/kwf5687/deepcam/All-Hist"
output_dir="/global/cscratch1/sd/kwf5687/deepcam/All-Hist/cam5_runs/${run_tag}"


#create files
mkdir -p ${output_dir}
touch ${output_dir}/train.out

#run training
srun -u -N ${SLURM_NNODES} -n ${totalranks} -c $(( 80 / ${rankspernode} )) --cpu_bind=cores \
	python ../train_hdf5_ddp.py \
	--wireup_method "nccl-slurm-pmi" \
	--wandb_certdir "/global/homes/k/kwf5687/deepcam/mlperf-deepcam" \
	--run_tag ${run_tag} \
	--data_c_dir_prefix ${data_c_dir_prefix} \
	--data_dir_prefix ${data_dir_prefix} \
	--output_dir ${output_dir} \
	--max_inter_threads 2 \
	--model_prefix "classifier" \
	--optimizer "LAMB" \
	--start_lr 2e-3 \
	--lr_schedule type="multistep",milestones="4800 16384",decay_rate="0.1" \
	--weight_decay 1e-2 \
	--validation_frequency 100000 \
	--training_visualization_frequency 0 \
	--validation_visualization_frequency 0 \
	--logging_frequency 10 \
	--save_frequency 400 \
	--max_epochs 4 \
	--amp_opt_level O1 \
	--local_batch_size_c 8 \
	--local_batch_size 2 |& tee -a ${output_dir}/train.out

