#!/bin/bash
#BSUB -P AST153
#BSUB -W 02:00
#BSUB -nnodes 11
#BSUB -J lk
#BSUB -o lk.%J
#BSUB -e lk.%J
#BSUB -N


#parameters
run_tag="deepcam_prediction_run_c2d_2-summit"
data_c_dir_prefix="/gpfs/alpine/ast153/scratch/kwf5687/deepcam/All-Hist_c"
data_dir_prefix="/gpfs/alpine/ast153/scratch/kwf5687/deepcam/All-Hist"
output_dir="/gpfs/alpine/ast153/scratch/kwf5687/deepcam/All-Hist/cam5_runs/${run_tag}"

#create files
mkdir -p ${output_dir}
touch ${output_dir}/train.out

#export OMP_NUM_THREADS=32
jsrun -n64 -a1 -c4 -g1 --smpiargs="off" python3 ../train_hdf5_ddp.py \
                                       --wireup_method "nccl" \
                                       --wandb_certdir "/ccs/home/kwf5687/mlperf-deepcam/src/deepCam" \
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
                                       --validation_frequency 100 \
                                       --training_visualization_frequency 0 \
                                       --validation_visualization_frequency 0 \
                                       --logging_frequency 10 \
                                       --save_frequency 100 \
                                       --max_epochs 200 \
                                       --amp_opt_level O1 \
				                       --local_batch_size_c 8 \
                                       --local_batch_size 2 |& tee -a ${output_dir}/train.out

