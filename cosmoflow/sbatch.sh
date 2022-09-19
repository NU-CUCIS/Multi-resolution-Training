#!/bin/bash  -l

#SBATCH -t 00:40:00
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --constraint=gpu
#SBATCH -G 32
#SBATCH --exclusive
#SBATCH -A m844

ulimit -c unlimited

export MPICH_MAX_THREAD_SAFETY=multiple

srun -n 32 -c 10 python3 main.py --epochs=100 \
                                 --batch_size=8 \
                                 --overlap=0 \
                                 --checkpoint=1 \
                                 --buffer_size=128 \
                                 --file_shuffle=1 \
                                 --record_acc=0 \
                                 --config="test.yaml" \
                                 --evaluate=1 \
                                 --async_io=0 
