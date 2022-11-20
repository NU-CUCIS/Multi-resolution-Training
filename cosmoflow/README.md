
## CosmoFlow Run Instructions
CosmoFlow is a parallel deep learning application developed for studying
data generated from cosmological N-body dark matter simulations.
The source codes of CosmoFlow are available on both
[github](https://github.com/NERSC/CosmoFlow)
and [MLPerf](https://mlcommons.org/en/training-hpc-10/).
Programs in this repo update the CosmoFlow source codes by incorporating
the [LBANN model](https://www.osti.gov/servlets/purl/1548314)
and parallelizing it using [Horovod](https://github.com/horovod/horovod#citation).
The [training data files](https://portal.nersc.gov/project/m3363/) are available at NERSC.

### Instructions to run on [Cori](https://docs.nersc.gov/systems/cori/)
1. Load and configure the modules.
```
CPU (KNL): module load tensorflow/intel-2.4.1
GPU:  module load cgpu tensorflow/gpu-2.2.0-py37
```
2. In `test.yaml` to point to the correct paths.
* `sourceDir/prj`: the top directory of the data files.
* `subDir`: the sub-directory under `sourceDir/prj`, where the coarse files are located.
* `subDir_denser`: the sub-directory under `sourceDir/prj`, where the dense files are located.

3. Set command-line options.
   * `--epochs`: the number of epochs for training.
   * `--batch_size`: the local batch size (the batch size for each process).
   * `--overlap`: (0:off / 1:on) disable/enable the I/O overlap feature.
   * `--checkpoint`: (0:off / 1:on) disable/enable the checkpointing.
   * `--buffer_size`: the I/O buffer size with respect to the number of samples.
   * `--record_acc`: (0:off / 1:on) disable/enable the accuracy recording.
   * `--config`: the file path for input data configuration.
   * `--enable`: (0:off / 1:on) disable/enable evaluation of the trained model.
   * `--async_io`: (0:off / 1:on) disable/enable the asynchronous I/O feature.

4. Run the codes.
```
sbatch sbatch.sh
```

### Instructions to run on [Summit](https://www.olcf.ornl.gov/summit/)
1. Load the modules.
```
module load open-ce/1.4.0
```
2. In `test_summit.yaml` to point to the correct paths.
* `sourceDir/prj`: the top directory of the data files.
* `subDir`: the sub-directory under `sourceDir/prj`, where the coarse files are located.
* `subDir_denser`: the sub-directory under `sourceDir/prj`, where the dense files are located.

3. Set command-line options.

4. Run the codes.
```
bsub myjob64.lsf
```
