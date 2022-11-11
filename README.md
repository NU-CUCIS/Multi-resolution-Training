# Using Multi-Resolution Data to Accelerate Training for DeepCAM
This repository contains an improvement of [DeepCAM](https://github.com/azrael417/mlperf-deepcam) by adding the support of utlizing multi-resolution data.
DeepCAM is a parallel deep learning climate segmentation benchmark. The source codes of DeepCAM are available on both [github](	
https://github.com/azrael417/mlperf-deepcam) and [MLPerf](https://mlcommons.org/en/training-hpc-10/).
The [training data files](https://app.globus.org/file-manager?origin_id=0b226e2c-4de0-11ea-971a-021304b0cca7&origin_path=%2F) are available at GLOBUS.


## CosmoFlow Run Instructions
### Instructions to run on Cori
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

### Instructions to run on Summit
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

## Publication
* Kewei Wang, Sunwoo Lee, Jan Balewski, Alex Sim, Peter Nugent, Ankit Agrawal, Alok Choudhary, Kesheng Wu, and Wei-keng Liao. Using Multi-Resolution Data to Accelerate Neural Network Training in Scientific Applications. In the 22nd IEEE/ACM International Symposium on Cluster, Cloud and Internet Computing (CCGrid), May 2022.

## Development team
  * Northwestern University
    + Kewei Wang <<keweiwang2019@u.northwestern.edu>>
    + Sunwoo Lee <<sunwoolee1.2014@u.northwestern.edu>>
    + Wei-keng Liao <<wkliao@northwestern.edu>>
  * Lawrence Berkeley National Laboratory
    + Alex Sim <<asim@lbl.gov>>
    + Jan Balewski <<balewski@lbl.gov>>
    + Peter Nugent <<penugent@lbl.gov>>
    + John Wu <<kwu@lbl.gov>>

## Questions/Comments
  * Kewei Wang <<keweiwang2019@u.northwestern.edu>>
  * Wei-keng Liao <<wkliao@northwestern.edu>>

## Project Funding Supports
This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research, Scientific Discovery through Advanced Computing ([SciDAC](https://www.scidac.gov)) program. This project is a joint work of Northwestern University and Lawrence Berkeley National Laboratory supported by the [RAPIDS Institute](https://rapids.lbl.gov).
