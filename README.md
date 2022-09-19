# Using Multi-Resolution Data to Accelerate Training for DeepCAM
This repository contains an improvement of [DeepCAM](https://github.com/azrael417/mlperf-deepcam) by adding the support of utlizing multi-resolution data.
DeepCAM is a parallel deep learning climate segmentation benchmark. The source codes of DeepCAM are available on both [github](	
https://github.com/azrael417/mlperf-deepcam) and [MLPerf](https://mlcommons.org/en/training-hpc-10/).
The [training data files](https://app.globus.org/file-manager?origin_id=0b226e2c-4de0-11ea-971a-021304b0cca7&origin_path=%2F) are available at GLOBUS.

## DeepCAM Run Instructions

Submission scripts are in `deepcam/src/deepCam/run_scripts`.
### Before you run

Make sure you have a working python environment with `pytorch`, `h5py`, `basemap` and `wandb` setup. 
If you want to use learning rate warmup, you must also install the warmup-scheduler package
available at https://github.com/ildoonet/pytorch-gradual-warmup-lr.

The training uses Weights & Biases (WandB/W&B, https://app.wandb.ai) as logging facility. 
In order to use it, please sign up, log in and create a new project. 
Create a file named `.wandbirc` containing the user login and the API key as follows:

```bash
<login> <API key>
```

Place this file in a directory accessible by the workers.

### Instructions to run on Cori
1. Set up a conda environment
```
module load python
conda create -n mlperf_deepcam
```
2. Activate the environment.
```
source activate mlperf_deepcam
```
3. Load and configure the modules.
```
CPU (KNL): module load pytorch/1.7.1
GPU:  module load cgpu pytorch/1.7.1-gpu
```
4. Install packages.
```
pip install --user h5py
pip install --user wandb
conda install basemap yt
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
```
5. Edit the entries.

```bash
export PROJ_LIB=/global/homes/k/kwf5687/.conda/envs/mlperf_deepcam/share/basemap
export PYTHONPATH=/global/homes/k/kwf5687/.conda/envs/mlperf_deepcam/lib/python3.8/site-packages:${PYTHONPATH}
```

6. In `src/deepCam/run_scripts/cori-gpu64.sh` to point to the correct paths and add 

```bash
--wandb_certdir <my-cert-dir>
```
to the arguments passed to the python training script. Here, `<my-cert-dir>`
should point to the directory which contains the `.wandbirc` file created before.

7. Run the codes.
```bash
# This example runs on 64 nodes.
cd src/deepCam/run_scripts
sbatch cori-gpu64.sh
```

### Instructions to run on Summit
1. Set up a conda environment
```
module load open-ce/1.1.3-py37-0
conda create --name mlperf_deepcam --clone open-ce-1.1.3-py37-0
conda activate mlperf_deepcam
```
2. Activate the environment.
```
conda activate mlperf_deepcam
```

3. Install packages.
```
pip install --user wandb
conda install basemap
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
git clone https://github.com/mlperf/logging.git mlperf-logging
pip install -e mlperf-logging
```
4. In `src/deepCam/run_scripts/myjob64.lsf` to point to the correct paths and add 

```bash
--wandb_certdir <my-cert-dir>
```
to the arguments passed to the python training script. Here, `<my-cert-dir>`
should point to the directory which contains the `.wandbirc` file created before.

5. Run the codes.
```bash
# This example runs on 64 nodes.
cd src/deepCam/run_scripts
bsub myjob64.lsf
```

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
