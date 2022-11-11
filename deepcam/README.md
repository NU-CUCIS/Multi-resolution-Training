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
