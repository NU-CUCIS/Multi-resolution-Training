## Training on Cori
### Instructions to modify file run_training_cori.sh
1. Edit the entries:
```
export PROJ_LIB=/global/homes/k/kwf5687/.conda/envs/mlperf_deepcam/share/basemap
export PYTHONPATH=/global/homes/k/kwf5687/.conda/envs/mlperf_deepcam/lib/python3.8/site-packages:${PYTHONPATH}
```
2. Add:
```
--wandb_certdir "/global/homes/k/kwf5687/deepcam/mlperf-deepcam" \
```
store the file named `.wandbirc` containing the user login and the API key as follows:

```
user 0741b9d6e85127758db3ba6da44b079521be8190
```
3. In file named:
```
/global/homes/k/kwf5687/.conda/envs/mlperf_deepcam/lib/python3.8/site-packages/mpl_toolkits/basemap/__init__.py
/global/homes/k/kwf5687/.conda/envs/mlperf_deepcam/lib/python3.8/site-packages/mpl_toolkits/basemap/proj.py
```
Replace the line import `dedent` with:
```
from inspect import cleandoc as dedent
```

### Fix error messages
1. error message:
```
Traceback (most recent call last):
  File "../train_hdf5_ddp.py", line 66, in <module>
    from utils import visualizer as vizc
  File "/global/u2/k/kwf5687/deepcam/mlperf-deepcam/src/deepCam/utils/visualizer.py", line 38, in <module>
    from mpl_toolkits.basemap import Basemap
  File "/global/homes/k/kwf5687/.conda/envs/mlperf_deepcam/lib/python3.8/site-packages/mpl_toolkits/basemap/__init__.py", line 157, in <module>
    epsgf = open(os.path.join(pyproj_datadir,'epsg'))
FileNotFoundError: [Errno 2] No such file or directory: '/global/homes/k/kwf5687/.conda/envs/mlperf_deepcam/share/basemap/epsg'
```
Add:
```
os.environ['PROJ_LIB'] = '/global/homes/k/kwf5687/.conda/envs/mlperf_deepcam/share/proj'
```
in file named `./src/deepCam/utils/visualizer.py`.

2. error message:
```
Traceback (most recent call last):
  File "/global/homes/k/kwf5687/.conda/envs/mlperf_deepcam/lib/python3.8/site-packages/mpl_toolkits/basemap/__init__.py", line 1234, in _readboundarydata
    bdatfile = open(os.path.join(basemap_datadir,name+'_'+self.resolution+'.dat'),'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/usr/common/software/pytorch/1.7.1/share/basemap/gshhs_i.dat'
```
Run:
```
conda install -c conda-forge basemap-data-hires 
```

3. error message:
```
Traceback (most recent call last):
  File "../train_hdf5_ddp.py", line 581, in <module>
    main(pargs)
  File "../train_hdf5_ddp.py", line 389, in main
    viz.plot(filename[sample_idx], outputfile, plot_input, plot_prediction, plot_label)
UnboundLocalError: local variable 'viz' referenced before assignment
```
Remove line 389 in file named `./src/deepCam/train_hdf5_ddp.py` when `training/validation_visualization_frequency` are set to 0 in file run_training_cori.sh.

## Training on Summit
### Fix error messages
1. error message:
```
Traceback (most recent call last):
  File "../train_hdf5_ddp.py", line 66, in <module>
    from utils import visualizer as vizc
  File "/autofs/nccs-svm1_home1/kwf5687/mlperf-deepcam/src/deepCam/utils/visualizer.py", line 38, in <module>
    from mpl_toolkits.basemap import Basemap
  File "/ccs/home/kwf5687/.conda/envs/mlperf_deepcam2/lib/python3.7/site-packages/mpl_toolkits/basemap/__init__.py", line 26, in <module>
    from matplotlib.cbook import dedent
ImportError: cannot import name 'dedent' from 'matplotlib.cbook' (/ccs/home/kwf5687/.conda/envs/mlperf_deepcam2/lib/python3.7/site-packages/matplotlib/cbook/__init__.py)
```
Replace the line import dedent in file named `/ccs/home/kwf5687/.conda/envs/mlperf_deepcam2/lib/python3.7/site-packages/mpl_toolkits/basemap/__init__.py` and `proj.py` with 
```
from inspect import cleandoc as dedent
```

2. error message:
```
CUDA Hook Library: Failed to find symbol mem_find_dreg_entries, ./a.out: undefined symbol: __PAMI_Invalidate_region
```
Can be fixed in 3 ways:
```
use jsrun -E LD_PRELOAD=/opt/ibm/spectrum_mpi/lib/pami_451/libpami.so ...
use jsrun --smpiargs="off" ...
use jsrun --smpiargs="-disable_gpu_hooks ...
```

3. Add extra wireup method in file named `comm.py`, function init():
```
elif method == "nccl":
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        import subprocess
        get_master = "echo $(cat {} | sort | uniq | grep -v batch | grep -v login | head -1 )".format(os.environ['LSB_DJOB_HOSTFILE'])
        os.environ['MASTER_ADDR'] = str(subprocess.check_output(get_master, shell=True))[2:-3]
        os.environ['MASTER_PORT'] = "23456"
        os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
        os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
        dist.init_process_group(backend = "nccl", rank=world_rank, world_size=world_size)
```

