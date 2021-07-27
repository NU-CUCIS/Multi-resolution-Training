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
