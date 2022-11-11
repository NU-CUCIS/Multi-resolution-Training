# Using Multi-Resolution Data to Accelerate Training for CosmoFlow and DeepCAM
This repository contains an improvement of [DeepCAM](https://github.com/azrael417/mlperf-deepcam) and [CosmoFlow](https://arxiv.org/abs/1808.04728) by adding the support of utlizing multi-resolution data.

DeepCAM is a parallel deep learning climate segmentation benchmark. The source codes of DeepCAM are available on both [github](	
https://github.com/azrael417/mlperf-deepcam) and [MLPerf](https://mlcommons.org/en/training-hpc-10/).
The [training data files](https://app.globus.org/file-manager?origin_id=0b226e2c-4de0-11ea-971a-021304b0cca7&origin_path=%2F) are available at GLOBUS.

CosmoFlow is a parallel deep learning application developed for studying
data generated from cosmological N-body dark matter simulations.
The source codes of CosmoFlow are available on both
[github](https://github.com/NERSC/CosmoFlow)
and [MLPerf](https://mlcommons.org/en/training-hpc-10/).
Programs in this repo update the CosmoFlow source codes by incorporating
the [LBANN model](https://www.osti.gov/servlets/purl/1548314)
and parallelizing it using [Horovod](https://github.com/horovod/horovod#citation).
The [training data files](https://portal.nersc.gov/project/m3363/) are available at NERSC.

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
