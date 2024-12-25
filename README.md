# AOT (Anti Occlusion Tracker)

## Installation
### Main
Main repo: https://github.com/philly12399/Anti_Occlusion_Tracker

Utils repo: https://github.com/philly12399/philly_utils

**Environment Installation**
```
## conda install
cd ./Anti_Occlusion_Tracker
conda env create -f AOT_environment.yml  --name AOT
conda activate AOT
pip install -r requirement.txt
cd Xinshuo_PyToolbox
pip install -r requirement.txt
cd ..

#replace $PREFIX, export everytime restart bash
export PYTHONPATH={$PREFIX/Anti_Occlusion_Tracker}:{$PREFIX/Anti_Occlusion_Tracker/Xinshuo_PyToolbox}:${PYTHONPATH}
```
**How to run code**
```
##run
bash NDT_precalculate.sh ##NDT precalculate
bash test.sh 
bash eval.sh 
```
`NDT_precalculate.sh`  reads config files in `configs/NDT`

`test.sh` reads config files in `configs/`

`eval.sh` reads config files in `scripts/KITTI/configs/`

### Point-MAE
Repo: https://github.com/philly12399/Point-MAE

download pretrain.pth and put it in `Point-MAE/models/`

![image](https://hackmd.io/_uploads/rJJVtjMSyl.png)

point-mae reconstruct point cloud to dense point cloud

require create **gt_db** first( **gt_db**: split point cloud of each detection bbox)

**Environment Installation**
```
#import docker image
docker import ./0316-pointmae.tar pointmae
# run docker image, replace {dir}
docker run -it --shm-size="16g" -e PATH=/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin -v {your_pointmae_dir}:/Point-MAE -v {your_data_dir}:/mydata \
-v /tmp/.X11-unix:/tmp/.X11-unix  --runtime=nvidia --gpus=all pointmae:latest bash
# create link for mounted /mydata , so dataflow is same as outside(without docker)
mkdir -p /home/philly12399/
ln -s  /mydata /home/philly12399/philly_ssd
```
**How to run Point-MAE**
```
cd /Point-MAE
bash vis_kitti.sh / bash vis_wayside.sh
```
change cfgs/

take `vis_wayside.sh` for example, it reads`cfgs/pretrain_test_on_wayside_det.yaml`

furthermore `dataset->test` reads`cfgs/dataset_configs/Wayside-DET.yaml`

the former controls pointmae parameters

the latter controls path of input data and sequences

### Tracking Visualizer
Repo: https://github.com/philly12399/3D-Detection-Tracking-Viewer

**Environment Installation**
```
## conda
cd ./3D-Detection-Tracking-Viewer
conda env create -f trackvis_environment.yml  --name trackvis
conda activate trackvis
pip install -r requirement.txt
```
**How to run visualizer**
```
python3 tracking_viewer.py
```
You can change point cloudpath,sequnces,bbox type,label path ... in `tracking_viewer.py`.

## Data & Procedure
**Data**
* D1. Pointcloud pcd (kitti format) 
* D2. Detection bbox
* D3. gt_db， split point cloud of each detection bbox
* D4. dense_pcd， dense point cloud from point cloud reconstruction
* D5. NDT voxelize cache of detection
* D6. Tracking results
* D6*. Refined Tracking results
* D7. Ground Truth
* D8. Tracking Metrics

**Procedure**
* P1. Generate gt_db， `philly_utils/utils/create_gt_db_kitti.py`， D1+D2->D3 
* P2. Generate dense_pcd，`Point-MAE/` point cloud reconstruct, D3->D4
* P3. `Anti_Occlusion_Tracker/NDT_precalculate.py` ,pre-calculate NDT voxelize for detection ， D4->D5
* P4. Tracking,`Anti_Occlusion_Tracker/main.py`, D2+D4->D6
* P5. Post process tracking results by track-level confidence, `Anti_Occlusion_Tracker/post_process.sh`, refine D6->D6*
* P6. Evaluation,`Anti_Occlusion_Tracker/scripts/KITTI/evaluate.py` D6+D7->D8
* P7. Tracking Visualizer,`3D-Detection-Tracking-Viewer/tracking_viewer.py` D1+bbox label(D2/D6/D7)->visualize

## Introduction

## Quick Demo on KITTI

## Benchmarking

## Acknowledgement

The idea of this method is inspired by "[AB3DMOT](https://github.com/xinshuoweng/AB3DMOT)"

