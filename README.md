# AOT (Anti Occlusion Tracker)

## Installation
Please follow carefully our provided [installation instructions](docs/INSTALL.md), to avoid errors when running the code.

## Introduction
### Data & Procedure
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



## Quick Demo on KITTI

## Benchmarking

## Acknowledgement

The idea of this method is inspired by "[AB3DMOT](https://github.com/xinshuoweng/AB3DMOT)"

