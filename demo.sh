# python3 main.py --dataset KITTI --split val --det_name pointrcnn
# python3 scripts/post_processing/trk_conf_threshold.py --dataset KITTI --result_sha pointrcnn_val_H1
# python3 scripts/post_processing/visualization.py --result_sha pointrcnn_val_H1_thres --split val

#  python3 main.py --config ./configs/KITTI_demo.yml 
 python3 main.py --config ./configs/KITTI_demo.yml 

# python3 scripts/post_processing/trk_conf_threshold.py --dataset Wayside --result_sha mark_val_H1
# python3 scripts/post_processing/visualization.py --dataset Wayside --result_sha mark_val_H1_thres --split val