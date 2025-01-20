# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

from __future__ import print_function
import matplotlib; matplotlib.use('Agg')
import os, numpy as np, time, sys, argparse
from AB3DMOT_libs.utils import Config, get_subfolder_seq, initialize
from Philly_libs.io_for_NDT_precalculate import load_detection, get_frame_det
from Philly_libs.philly_io import *
from Philly_libs.philly_utils import *
from datetime import datetime
from multiprocessing import Pool
from Philly_libs.NDT import NDT_voxelize,draw_NDT_voxel
from tqdm import tqdm
from AB3DMOT_libs.box import Box3D
def parse_args():
    parser = argparse.ArgumentParser(description='AB3DMOT')
    # parser.add_argument('--dataset', type=str, default='nuScenes', help='KITTI, nuScenes, Wayside')
    parser.add_argument('-c','--config', type=str, required=True, help='Config file path')
    parser.add_argument('--split', type=str, default='', help='train, val, test')
    parser.add_argument('--det_name', type=str, default='', help='pointrcnn')
    parser.add_argument('--frame', type=int, default=-1, help='frame num')

    args = parser.parse_args()
    return args

def process_dets(dets):
        # convert each detection into the class Box3D 
        # inputs: 
        # 	dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
        dets_new = []
        for det in dets:
            det_tmp = Box3D.array2bbox_raw(det)
            dets_new.append(det_tmp)
        return dets_new

import pdb    
def main_inner(cfg):
    # get data-cat-split specific path
    # print(cfg)
    cat_list = cfg.cat_list
    # cat = cat_list[0]
    np.random.seed(0)
    
    # for multiple rule of det name , 
    det_root = os.path.join('./data', cfg.dataset, 'detection', cfg.det_name)
    if(os.path.exists(det_root) == False):
        print(f"Didn't find det path {det_root}")
        all_sha = '%s_%s_%s' % (cfg.det_name, 'all', cfg.split)        
        det_root = os.path.join('./data', cfg.dataset, 'detection', all_sha)
        print(f"Find det path {det_root} instead.")        
        if(os.path.exists(det_root) == False):
            print(f"Didn't find det path {det_root}")           
            assert False       
            
    ##PCD INFO
    if('pcd_db_root' in cfg):
        pcd_db_root = cfg.pcd_db_root
        pcd_info = read_pkl(os.path.join(pcd_db_root, 'info.pkl'))
        pcd_db = os.path.join(pcd_db_root, 'gt_database')

    ##SEQ SETTING
    subfolder, det_id2str, hw, seq_eval, data_root = get_subfolder_seq(cfg.dataset, cfg.split)
    if(cfg.seq_eval != []):
        seq_eval = [str(s).zfill(4) for s in cfg.seq_eval]

    trk_root = os.path.join(data_root, 'tracking')

    # loop every sequence
    #TIME
    for seq_name in seq_eval:
        start=time.time()
        format = cfg.label_format.lower()
        if(seq_name == "0021"): format = "wayside"  
        Box3D.set_label_format(format)      
        seq_file = os.path.join(det_root, seq_name+'.txt')
        seq_dets, flag, frame_det_idx = load_detection(seq_file, format=format, cat=cat_list) 	# load detection
        NDT_cache_path = os.path.join(cfg.NDT_cache_path,seq_name)
        os.system(f"mkdir -p {NDT_cache_path}")
        if not flag:    continue  # no detection        
        ##Pcd info
        pcd_info_seq_2d = pcd_info_seq_preprocess(pcd_info[seq_name], pcd_db, len(frame_det_idx), frame_det_idx)
        dets_frame = process_dets(get_frame_det(seq_dets, format=format))
        
        pcd_info_seq = []
        for sublist in pcd_info_seq_2d:
            for element in sublist:
                pcd_info_seq.append(element)     
        pcds = load_dense_byinfo(pcd_info_seq)
        
        ## For Multi threading 
        NDT_Voxels = []
        MT_pcd = pcds
        MT_dets = dets_frame
        MT_outpath = []
        for info in pcd_info_seq:
            frame_str = str(info['obj']['frame_id']).zfill(6)
            det_idx = str(info['obj_det_idx']).zfill(4)
            class_str = info['obj']['obj_type'].lower()
            MT_outpath.append(os.path.join(NDT_cache_path, f"{frame_str}_{det_idx}_{class_str}.pkl"))
            
        # for i in range(3000):
        #     print(MT_outpath[i])
        #     v1,_,_=NDT_voxelize(MT_pcd[i],MT_dets[i],cfg.NDT_cfg,True)
        # # draw_NDT_voxel(v1,random=False)
        # exit()
        
        assert len(MT_pcd) == len(MT_dets) 
        assert len(MT_dets) == len(MT_outpath)
        CHUNK=64
        num_groups = (len(MT_pcd) + CHUNK - 1) // CHUNK
        MT_DATAS=[[] for i in range(num_groups)]
        RESULTS=[[] for i in range(num_groups)]       
        for i in range(len(MT_pcd)):          
            MT_DATAS[i//CHUNK].append((MT_pcd[i],MT_dets[i],MT_outpath[i]))
            
        print(f"Start pre calculate NDT of detection in {seq_name}.\nTotal Objects:{len(MT_pcd)}, CHUNK SIZE: {CHUNK}")
        # pool = [Pool() for _ in range(CHUNK)]
        with Pool(CHUNK) as pool:
            for ci, chunkdata in enumerate(tqdm(MT_DATAS)):            
                if(len(chunkdata)>0):
                    ## Multi threading init                
                    ## Multi threading and run NDT_voxelize(non blocking)
                    thread=[pool.apply_async(NDT_voxelize, (chunkdata[i][0],chunkdata[i][1],cfg.NDT_cfg))  for i in range(len(chunkdata))]
                    result=[None for _ in range(len(chunkdata))]
                    ## Collect result(blocking)
                    for i in range(len(chunkdata)):
                        result[i],_,_ = thread[i].get(timeout=50)  #valid,invalid,all ; collect valid only
                    ## Clean  pool
                    # for p in pool:  p.close() 
                    ## Update result for  det/trk
                    for i in range(len(chunkdata)):
                        RESULTS[ci].append((result[i],chunkdata[i][2]))
        end=time.time()       
        print(f"Writing NDT cache file...")
        for r1 in RESULTS:
            for r2 in r1:
                with open(r2[1], 'wb') as file:
                    pickle.dump(r2[0], file)
        with open(os.path.join(cfg.NDT_cache_path, "log.txt"), 'a') as file:
            objnum=len(MT_pcd)
            t1=round(end-start,1)
            ops=round(objnum/t1,1)
            msg = f"{seq_name}: {objnum} objects; {t1} seconds; OPS: {ops}; ChunkSize: {CHUNK}\n"
            print(msg)
            file.write(msg)   
    return 

def main(args):
    start=time.time()
    # load config files
    config_path = args.config
    cfg, settings_show = Config(config_path)

    # overwrite split and detection method
    if args.split is not '': cfg.split = args.split
    if args.det_name is not '': cfg.det_name = args.det_name
    print("Start pre calculate NDT of detection")
    assert cfg.NDT_flag==True, "NDT_flag should be True"
    #seq
    if ("seq_eval" not in cfg):
        cfg.seq_eval=[]

    #cat to capitalize
    cfg.cat_list = [cat.capitalize() for cat in cfg.cat_list]
    timestr = datetime.now().strftime("%m-%d-%H-%M") 
    DUPLICATE = True  
    NDT_cache_path = os.path.join(cfg.NDT_cache_root, cfg.NDT_cache_name)
    if((not DUPLICATE) and os.path.exists(NDT_cache_path)):
        print(f"NDT cache {NDT_cache_path} exist;")               
        NDT_cache_path = os.path.join(cfg.NDT_cache_root,f"{cfg.NDT_cache_name}_{timestr}")
        print(f"Write NDT cache to {NDT_cache_path}")               
    os.system(f"mkdir -p {NDT_cache_path}")   
    cfg.NDT_cache_path = NDT_cache_path
    
    os.system(f"cp {config_path} {NDT_cache_path}/config.yml")
    
    main_inner(cfg)
    end=time.time()
    
    with open(os.path.join(cfg.NDT_cache_path, "log.txt"), 'a') as file:
        msg=f"Total time: {round(end-start,1)} seconds\n"
        print(msg)
        file.write(msg)
    
    
if __name__ == '__main__':

    args = parse_args()
    main(args)