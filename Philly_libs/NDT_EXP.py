import numpy as np
import open3d as o3d
import os
# import click
import sys
import pdb
import Philly_libs.philly_io as io_utils
import Philly_libs.plot as plot
from Philly_libs.NDT import *
from Philly_libs.philly_utils import DictToObj

def get_label(file):
    lines = np.genfromtxt(file, delimiter=' ', dtype=float)
    trackid=[int(l[1]) for l in lines]
    return trackid

def get_obj_by_trackid(root,label_path,seq,vis=False):
    gtdb=os.path.join(root,"gt_database/")
    pkl_path=os.path.join(root,"info.pkl")
    pkl=io_utils.read_pkl(pkl_path)
    label_path = os.path.join(label_path,seq+'.txt')
    trackid = get_label(label_path)
    mode='mae_dense_path'
    tracks_obj = {}
    
    for i,x in enumerate(pkl[seq]):
        dense_file=os.path.join(gtdb,x[mode])            
        # print(dense_file)     
        x['obj']['track_id'] = trackid[i]
        obj_type=x['obj']['obj_type']
        x['obj']['idx'] = i
        
        if(not obj_type in tracks_obj):
            tracks_obj[obj_type] = {}
        if(not trackid[i] in tracks_obj[obj_type]):
            tracks_obj[obj_type][trackid[i]] = []
        tracks_obj[obj_type][trackid[i]].append(x)
        
        if(vis):
            dense=io_utils.read_points_from_bin(dense_file,unique=True)
            bbox=x['obj']['box3d']
            bbox['x'],bbox['y'],bbox['z']=0,0,0
            bbox['roty']=0
            plot.draw_pcd_and_bbox_v2(dense,bbox)     
    return tracks_obj

def write_trackid_info(tracks_obj,EXP_PATH):
    os.system(f"mkdir -p {EXP_PATH}")
    os.system(f"mkdir -p {os.path.join(EXP_PATH,'pcd')}")
    io_utils.write_pkl(tracks_obj,os.path.join(EXP_PATH,"info_by_trackid.pkl"))
    return

def get_NDT_by_trackid(tracks_obj,root_dense,NDT_cache_path,EXP_PATH):
    root_dense = os.path.join(root_dense,'gt_database/')
    cls=tracks_obj.keys()
    # cls=['cyclist']
    for c in cls:
        trks=tracks_obj[c]
        trackid_exp_path=os.path.join(EXP_PATH,c)
        os.system(f"mkdir -p {trackid_exp_path}")
        cnt=0        
        for trackid in trks:
            track = trks[trackid]
            print(f"processing {c} trackid:{trackid}, {cnt}/{len(trks)}")
            for i,x in enumerate(track):
                s1=str(x['velodyne_idx']).zfill(6)
                s2=str(x['obj_det_idx']).zfill(4)
                s3=c
                old_NDT_path=os.path.join(NDT_cache_path,f"{s1}_{s2}_{s3}.pkl")
                new_NDT_path=os.path.join(trackid_exp_path,f"{str(trackid).zfill(4)}_{str(i).zfill(4)}.pkl")
                os.system(f"cp -r {old_NDT_path} {new_NDT_path}")
                # dense_file=os.path.join(root_dense,x['mae_dense_path'])
                # dense=io_utils.read_points_from_bin(dense_file,unique=True)
                # bbox=x['obj']['box3d']
                # bbox['x'],bbox['y'],bbox['z'],bbox['roty']=0,0,0,0
                # plot.draw_pcd_and_bbox_v2(dense,bbox)  
            cnt+=1                
    return
    
def merge_NDT_of_track(tracks_obj,root_dense,EXP_PATH,max_occ=1):
    root_dense = os.path.join(root_dense,'gt_database/')
    cls=tracks_obj.keys()
    cls=['car']
    for c in cls:
        trks=tracks_obj[c]
        trackid_exp_path=os.path.join(EXP_PATH,c)
        merged_track_exp_path=os.path.join(EXP_PATH,f"{c}_all")      
        os.system(f"mkdir -p {merged_track_exp_path}")
        merged_member={}
        for trackid in trks:
            track = trks[trackid]
            print(f"merging {c} trackid:{trackid} len:{len(trks)}")
            trackid_str=str(trackid).zfill(4)
            merged_member[trackid_str]=[]            
            merged_dense_path=os.path.join(merged_track_exp_path,f"{trackid_str}.pkl")
            merged_dense = np.empty((0, 4)).astype(np.float32) 
            for i,x in enumerate(track):
                if(x['valid'] and x['obj']['occlusion']<=max_occ):
                    dense_file=os.path.join(root_dense,x['mae_dense_path'])
                    dense=io_utils.read_points_from_bin(dense_file,unique=True)                    
                    dense = np.hstack((dense, np.zeros((dense.shape[0], 1)))).astype(np.float32)
                    merged_dense = np.concatenate((merged_dense, dense), axis=0)     
                    merged_member[trackid_str].append(i)
            merged_dense.tofile(merged_dense_path)
            md2 = io_utils.read_points_from_bin(merged_dense_path,unique=True)
        io_utils.write_pkl(merged_member,os.path.join(merged_track_exp_path,"merged_member.pkl"))            
    return

def merged_pcd_test(tracks_obj, EXP_PATH):
    cls=tracks_obj.keys()
    cls=['car']
    RANDOM_SAMPLE=4096

    for c in cls:
        trks=tracks_obj[c]
        trackid_exp_path=os.path.join(EXP_PATH,c)
        merged_track_exp_path=os.path.join(EXP_PATH,f"{c}_all")      
        merged_member=io_utils.read_pkl(os.path.join(merged_track_exp_path,"merged_member.pkl"))
        for trackid in merged_member:
            merged_idx = merged_member[trackid]            
            print(f"track:{trackid}, merge {len(merged_idx)} frames")
            if(len(merged_idx)<=0):
                continue
            track = trks[int(trackid)]
            track_member = [track[i] for i in merged_idx]
            merged_dense_path=os.path.join(merged_track_exp_path,f"{trackid}.pkl")
            merged_dense = io_utils.read_points_from_bin(merged_dense_path,unique=True)   
            bbox=track_member[0]['obj']['box3d']
            bbox['x'],bbox['y'],bbox['z'],bbox['roty']=0,0,0,0
            bbox_obj=DictToObj(bbox)
            sampled_dense = merged_dense[np.random.choice(merged_dense.shape[0], RANDOM_SAMPLE, replace=False)]
            dense_NDT = NDT_voxelize(sampled_dense,bbox_obj)[0]
            io_utils.write_pkl(dense_NDT,os.path.join(merged_track_exp_path,f"{trackid}_NDT.pkl"))    
            continue
    return

def NDT_exp_frame_to_merge(tracks_obj, EXP_PATH):
    cls=tracks_obj.keys()
    cls=['car']
    RANDOM_SAMPLE=4096
    avg_score_same={}
    
    NDT_OF_TRACK={}
    NDT_FRAME_OF_TRACK={}
    INFO_OF_TRACK={}
    none_trk=[]
    for c in cls:
        trks=tracks_obj[c]
        trackid_exp_path=os.path.join(EXP_PATH,c)
        merged_track_exp_path=os.path.join(EXP_PATH,f"{c}_all")      
        merged_member=io_utils.read_pkl(os.path.join(merged_track_exp_path,"merged_member.pkl"))
        cnt=0
        for trackid in merged_member:
            # if(cnt>5): break
            merged_idx = merged_member[trackid]            
            if(len(merged_idx)<=0):
                none_trk.append(trackid)
                continue
            track = trks[int(trackid)]
            dense_NDT = io_utils.read_pkl(os.path.join(merged_track_exp_path,f"{trackid}_NDT.pkl"))   
            # pdb.set_trace()
            frames=[]
            for idx in merged_idx:
                frame = io_utils.read_pkl(os.path.join(trackid_exp_path,f"{trackid}_{str(idx).zfill(4)}.pkl")) 
                frames.append(frame)
                
            NDT_OF_TRACK[trackid]=dense_NDT 
            NDT_FRAME_OF_TRACK[trackid]=frames
            INFO_OF_TRACK[trackid]=track
            cnt+=1
            
            continue
        
    t_list=list(merged_member.keys())
    for nt in none_trk:
        print(f'remove {nt}')
        t_list.remove(nt)
    # t_list=t_list[:5]
    score_map={}
    avg_score_diff={} 
    avg_score_same={}
    frame_map={}
    for trackid_i in t_list:
        score_map[trackid_i]={}
        total_score=0
        frame_score=[[] for i in range(len(NDT_FRAME_OF_TRACK[trackid_i]))]
        print(f"track:{trackid_i}")
        for trackid_j in t_list:
            score=0
            for fi,frame in enumerate(NDT_FRAME_OF_TRACK[trackid_i]):
                s = NDT_score(frame,NDT_OF_TRACK[trackid_j])
                frame_score[fi].append(s)
                score += s                
            score/=len(NDT_FRAME_OF_TRACK[trackid_i])
            if(trackid_i!=trackid_j):
                total_score+=score
            score_map[trackid_i][trackid_j]=score
        frame_map[trackid_i]=frame_score
        avg_score_diff[trackid_i] = total_score/(len(t_list)-1)
        avg_score_same[trackid_i] = score_map[trackid_i][trackid_i]
    
    print(avg_score_diff)
    print('==================================')
    print(avg_score_same)
    print('==================================')
    print(score_map)
    
    analys_path=os.path.join(EXP_PATH,"analysis")      
    
    io_utils.write_pkl(avg_score_diff,os.path.join(analys_path,"avg_frame_merge_diff.pkl"))
    io_utils.write_pkl(avg_score_same,os.path.join(analys_path,"avg_frame_merge_same.pkl"))
    io_utils.write_pkl(score_map,os.path.join(analys_path,"map_frame_merge_score.pkl"))
    io_utils.write_pkl(frame_map,os.path.join(analys_path,"frame_merge_score_all_frame.pkl"))
    return

def NDT_exp_merge_to_merge(tracks_obj, EXP_PATH):
    cls=tracks_obj.keys()
    cls=['car']
    RANDOM_SAMPLE=4096
    avg_score_same={}
    
    NDT_OF_TRACK={}
    SAMPLE_PCD_OF_TRACK={}
    INFO_OF_TRACK={}
    none_trk=[]
    for c in cls:
        trks=tracks_obj[c]
        trackid_exp_path=os.path.join(EXP_PATH,c)
        merged_track_exp_path=os.path.join(EXP_PATH,f"{c}_all")      
        merged_member=io_utils.read_pkl(os.path.join(merged_track_exp_path,"merged_member.pkl"))
        for trackid in merged_member:
            merged_idx = merged_member[trackid]            
            if(len(merged_idx)<=0):
                none_trk.append(trackid)
                continue
            track = trks[int(trackid)]
            dense_NDT = io_utils.read_pkl(os.path.join(merged_track_exp_path,f"{trackid}_NDT.pkl"))   
            # pdb.set_trace()
            NDT_OF_TRACK[trackid]=dense_NDT 
            SAMPLE_PCD_OF_TRACK[trackid]=dense_NDT
            INFO_OF_TRACK[trackid]=track
            continue
        
    t_list=list(merged_member.keys())
    for nt in none_trk:
        t_list.remove(nt)
    # t_list=t_list[:5]
    score_map={}
    avg_score_diff={}
    avg_score_same={}
    
    for trackid_i in t_list:
        score_map[trackid_i]={}
        total_score=0
        print(f"track:{trackid_i}")
        for trackid_j in t_list:
            score=NDT_score(SAMPLE_PCD_OF_TRACK[trackid_i],NDT_OF_TRACK[trackid_j])
            if(trackid_i!=trackid_j):
                total_score+=score
            score_map[trackid_i][trackid_j]=score
        avg_score_diff[trackid_i]=total_score/(len(t_list)-1)
        avg_score_same[trackid_i]=score_map[trackid_i][trackid_i]
    print(avg_score_diff)
    print('==================================')
    print(avg_score_same)
    print('==================================')
    print(score_map)
    
    analys_path=os.path.join(EXP_PATH,"analysis")      
    io_utils.write_pkl(avg_score_diff,os.path.join(analys_path,"avg_merge_merge_diff.pkl"))
    io_utils.write_pkl(avg_score_same,os.path.join(analys_path,"avg_merge_merge_same.pkl"))
    io_utils.write_pkl(score_map,os.path.join(analys_path,"map_merge_merge_score.pkl"))
    return

def NDT_filter(x):
    return x['valid'] and x['obj']['occlusion']<=1


    
if __name__ == '__main__':
    seq=21
    seq = str(seq).zfill(4)
    
    root_gtdb="/home/philly12399/philly_ssd/point_mae/gt_db/kitti/diff0_gtdb/"   
    root_dense="/home/philly12399/philly_ssd/point_mae/output/gtdet/dense_128_all/" 
    label_path="/home/philly12399/philly_ssd/KITTI_tracking/training/label_02/"
    NDT_cache_path="/home/philly12399/philly_ssd/ab3dmot/NDT_pkl/gtdet/cache-128/"
    EXP_PATH="/home/philly12399/philly_ssd/NDT_EXP/"
    
    EXP_PATH = os.path.join(EXP_PATH,seq)  
    NDT_cache_path = os.path.join(NDT_cache_path,seq)
    os.system(f"mkdir -p {EXP_PATH}")
    overwrite=False
    INFO_PATH=os.path.join(EXP_PATH,"info_by_trackid.pkl")
    # if(overwrite or (not os.path.exists(INFO_PATH))):
    #     tracks_obj = get_obj_by_trackid(root_dense,label_path,seq)
    #     write_trackid_info(tracks_obj,EXP_PATH)
    #     print("create new trackid info")
    # else:
    #     tracks_obj = io_utils.read_pkl(INFO_PATH)
    #     print("load trackid info")
        
    # pdb.set_trace()
    # 把原本的NDT cache照trackid分類
    # get_NDT_by_trackid(tracks_obj,root_dense,NDT_cache_path,EXP_PATH)
    # merge_NDT_of_track(tracks_obj,root_dense,EXP_PATH)
    # NDT_exp_merge_to_merge(tracks_obj,EXP_PATH)
    # NDT_exp_frame_to_merge(tracks_obj,EXP_PATH)
    
    # analys="/home/philly12399/philly_ssd/NDT_EXP/0021/analysis/"
    # x="avg_frame_merge_same"
    # io_utils.pkl_to_txt(os.path.join(analys,f"{x}.pkl"),os.path.join(analys,f"{x}.txt"))
  