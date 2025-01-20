import warnings, numpy as np, os
from  Philly_libs.philly_io import *
import pdb
################## loading
def load_detection(file, format="",cat = []):
    # load from raw file 

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if(format.lower()=="wayside"): str2id = {'car':1,'cyclist':2}
        elif(format.lower()=="kitti"): str2id = {'car':2,'cyclist':3}	
        catid=[]
        for i in range(len(cat)):
            catid.append(str2id[cat[i].lower()])
            
        if(format.lower()=="wayside" or format.lower()=="kitti" ):
            dets1 = np.genfromtxt(file, delimiter=' ', dtype=float) #for label
            dets2 = np.genfromtxt(file, delimiter=' ', dtype=str) #for class only
            frame_num = 0
            for i in range(len(dets1)):
                frame_num = max(frame_num, int(dets1[i][0]))
            # bind with gtdb dense
            frame_cnt = [0 for i in range(frame_num+1)]
            frame_det_idx = [[] for i in range(frame_num+1)]
   
            l = []
            for i in range(len(dets1)):
                frame=int(dets1[i][0])
                dets2[i][2] = dets2[i][2].lower()
                # if(dets2[i][2] in cls_map): #truck to car
                #     dets2[i][2] = cls_map[dets2[i][2]]	  
                # pdb.set_trace()
                if(dets2[i][2] not in str2id):
                    frame_cnt[frame]+=1
                    # assert False, "You have to use gtdet diff0 as NDT preprocess(no filtered)."
                    continue    
                dets1[i][2] = str2id[dets2[i][2]]
                if(dets1[i][2] in catid):
                    l.append(dets1[i])
                    frame_det_idx[frame].append(frame_cnt[frame])
                frame_cnt[frame]+=1                    
            dets = np.array(l)
        else: assert False
    if len(dets.shape) == 1: dets = np.expand_dims(dets, axis=0) 	
    if dets.shape[1] == 0:		# if no detection in a sequence
        return [], False,None
    else:
        return dets, True,frame_det_idx


def get_frame_det(dets_all, format=""):
    if(format.lower()=="wayside" or format.lower()=="kitti"):
        # matched_dets = dets_all[dets_all[:, 0] == frame , 0:]	
        matched_dets = dets_all        
        ori_array = matched_dets[:, 5].reshape((-1, 1))		# orientation
        other_array = matched_dets[:, [2,6,7,8,9,-1]] # other information, e.g, 2D box, ...
        # additional_info = np.concatenate((ori_array, other_array), axis=1) # ori,class,2dbox,confidence
        # get 3D box
        if(format.lower()=="wayside"):
            dets = matched_dets[:, [12,11,10,13,14,15,16]]	 #
        else:
            dets = matched_dets[:, [10,11,12,13,14,15,16]]		
        # dets_frame = {'dets': dets, 'info': additional_info}
        return dets
    else: assert False

