# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import warnings, numpy as np, os
from xinshuo_io import mkdir_if_missing, load_txt_file, save_txt_file

################## loading
def load_detection(file, format="",cat = ""):
    # load from raw file 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if(format.lower()=="wayside"): str2id = {'car':1,'cyclist':2}
        elif(format.lower()=="kitti"): str2id = {'car':2,'cyclist':3}		
  
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
                if(dets2[i][2] not in str2id):
                    frame_cnt[frame]+=1
                    continue    
                dets1[i][2] = str2id[dets2[i][2]]
                if(dets1[i][2] == str2id[cat.lower()]):
                    l.append(dets1[i])
                    frame_det_idx[frame].append(frame_cnt[frame])
                frame_cnt[frame]+=1                    
            dets = np.array(l)
        elif(format.lower()=="ab3dmot"):
            dets = np.loadtxt(file, delimiter=',') 	# load detections, N x 15
        else: assert False
  
    if len(dets.shape) == 1: dets = np.expand_dims(dets, axis=0) 	
    if dets.shape[1] == 0:		# if no detection in a sequence
        return [], False,None
    else:
        return dets, True,frame_det_idx


def get_frame_det(dets_all, frame, format=""):
    if(format.lower()=="wayside" or format.lower()=="kitti"):
        # get irrelevant information associated with an object, not used for associationg
        matched_dets = dets_all[dets_all[:, 0] == frame , 0:]	
        ori_array = matched_dets[:, 5].reshape((-1, 1))		# orientation
        other_array = matched_dets[:, [2,6,7,8,9,-1]] # other information, e.g, 2D box, ...
        additional_info = np.concatenate((ori_array, other_array), axis=1) # ori,class,2dbox,confidence
        # get 3D box

        if(format.lower()=="wayside"):
            dets = matched_dets[:, [10,12,11,13,15,14,16]] #switch height,z to width,y
        else:
            dets = matched_dets[:, [10,11,12,13,14,15,16]]		
        dets_frame = {'dets': dets, 'info': additional_info}
        return dets_frame
    elif(format.lower()=="ab3dmot"):
        # get irrelevant information associated with an object, not used for associationg
        ori_array = dets_all[dets_all[:, 0] == frame, -1].reshape((-1, 1))		# orientation
        other_array = dets_all[dets_all[:, 0] == frame, 1:7] 					# other information, e.g, 2D box, ...

        additional_info = np.concatenate((ori_array, other_array), axis=1)		

        # get 3D box
        dets = dets_all[dets_all[:, 0] == frame, 7:14]		

        dets_frame = {'dets': dets, 'info': additional_info}
        return dets_frame
    else: assert False

def load_highlight(file):
    # load file with each line containing seq_id, frame_id, ID, error_type
    # used to highlight errors in the video visualization, such as IDS, FP
    # but cannot be used to highlight FRAG (next frame) and FN now

    highlight, _ = load_txt_file(file)
    data_dict = dict()
    for data_tmp in highlight:
        
        # parse data in each line, seq_id, frame_id, ID, error_type
        seq_id, frame_id, id_tmp, err_type = data_tmp.split(', ')
        seq_id, frame_id, id_tmp = int(seq_id), int(frame_id), int(id_tmp)
        
        # create entry in the dictionary, with key -> seq, 
        # val -> dict{key -> frame, value -> dict{key -> ID, value -> err}}
        if seq_id not in data_dict.keys():
            data_dict[seq_id] = dict()
        if frame_id not in data_dict[seq_id]:
            data_dict[seq_id][frame_id] = dict()
        assert id_tmp not in data_dict[seq_id][frame_id], 'error, each ID should not be highlighted twice'

        # assign the err_type to the ID
        data_dict[seq_id][frame_id][id_tmp] = err_type
        
    return data_dict

#################### saving

def get_saving_dir(eval_dir_dict, seq_name, save_dir, num_hypo):

    # create dir and file for saving
    eval_file_dict, save_trk_dir = dict(), dict()
    for index in range(num_hypo):
        eval_file_dict[index] = os.path.join(eval_dir_dict[index], seq_name + '.txt')
        eval_file_dict[index] = open(eval_file_dict[index], 'w')
        save_trk_dir[index] = os.path.join(save_dir, 'trk_withid_%d' % index, seq_name); mkdir_if_missing(save_trk_dir[index])
    affinity_dir = os.path.join(save_dir, 'affi', seq_name); #mkdir_if_missing(affinity_dir)
    affinity_vis = os.path.join(save_dir, 'affi_vis', seq_name);# mkdir_if_missing(affinity_vis)

    return eval_file_dict, save_trk_dir, affinity_dir, affinity_vis
import pdb
def save_results(res, save_trk_file, eval_file, det_id2str, frame, score_threshold, format=""):

    # box3d in the format of h, w, l, x, y, z, theta in camera coordinate
    bbox3d_tmp, id_tmp, ori_tmp, type_tmp, bbox2d_tmp_trk, conf_tmp = \
        res[0:7], res[7], res[8], det_id2str[res[9]], res[10:14], res[14] 		
    frame = int(res[15])
    if(format.lower() == "wayside"):
        bbox3d_tmp =  bbox3d_tmp[[0,2,1,3,5,4,6]] #switch height,z , width,y back (in get_frame_det)
    # save in detection format with track ID, can be used for dection evaluation and tracking visualization
    str_to_srite = '%s -1 -1 %f %f %f %f %f %f %f %f %f %f %f %f %f %d\n' % (type_tmp, ori_tmp,
        bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
        bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[6], conf_tmp, id_tmp)
    save_trk_file.write(str_to_srite)

    # save in tracking format, for 3D MOT evaluation
    if conf_tmp >= score_threshold:
        str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (frame, id_tmp, 
            type_tmp, ori_tmp, bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
            bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[6], conf_tmp)
        eval_file.write(str_to_srite)

def save_affinity(affi_data, save_path):
    ######### save txt files for faster check, with aligned formatting

    # compute the number of digit for the largest values for better alignment of saving
    min_val, max_val = np.min(affi_data), np.max(affi_data)
    biggest = max(abs(min_val), abs(max_val))
    num_digit = 0
    while True:
        if biggest < 1: break
        num_digit += 1
        biggest = biggest / 10.0
    
    # see if there is a negative sign, so need to a one more digit 
    negative = False
    if min_val < 0: negative = True
    if negative: num_digit += 1

    # add digits depending on the decimals we want to preserve
    decimals = 2
    num_digit += decimals + 1 		# meaning that we want to preserve the dot plus the decimals

    # save
    fmt = '%%%d.%df' % (num_digit, decimals)
    np.savetxt(save_path, affi_data, fmt=fmt, delimiter=', ')

def combine_files(file_list, save_path, sort=True):
    # combine txt files and sort them in frame order, used to collect results from 
    # different class categories

    # collect all files
    data_all = list()
    for file_tmp in file_list:
        data, num_lines = load_txt_file(file_tmp)
        data_all += data

    # sort based on frame number
    if sort:
        data_all.sort(key = lambda x: int(x.split(' ')[0]))

    save_txt_file(data_all, save_path)
