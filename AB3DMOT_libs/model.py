# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import numpy as np, os, copy, math
from AB3DMOT_libs.box import Box3D
from AB3DMOT_libs.matching import data_association
from AB3DMOT_libs.kalman_filter import KF
from AB3DMOT_libs.vis import vis_obj
from Philly_libs.philly_matching import data_association as data_association_philly
from Philly_libs.TrackBuffer import TrackBuffer , KF_predict
from Philly_libs.NDT import draw_NDT_voxel
from Philly_libs.philly_io import read_pkl
from Philly_libs.kitti_utils import *
from Philly_libs.philly_utils import interpolate_bbox
from xinshuo_miscellaneous import print_log
from xinshuo_io import mkdir_if_missing
import time
from multiprocessing import Pool
import pdb
import pickle
np.set_printoptions(suppress=True, precision=3)

# A Baseline of 3D Multi-Object Tracking
class AB3DMOT(object):			  	
    def __init__(self, cfg, cat, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=None, log=None, ID_init=0, seq_name = ""):                    

        # vis and log purposes
        self.img_dir = img_dir
        self.vis_dir = vis_dir
        self.hw = hw
        self.log = log

        # counter
        self.track_buf = []
        self.frame_count = 0
        self.ID_count = [ID_init]
        self.id_now_output = []

        # config
        self.cat = cat
        self.ego_com = cfg.ego_com 			# ego motion compensation
        self.calib = calib
        self.oxts = oxts


        assert cat.lower() in cfg.base_param, f"no param of class {cat.lower()}"
        param=copy.deepcopy(cfg.base_param[cat.lower()])
        self.get_param(cfg, cat, param)
        self.print_param()
  
        self.label_format = cfg.label_format.lower()        
        self.label_coord = cfg.label_coord
        self.history = 1
        self.kf_initial_speed=cfg.kf_initial_speed
        
        self.output_kf_cls = cfg.output_kf_cls
        self.output_mode =  cfg.output_mode.lower()
        assert self.output_mode == 'kf' or self.output_mode == 'interpolate'
        Box3D.set_label_format(self.label_format)
        ##NDT
        self.NDT_flag = cfg.NDT_flag
        self.NDT_cfg = None
        self.NDT_thres = None
        if(self.NDT_flag):
            self.NDT_cfg  = cfg.NDT_cfg
            self.NDT_cache_path = os.path.join(cfg.NDT_cache_root,cfg.NDT_cache_name,seq_name)                    
            if(not os.path.exists(self.NDT_cache_path)):
                print(f"Load NDT cache failed, {self.NDT_cache_path} not exists")
                assert False
            self.NDT_thres = cfg.NDT_thres[cat.lower()]
                                
          # debug
        # self.debug_id = 2
        self.debug_id = None
        self.debug_id_new=1
        self.debugger=[]
        
        self.global_cnt=0
        # self.two_stage=False
        self.two_stage=cfg.two_stage
        # self.stage2_param = cfg.base_param[cat.lower()] 
        if("stage2_param" in cfg):
            self.stage2_param = copy.deepcopy(cfg.stage2_param[cat.lower()])
            if self.stage2_param['metric'] in ['dist_3d', 'dist_2d', 'm_dis']: 
                if(self.stage2_param['thres'] >=0 ):
                    self.stage2_param['thres']*= -1
        if("conf_thres" in cfg):
            self.conf_thres= cfg.conf_thres
        else:
            self.conf_thres=-9999
    def get_param(self, cfg, cat, param=None):
        # get parameters for each dataset
        if(param !=None):
            algm, metric, thres, min_hits, max_age = param['algm'], param['metric'], param['thres'], param['min_hits'], param['max_age']
        else:
            if cfg.dataset == 'KITTI':
                if cfg.det_name == 'pvrcnn':				# tuned for PV-RCNN detections
                    if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.2, 3, 2
                    elif cat == 'Pedestrian': 	algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 4 		
                    elif cat == 'Cyclist': 		algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 2, 3, 4
                    else: assert False, 'error'
                elif cfg.det_name == 'pointrcnn':			# tuned for PointRCNN detections
                    if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.2, 3, 2
                    elif cat == 'Pedestrian': 	algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 4 		
                    elif cat == 'Cyclist': 		algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 2, 3, 4
                    else: assert False, 'error'
                elif cfg.det_name == 'deprecated':			
                    if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 6, 3, 2
                    elif cat == 'Pedestrian': 	algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 1, 3, 2		
                    elif cat == 'Cyclist': 		algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 6, 3, 2
                    else: assert False, 'error'
                else:			# GT
                    if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.2, 3, 2
                    elif cat == 'Pedestrian': 	algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 4 		
                    elif cat == 'Cyclist': 		algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 2, 3, 4
                    else: assert False, 'error'    
                # else: assert False, 'error'
            elif cfg.dataset == 'nuScenes':
                if cfg.det_name == 'centerpoint':		# tuned for CenterPoint detections
                    if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 2
                    elif cat == 'Pedestrian': 	algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.5, 1, 2
                    elif cat == 'Truck': 		algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 2
                    elif cat == 'Trailer': 		algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.3, 3, 2
                    elif cat == 'Bus': 			algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 2
                    elif cat == 'Motorcycle':	algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.7, 3, 2
                    elif cat == 'Bicycle': 		algm, metric, thres, min_hits, max_age = 'greedy', 'dist_3d',    6, 3, 2
                    else: assert False, 'error'
                elif cfg.det_name == 'megvii':			# tuned for Megvii detections
                    if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.5, 1, 2
                    elif cat == 'Pedestrian': 	algm, metric, thres, min_hits, max_age = 'greedy', 'dist_3d',    2, 1, 2
                    elif cat == 'Truck': 		algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.2, 1, 2
                    elif cat == 'Trailer': 		algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.2, 3, 2
                    elif cat == 'Bus': 			algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.2, 1, 2
                    elif cat == 'Motorcycle':	algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.8, 3, 2
                    elif cat == 'Bicycle': 		algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.6, 3, 2
                    else: assert False, 'error'
                elif cfg.det_name == 'deprecated':		
                    if cat == 'Car': 			metric, thres, min_hits, max_age = 'dist', 10, 3, 2
                    elif cat == 'Pedestrian': 	metric, thres, min_hits, max_age = 'dist',  6, 3, 2	
                    elif cat == 'Bicycle': 		metric, thres, min_hits, max_age = 'dist',  6, 3, 2
                    elif cat == 'Motorcycle':	metric, thres, min_hits, max_age = 'dist', 10, 3, 2
                    elif cat == 'Bus': 			metric, thres, min_hits, max_age = 'dist', 10, 3, 2
                    elif cat == 'Trailer': 		metric, thres, min_hits, max_age = 'dist', 10, 3, 2
                    elif cat == 'Truck': 		metric, thres, min_hits, max_age = 'dist', 10, 3, 2
                    else: assert False, 'error'
            elif cfg.dataset == 'Wayside':
                # if cfg.det_name == 'pvrcnn':				# tuned for PV-RCNN detections
                if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.2, 3, 10		
                elif cat == 'Cyclist': 		algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 2, 3, 4
                else: assert False, 'error'
            else: assert False, 'no such dataset'

        # add negative due to it is the cost
        if metric in ['dist_3d', 'dist_2d', 'm_dis']: thres *= -1	
        self.algm, self.metric, self.thres, self.max_age, self.min_hits = \
            algm, metric, thres, max_age, min_hits
        print(f'algm:{self.algm}, metric:{self.metric}, thres:{self.thres}, max_age:{self.max_age}, min_hits:{self.min_hits}')
        # define max/min values for the output affinity matrix
        if self.metric in ['dist_3d', 'dist_2d', 'm_dis']: self.max_sim, self.min_sim = 0.0, -100.
        elif self.metric in ['iou_2d', 'iou_3d']:   	   self.max_sim, self.min_sim = 1.0, 0.0
        elif self.metric in ['giou_2d', 'giou_3d']: 	   self.max_sim, self.min_sim = 1.0, -1.0

    def print_param(self):
        print_log('\n\n***************** Parameters for %s *********************' % self.cat, log=self.log, display=False)
        print_log('matching algorithm is %s' % self.algm, log=self.log, display=False)
        print_log('distance metric is %s' % self.metric, log=self.log, display=False)
        print_log('distance threshold is %f' % self.thres, log=self.log, display=False)
        print_log('min hits is %f' % self.min_hits, log=self.log, display=False)
        print_log('max age is %f' % self.max_age, log=self.log, display=False)
        print_log('ego motion compensation is %d' % self.ego_com, log=self.log, display=False)

    def process_dets(self, dets):
        # convert each detection into the class Box3D 
        # inputs: 
        # 	dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]

        dets_new = []
        for det in dets:
            det_tmp = Box3D.array2bbox_raw(det)
            dets_new.append(det_tmp)
        return dets_new
    
    def move_dets_origin_axis(self, frame, dets):
        dets_new = []
        for det in dets:
            d = [det.x, det.y, det.z, det.ry]
            d_origin = move_to_origin_axis(self.oxts, self.calib, self.label_coord, frame, d)    
            bbox_origin = Box3D.array2bbox_raw([det.h, det.w, det.l] + d_origin)
            dets_new.append(bbox_origin)
        return dets_new
    
    def within_range(self, theta):
        # make sure the orientation is within a proper range

        if theta >= np.pi: theta -= np.pi * 2    # make the theta still in the range
        if theta < -np.pi: theta += np.pi * 2

        return theta

    def orientation_correction(self, theta_pre, theta_obs):
        # update orientation in propagated tracks and detected boxes so that they are within 90 degree
        
        # make the theta still in the range
        theta_pre = self.within_range(theta_pre)
        theta_obs = self.within_range(theta_obs)

        # if the angle of two theta is not acute angle, then make it acute
        if abs(theta_obs - theta_pre) > np.pi / 2.0 and abs(theta_obs - theta_pre) < np.pi * 3 / 2.0:     
            theta_pre += np.pi       
            theta_pre = self.within_range(theta_pre)

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(theta_obs - theta_pre) >= np.pi * 3 / 2.0:
            if theta_obs > 0: theta_pre += np.pi * 2
            else: theta_pre -= np.pi * 2

        return theta_pre, theta_obs        
    

    def prediction(self, frame, history = 5):
        # get predicted locations from existing tracks

        pred = []
        for t in range(len(self.track_buf)):
            # propagate locations
            trk = self.track_buf[t]
            pred_of_trk=[]      
            kf_history = list(reversed(trk.kf_buffer[-history:]))
            time_history =  list(reversed(trk.time_stamp[-history:]))
            for i, kf in enumerate(kf_history):
                dt = frame - time_history[i]
                new_kf = KF_predict(kf,dt)                
                new_kf.x[3] = self.within_range(new_kf.x[3])
                pred_of_trk.append(new_kf)
                
            trk.kf = pred_of_trk[0] # newset kf
            for i, p in enumerate(pred_of_trk): #postprocess
                pred_of_trk[i] = Box3D.array2bbox(p.x.reshape((-1))[:7])
                
            while(len(pred_of_trk)<history): # fill with None
                pred_of_trk.append(None)
            assert len(pred_of_trk) == history
            # update statistics
            trk.time_since_update += 1 		
            pred.append(pred_of_trk)
            # if trk.id == 1:
            #     for p in pred_of_trk:
            #         print(p.__str__())
            # if trk.id == self.debug_id:
            #     print('\n before prediction')
            #     print(trk.kf.x.reshape((-1)))
            #     print('\n current velocity')
            #     print(trk.get_velocity())

            # if trk.id == self.debug_id:
            #     print('After prediction')
            #     print(new_kf.x.reshape((-1)))    
        # pred = [p[0] for p in pred]     
        return pred

    def update(self, matched, unmatched_trks, dets, info, voxels, pcd, frame):
        # update matched trackers with assigned detections
        dets = copy.copy(dets)
        for t, trk in enumerate(self.track_buf):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]     # a list of index
                assert len(d) == 1, 'error'

                # update statistics
                trk.time_since_update = 0		# reset because just updated
                trk.hits += 1

                # update orientation in propagated tracks and detected boxes so that they are within 90 degree
                bbox3d = Box3D.bbox2array(dets[d[0]])
                trk.kf.x[3], bbox3d[3] = self.orientation_correction(trk.kf.x[3], bbox3d[3])                
                    

                # kalman filter update with observation
                trk.kf.update(bbox3d)

                if trk.id == self.debug_id:
                    print('after matching')
                    print(trk.kf.x.reshape((-1)))
                    print('\n current velocity')
                    print(trk.get_velocity())

                trk.kf.x[3] = self.within_range(trk.kf.x[3])
                trk.info = info[d, :][0]
                trk.update_buffer(bbox3d, voxels[d[0]], pcd[d[0]], frame)
            else:
                trk.match = False
            # debug use only
            # else:
                # print('track ID %d is not matched' % trk.id)

    def birth(self, dets, info, unmatched_dets, voxels, pcd, frame):
        # create and initialise new trackers for unmatched detections
        dets = copy.copy(dets)
        assert len(dets) == len(voxels)
        new_id_list = list()					# new ID generated for unmatched detections
        for i in unmatched_dets:        			# a scalar of index
            bbox3d = Box3D.bbox2array(dets[i])
            trk = TrackBuffer(info[i, :], self.ID_count[0], bbox3d, voxels[i], pcd[i], frame, kf_initial_speed=self.kf_initial_speed, NDT_cfg = self.NDT_cfg)
            self.track_buf.append(trk)
            new_id_list.append(trk.id)
            # print('track ID %s has been initialized due to new detection' % trk.id)
            self.ID_count[0] += 1
        # print(new_id_list)  
        return new_id_list

    def output(self, frame): #death
        # output exiting tracks that have been stably associated, i.e., >= min_hits
        # and also delete tracks that have appeared for a long time, i.e., >= max_age
        num_trks = len(self.track_buf)
        results = []
        for trk in reversed(self.track_buf):
            # change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
            if(trk.match == False): #unmatch就用kf的
                det = copy.deepcopy(trk.kf.x[:7]).reshape((7, ))
            else:#match 就用det
                det = copy.deepcopy(trk.bbox[-1])
            det[0],det[1],det[2],det[3] = move_to_frame_axis(self.oxts, self.calib, self.label_coord, frame, det[:4])
            d = Box3D.bbox2array_raw(Box3D.array2bbox(det))

            info = copy.deepcopy(trk.info)
            if(self.output_kf_cls and trk.match == False): 
                info[1]+=10                
            npdet = np.concatenate((d, [trk.id], info, [frame])).reshape(1, -1)
            
            trk.output_buf.append(npdet)
            trk.output_buf_time.append(frame)

            if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):      
                if(trk.match == True): ## match才輸出
                    for o in trk.output_buf:
                        results.append(o)
                    trk.output_buf = []
                    trk.output_buf_time=[]
    
            num_trks -= 1

            # deadth, remove dead tracklet
            #remove begin track
            if(trk.hits < self.min_hits and self.frame_count > self.min_hits and trk.match == False ):
                self.track_buf.pop(num_trks)
            elif (trk.time_since_update >= self.max_age): 
                self.track_buf.pop(num_trks)
            
        return results
    
    def output_interpolate(self, frame): #death
        # output exiting tracks that have been stably associated, i.e., >= min_hits
        # and also delete tracks that have appeared for a long time, i.e., >= max_age
        num_trks = len(self.track_buf)
        results = []
        for trk in reversed(self.track_buf):
            # change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
            if(trk.match == False): #unmatch就用kf的
                det = copy.deepcopy(trk.kf.x[:7]).reshape((7, ))
            else:#match 就用det
                det = copy.deepcopy(trk.bbox[-1])
            det[0],det[1],det[2],det[3] = move_to_frame_axis(self.oxts, self.calib, self.label_coord, frame, det[:4])
            d = Box3D.bbox2array_raw(Box3D.array2bbox(det))
            
            if(trk.match):
                trk.output_buf.append(d)
                trk.output_buf_time.append(frame)
     
            npdet = np.concatenate((d, [trk.id], trk.info, [frame])).reshape(1, -1)

            if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
                if(trk.match):
                    if(len(trk.output_buf) == 1):
                        results.append(npdet)
                        
                    else: #interpolate
                        output = interpolate_bbox(trk.output_buf[-2], trk.output_buf[-1], trk.output_buf_time[-2], trk.output_buf_time[-1], trk.id, copy.deepcopy(trk.info), self.output_kf_cls)
                        results.extend(output)
            num_trks -= 1
            # deadth, remove dead tracklet
            if (trk.time_since_update >= self.max_age): 
                self.track_buf.pop(num_trks)
        return results

    def track(self, dets_all, frame, seq_name, pcd, det_idx):
        
        """
        Params:
              dets_all: dict
                dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
                info: a array of other info for each det
            frame:    str, frame number, used to query ego pose
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        #Filter low confidence detection(dets_all,pcd,det_idx)
        conf_flag = dets_all['info'][:,6] >= self.conf_thres
        for i in reversed(range(len(conf_flag))):
            if(not conf_flag[i]):
                dets_all['dets'] = np.delete(dets_all['dets'],i,0)
                dets_all['info'] = np.delete(dets_all['info'],i,0)                
                det_idx.pop(i)
                if(self.NDT_flag):
                    pcd.pop(i)
        
        dets, info = dets_all['dets'], dets_all['info']         # dets: N x 7, float numpy array
        if self.debug_id: print('\nframe is %s' % frame)
        # logging
        print_str = '\n\n*****************************************\n\nprocessing seq_name/frame %s/%d' % (seq_name, frame)
        print_log(print_str, log=self.log, display=False)
        self.frame_count += 1

        # recall the last frames of outputs for computing ID correspondences during affinity processing
        self.id_past_output = copy.copy(self.id_now_output)
        self.id_past = [trk.id for trk in self.track_buf]
        # process detection format

        dets = self.process_dets(dets)
        
        if(self.ego_com and self.oxts is not None):
            old_dets = dets
            dets = self.move_dets_origin_axis(frame, dets)
        else:
            old_dets = dets
            
        # tracks propagation based on velocity
        trks = self.prediction(frame, history = self.history) 
        old_trks = trks
        
        # if(frame>=1):        
            # for ti in range(len(self.track_buf)):
                # print("id ",self.track_buf[ti].id)
                # print("last frame bbox ",self.track_buf[ti].bbox[-1].__str__())
                # print("predict ",trks[ti][0].__str__())            
                # print("speed ",self.track_buf[ti].kf_buffer[-1].x[-3:])  
                # print(self.track_buf[ti].bbox[-1].__str__())  
                # self.track_buf[ti].update_NDT()
            
        if(self.NDT_flag):
            NDT_Voxels = []  
            NDT_Voxels_PATH = []          
            frame_str = str(frame).zfill(6)
            for i in range(len(pcd)):
                cache_name=f"{frame_str}_{str(det_idx[i]).zfill(4)}_{self.cat.lower()}.pkl"
                NDT_Voxels_PATH.append(cache_name)
                NDTV = read_pkl(os.path.join(self.NDT_cache_path, cache_name))
                NDT_Voxels.append(NDTV)
                # if(NDTV!=None):
                #     print(cache_name)                    
                #     draw_NDT_voxel(NDTV,random=False,drawbox=True)        
        else:
            NDT_Voxels = [[] for i in range(len(dets))]   
            NDT_Voxels_PATH = [[] for i in range(len(dets))]
        # matching

        # matched, unmatched_dets, unmatched_trks, cost, affi = data_association(dets, trks, self.metric, self.thres, self.algm)
        trk_mask_1 = []
        for t,trk in enumerate(self.track_buf):
            if(trk.time_since_update > 1):
                trk_mask_1.append(t)
        ##if one stage use default, if two stage, stage1 don't use NDT
        stage1_NDT_flag = (not self.two_stage) and self.NDT_flag
        matched1, unmatched_dets1, unmatched_trks1, cost1, affi1, stage2_stat = data_association_philly(dets, trks, NDT_Voxels, self.track_buf, [], self.metric, self.thres, self.algm, history = self.history, NDT_flag=stage1_NDT_flag, NDT_thres=self.NDT_thres)
        
        if(not self.two_stage):
            matched,unmatched_dets, unmatched_trks = matched1,unmatched_dets1, unmatched_trks1
        else:
            # #collect da first stage, unmatched 
            dets2=[dets[i] for i in unmatched_dets1]
            NDT_Voxels2=[NDT_Voxels[i] for i in unmatched_dets1]
            trks2=[trks[i] for i in unmatched_trks1]
            tb2=[self.track_buf[i] for i in unmatched_trks1]

            trk_mask_2 = []
            matched2, unmatched_dets2, unmatched_trks2, cost2, affi2, stage2_stat = data_association_philly(dets2, trks2, NDT_Voxels2, tb2, trk_mask_2, self.stage2_param['metric'], self.stage2_param['thres'], self.stage2_param['algm'], history = self.history, NDT_flag=self.NDT_flag, NDT_thres=self.NDT_thres,)
            
            #Merge stage1 and stage2, map unmatched id back
            matched = matched1.tolist()   
            unmatched_dets, unmatched_trks =[],[]        
            for m in matched2:
                matched.append([unmatched_dets1[m[0]],unmatched_trks1[m[1]]])
            for m in unmatched_dets2:
                unmatched_dets.append(unmatched_dets1[m])
            for m in unmatched_trks2:
                unmatched_trks.append(unmatched_trks1[m])
                
            matched = np.array(matched)
        

        # update trks with matched detection measurement
        self.update(matched, unmatched_trks, dets, info, NDT_Voxels, pcd, frame)
        # create and initialise new trackers for unmatched detections
        new_id_list = self.birth(dets, info, unmatched_dets, NDT_Voxels, pcd, frame)
        # output existing valid tracks

        if(self.output_mode=='kf'):
            results = self.output(frame)
        elif (self.output_mode=='interpolate'):
            results = self.output_interpolate(frame)
        # assert(len(results)== len(pcd))
        if len(results) > 0: results = [np.concatenate(results)]		# h,w,l,x,y,z,theta, ID, other info, confidence
        else:            	 results = [np.empty((0, 16))]
        self.id_now_output = results[0][:, 7].tolist()					# only the active tracks that are outputed

        # post-processing affinity to convert to the affinity between resulting tracklets
        # if self.affi_process:
        #     affi = self.process_affi(affi, matched, unmatched_dets, new_id_list)
            # print_log('processed affinity matrix is', log=self.log, display=False)
            # print_log(affi, log=self.log, display=False)

        # logging
        # print_log('\ntop-1 cost selected', log=self.log, display=False)
        # print_log(cost, log=self.log, display=False)
        
        # for result_index in range(len(results)):
            # print_log(results[result_index][:, :8], log=self.log, display=False)
            # print_log('', log=self.log, display=False)
        
        if('aff' in stage2_stat):
            self.global_cnt +=len(stage2_stat['pair'])
            print_log(f"Stage2 activate.", log=self.log, display=False)
            print_log(f"Det: {stage2_stat['NDT_det_index']}", log=self.log, display=False)
            print_log(f"Trk: {stage2_stat['NDT_trk_index']}", log=self.log, display=False)
            print_log(f"NDT aff: {stage2_stat['aff']}", log=self.log, display=False)
            print_log(f"Dist: {stage2_stat['dist']}", log=self.log, display=False)            
            print_log(f"Pair: {stage2_stat['pair']}", log=self.log, display=False)
            if(len(stage2_stat['pair'])>0):
                print_log(f"Revive success.", log=self.log, display=False)
        elif('log' in stage2_stat):  
            print_log(f"Stage2 doesn't activate.\n {stage2_stat['log']}", log=self.log, display=False)
            if('unmatched_det' in stage2_stat):
                ud_idx=[det_idx[ud] for ud in unmatched_dets]
                print_log(f"unmatched_det_idx: {ud_idx}", log=self.log, display=False)
        return results, affi1
