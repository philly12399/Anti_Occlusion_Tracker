# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import numpy as np, os, copy, math
from AB3DMOT_libs.box import Box3D
from AB3DMOT_libs.matching import data_association
from AB3DMOT_libs.kalman_filter import KF
from AB3DMOT_libs.vis import vis_obj
from Philly_libs.philly_matching import data_association as data_association_philly
from Philly_libs.TrackBuffer import TrackBuffer , KF_predict
from Philly_libs.NDT import NDT_voxelize,draw_NDT_voxel
from Philly_libs.philly_io import read_pkl
from xinshuo_miscellaneous import print_log
from xinshuo_io import mkdir_if_missing
import time
from multiprocessing import Pool
import pdb
import pickle
np.set_printoptions(suppress=True, precision=3)

# A Baseline of 3D Multi-Object Tracking
class AB3DMOT(object):			  	
    def __init__(self, cfg, cat, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=None, log=None, ID_init=0):                    

        # vis and log purposes
        self.img_dir = img_dir
        self.vis_dir = vis_dir
        self.vis = cfg.vis
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
        self.affi_process = cfg.affi_pro	# post-processing affinity
        self.get_param(cfg, cat)
        self.print_param()
  
        self.label_format = None
        if('label_format' in cfg):
            self.label_format = cfg.label_format
            
        self.history = 5
        if('history' in cfg):
            self.history = cfg.history
        Box3D.set_label_format(self.label_format)
        ##NDT
        self.NDT_cfg = None
        if('NDT_cfg' in cfg):
            self.NDT_cfg = cfg.NDT_cfg
        self.NDT_out_path = None
        if('NDT_out_path' in cfg):
            self.NDT_out_path = cfg.NDT_out_path
          # debug
        # self.debug_id = 2
        self.debug_id = None

    def get_param(self, cfg, cat):
        # get parameters for each dataset

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
            else: assert False, 'error'
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
            if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.2, 3, 2		
            elif cat == 'Cyclist': 		algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 2, 3, 4
            else: assert False, 'error'
        else: assert False, 'no such dataset'

        # add negative due to it is the cost
        if metric in ['dist_3d', 'dist_2d', 'm_dis']: thres *= -1	
        self.algm, self.metric, self.thres, self.max_age, self.min_hits = \
            algm, metric, thres, max_age, min_hits

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

    def ego_motion_compensation(self, frame, trks):
        # inverse ego motion compensation, move trks from the last frame of coordinate to the current frame for matching
        
        from AB3DMOT_libs.kitti_oxts import get_ego_traj, egomotion_compensation_ID
        assert len(self.track_buf) == len(trks), 'error'
        ego_xyz_imu, ego_rot_imu, left, right = get_ego_traj(self.oxts, frame, 1, 1, only_fut=True, inverse=True) 
        for index in range(len(self.track_buf)):
            trk_tmp = trks[index]
            xyz = np.array([trk_tmp.x, trk_tmp.y, trk_tmp.z]).reshape((1, -1))
            compensated = egomotion_compensation_ID(xyz, self.calib, ego_rot_imu, ego_xyz_imu, left, right)
            trk_tmp.x, trk_tmp.y, trk_tmp.z = compensated[0]

            # update compensated state in the Kalman filter
            try:
                self.track_buf[index].kf.x[:3] = copy.copy(compensated).reshape((-1))
            except:
                self.track_buf[index].kf.x[:3] = copy.copy(compensated).reshape((-1, 1))

        return trks

    def visualization(self, img, dets, trks, calib, hw, save_path, height_threshold=0):
        # visualize to verify if the ego motion compensation is done correctly
        # ideally, the ego-motion compensated tracks should overlap closely with detections
        import cv2 
        from PIL import Image
        from AB3DMOT_libs.vis import draw_box3d_image
        from xinshuo_visualization import random_colors

        dets, trks = copy.copy(dets), copy.copy(trks)
        img = np.array(Image.open(img))
        max_color = 20
        colors = random_colors(max_color)       # Generate random colors

        # visualize all detections as yellow boxes
        for det_tmp in dets: 
            img = vis_obj(det_tmp, img, calib, hw, (255, 255, 0))				# yellow for detection
        
        # visualize color-specific tracks
        count = 0
        ID_list = [tmp.id for tmp in self.track_buf]
        for trk_tmp in trks: 
            ID_tmp = ID_list[count]
            color_float = colors[int(ID_tmp) % max_color]
            color_int = tuple([int(tmp * 255) for tmp in color_float])
            str_vis = '%d, %f' % (ID_tmp, trk_tmp.o)
            img = vis_obj(trk_tmp, img, calib, hw, color_int, str_vis)		# blue for tracklets
            count += 1
        
        img = Image.fromarray(img)
        img = img.resize((hw['image'][1], hw['image'][0]))
        img.save(save_path)

    def prediction(self, history = 5):
        # get predicted locations from existing tracks

        pred = []
        for t in range(len(self.track_buf)):
            # propagate locations
            trk = self.track_buf[t]
            pred_of_trk=[]      
            for i, kf in enumerate(reversed(trk.kf_buffer[-history:])):
                new_kf = KF_predict(kf,i)                
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

                if trk.id == self.debug_id:
                    print('After ego-compoensation')
                    print(trk.kf.x.reshape((-1)))
                    print('matched measurement')
                    print(bbox3d.reshape((-1)))
                    # print('uncertainty')
                    # print(trk.kf.P)
                    # print('measurement noise')
                    # print(trk.kf.R)

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
            trk = TrackBuffer(info[i, :], self.ID_count[0], bbox3d, voxels[i], pcd[i], frame)
            self.track_buf.append(trk)
            new_id_list.append(trk.id)
            # print('track ID %s has been initialized due to new detection' % trk.id)
            self.ID_count[0] += 1

        return new_id_list

    def output(self): #death
        # output exiting tracks that have been stably associated, i.e., >= min_hits
        # and also delete tracks that have appeared for a long time, i.e., >= max_age
        num_trks = len(self.track_buf)
        results = []
        for trk in reversed(self.track_buf):
            # change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
            if(trk.match == False): #unmatch就用kf的
                d = Box3D.array2bbox(trk.kf.x[:7].reshape((7, ))) 
            else:#match 就用det
                d = Box3D.array2bbox(trk.bbox[-1])     # bbox location self
            d = Box3D.bbox2array_raw(d)
            
            if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):      
                if(trk.match ==True): ## match才輸出
                    results.append(np.concatenate((d, [trk.id], trk.info)).reshape(1, -1)) 		
            num_trks -= 1

            # deadth, remove dead tracklet
            if (trk.time_since_update >= self.max_age): 
                self.track_buf.pop(num_trks)

        return results

    def process_affi(self, affi, matched, unmatched_dets, new_id_list):

        # post-processing affinity matrix, convert from affinity between raw detection and past total tracklets
        # to affinity between past "active" tracklets and current active output tracklets, so that we can know 
        # how certain the results of matching is. The approach is to find the correspondes of ID for each row and
        # each column, map to the actual ID in the output trks, then purmute/expand the original affinity matrix
        
        ###### determine the ID for each past track
        trk_id = self.id_past 			# ID in the trks for matching

        ###### determine the ID for each current detection
        det_id = [-1 for _ in range(affi.shape[0])]		# initialization

        # assign ID to each detection if it is matched to a track
        for match_tmp in matched:		
            det_id[match_tmp[0]] = trk_id[match_tmp[1]]

        # assign the new birth ID to each unmatched detection
        count = 0
        assert len(unmatched_dets) == len(new_id_list), 'error'
        for unmatch_tmp in unmatched_dets:
            det_id[unmatch_tmp] = new_id_list[count] 	# new_id_list is in the same order as unmatched_dets
            count += 1
        assert not (-1 in det_id), 'error, still have invalid ID in the detection list'

        ############################ update the affinity matrix based on the ID matching
        
        # transpose so that now row is past trks, col is current dets	
        affi = affi.transpose() 			

        ###### compute the permutation for rows (past tracklets), possible to delete but not add new rows
        permute_row = list()
        for output_id_tmp in self.id_past_output:
            index = trk_id.index(output_id_tmp)
            permute_row.append(index)
        affi = affi[permute_row, :]	
        assert affi.shape[0] == len(self.id_past_output), 'error'

        ###### compute the permutation for columns (current tracklets), possible to delete and add new rows
        # addition can be because some tracklets propagated from previous frames with no detection matched
        # so they are not contained in the original detection columns of affinity matrix, deletion can happen
        # because some detections are not matched

        max_index = affi.shape[1]
        permute_col = list()
        to_fill_col, to_fill_id = list(), list() 		# append new columns at the end, also remember the ID for the added ones
        for output_id_tmp in self.id_now_output:
            try:
                index = det_id.index(output_id_tmp)
            except:		# some output ID does not exist in the detections but rather predicted by KF
                index = max_index
                max_index += 1
                to_fill_col.append(index); to_fill_id.append(output_id_tmp)
            permute_col.append(index)

        # expand the affinity matrix with newly added columns
        append = np.zeros((affi.shape[0], max_index - affi.shape[1]))
        append.fill(self.min_sim)
        affi = np.concatenate([affi, append], axis=1)

        # find out the correct permutation for the newly added columns of ID
        for count in range(len(to_fill_col)):
            fill_col = to_fill_col[count]
            fill_id = to_fill_id[count]
            row_index = self.id_past_output.index(fill_id)

            # construct one hot vector because it is proapgated from previous tracks, so 100% matching
            affi[row_index, fill_col] = self.max_sim		
        affi = affi[:, permute_col]

        return affi

    def track(self, dets_all, frame, seq_name, pcd_info, pcd):
        
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
        # tracks propagation based on velocity
        trks = self.prediction(history = self.history)
        ## Comment for wayside (don't need)
        # # ego motion compensation, adapt to the current frame of camera coordinate
        # if (frame > 0) and (self.ego_com) and (self.oxts is not None):
        # 	trks = self.ego_motion_compensation(frame, trks)

        # # visualization
        # if self.vis and (self.vis_dir is not None):
        # 	img = os.path.join(self.img_dir, f'{frame:06d}.png')
        # 	save_path = os.path.join(self.vis_dir, f'{frame:06d}.jpg'); mkdir_if_missing(save_path)
        # 	self.visualization(img, dets, trks, self.calib, self.hw, save_path)
        NDT_LOAD_PKL=True
        TT0=time.time()
        ## For Multi threading 
        NDT_Voxels = []
        MT_pcd = copy.copy(pcd)
        MT_dets = copy.copy(dets)
        update_tid=[]
        ## some trk don't need voxelize again
        for i,t in enumerate(self.track_buf):
            if(t.NDT_updated == False and t.pcd_of_track is not None): 
                mean_box = Box3D.array2bbox(np.append([0,0,0,0],np.mean(t.bbox,0)[-3:]))   
                MT_pcd.append(t.pcd_of_track)
                update_tid.append(i)
                MT_dets.append(mean_box)
                
        ## Multi threading 
        if(NDT_LOAD_PKL):
            TT2=time.time()
            ## Update result for  det/trk
            for i in range(len(MT_pcd)):
                if(i < len(pcd)): ##det
                    NDTV = read_pkl(os.path.join(self.NDT_out_path, f"{frame}_{i}_det"))
                    NDT_Voxels.append(NDTV)
                else: ##track
                    NDTV = read_pkl(os.path.join(self.NDT_out_path, f"{frame}_{update_tid[i-len(pcd)]}_trk"))                  
                    t = self.track_buf[update_tid[i-len(pcd)]]
                    t.update_NDT(NDTV)
                # draw_NDT_voxel(NDTV)
        else:            
            if(len(MT_pcd)>0):
                ## Multi threading init
                pool = [Pool() for _ in range(len(MT_pcd))]
                ## Multi threading and run NDT_voxelize(non blocking)
                thread=[pool[i].apply_async(NDT_voxelize, (MT_pcd[i],MT_dets[i],self.NDT_cfg))  for i in range(len(MT_pcd))]
                result=[None for _ in range(len(MT_pcd))]
                ## Collect result(blocking)
                for i in range(len(MT_pcd)):
                    result[i],_,_ = thread[i].get(timeout=50)  #valid,invalid,all ; collect valid only
                ## Clean  pool
                for p in pool:  p.close() 
                ## Update result for  det/trk
                for i in range(len(MT_pcd)):
                    if(i < len(pcd)): ##det
                        NDT_Voxels.append(result[i])
                    else: ##track
                        t = self.track_buf[update_tid[i-len(pcd)]]
                        t.update_NDT(result[i])
                        # if(t.id == 3):   
                        #     print(frame)                     
                        #     draw_NDT_voxel(t.NDT_of_track)
            for i, NDTV in enumerate(NDT_Voxels):
                with open(os.path.join(self.NDT_out_path, f"{frame}_{i}_det"), 'wb') as file:
                    pickle.dump(NDTV, file)
            for i, tid in enumerate(update_tid):
                with open(os.path.join(self.NDT_out_path, f"{frame}_{tid}_trk"), 'wb') as file:
                    pickle.dump(self.track_buf[tid].NDT_of_track, file)
            
        TT1=time.time()
        
        
        # matching

        # matched, unmatched_dets, unmatched_trks, cost, affi = data_association(dets, trks, self.metric, self.thres, self.algm)
        matched, unmatched_dets, unmatched_trks, cost, affi = data_association_philly(dets, trks, NDT_Voxels, self.track_buf, self.metric, self.thres, self.algm, history = self.history)
        TT3=time.time()
          # print_log('detections are', log=self.log, display=False)
        # print_log(dets, log=self.log, display=False)
        # print_log('tracklets are', log=self.log, display=False)
        # print_log(trks, log=self.log, display=False)
        # print_log('matched indexes are', log=self.log, display=False)
        # print_log(matched, log=self.log, display=False)
        # print_log('raw affinity matrix is', log=self.log, display=False)
        # print_log(affi, log=self.log, display=False)

        # update trks with matched detection measurement
        self.update(matched, unmatched_trks, dets, info, NDT_Voxels, pcd, frame)
        # create and initialise new trackers for unmatched detections
        new_id_list = self.birth(dets, info, unmatched_dets, NDT_Voxels, pcd, frame)

        # output existing valid tracks
        results = self.output()
        # assert(len(results)== len(pcd))
        if len(results) > 0: results = [np.concatenate(results)]		# h,w,l,x,y,z,theta, ID, other info, confidence
        else:            	 results = [np.empty((0, 15))]
        self.id_now_output = results[0][:, 7].tolist()					# only the active tracks that are outputed

        # post-processing affinity to convert to the affinity between resulting tracklets
        if self.affi_process:
            affi = self.process_affi(affi, matched, unmatched_dets, new_id_list)
            # print_log('processed affinity matrix is', log=self.log, display=False)
            # print_log(affi, log=self.log, display=False)

        # logging
        print_log('\ntop-1 cost selected', log=self.log, display=False)
        print_log(cost, log=self.log, display=False)
        for result_index in range(len(results)):
            print_log(results[result_index][:, :8], log=self.log, display=False)
            print_log('', log=self.log, display=False)
        # print(f"MT_NDT_Voxelize_timetime:{TT1-TT0}, with len {len(MT_pcd)}; DA_time:{TT3-TT1};")
        print(f"LOAD_NDT_TIME:{TT1-TT2}")
        # print(f"DET_NDT_Voxelize_time:{TT1-TT0}; Track_NDT_Voxelize_time:{TT2-TT1};DA_time:{TT3-TT2};")
        return results, affi