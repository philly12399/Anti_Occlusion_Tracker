import numpy as np
from filterpy.kalman import KalmanFilter
import copy
from Philly_libs.philly_utils import *
from AB3DMOT_libs.box import Box3D
from Philly_libs.NDT import NDT_voxelize

class TrackBuffer():
    def __init__(self, info, ID, bbox3D, voxel, pcd, time_stamp, kf_initial_speed=30, NDT_cfg=None):
        self.initial_speed = kf_initial_speed*(1000/36000)
        # self.label_format=label_format # for rot in initial speed
        

        self.id = ID
        self.info = info
        self.time_since_update = 0
        self.hits = 1  
  
        self.bbox = [] #x,y,z,ry,l,w,h (l,w,h in camera)
        self.NDT_voxels = []
        self.time_stamp = []
        self.kf_buffer = []
        self.match = True
        self.status = []
        self.output_buf = []
        self.output_buf_time = []
        
        self.pcd_of_track = None
        self.pcd_of_track_all=None
        self.NDT_of_track = None
        self.NDT_updated = False
        self.NDT_cfg = NDT_cfg
        self.KF_init(bbox3D)
        ##UPDATE
        self.update_buffer(bbox3D, voxel, pcd, time_stamp)

    def get_velocity(self):
        # return the object velocity in the state
        return self.kf.x[7:]

    def update_buffer(self, bbox, voxel, pcd, time_stamp):
        self.bbox.append(bbox)
        self.NDT_voxels.append(voxel)
        self.time_stamp.append(time_stamp)
        self.kf_buffer.append(self.kf)
        self.match = True

        if(pcd is not None):
            self.pcd_of_track_all, self.pcd_of_track = POT_append_downsample(self.pcd_of_track_all,pcd)
            self.NDT_updated = False # pcd update, so NDT need recalculate
            # if(self.id == 3):
            # 	print(time_stamp,self.id)
            # 	draw_pts(self.pcd_of_track)
   
    def update_NDT(self):
        if(self.NDT_updated == True):
            return
        mean_dim = np.mean(self.bbox,0)[-3:]     
        if(Box3D.label_format.lower() == "wayside"):
            mean_dim[0], mean_dim[1], mean_dim[2] = mean_dim[2], mean_dim[0], mean_dim[1] 
        mean_box = Box3D.array2bbox(np.append([0,0,0,0],mean_dim))#l,w,h
        valid,invalid,allv = NDT_voxelize(self.pcd_of_track, mean_box, cfg = self.NDT_cfg, draw=False) #valid,invalid,all
        if(valid==None):
            return
        self.NDT_of_track = valid
        self.NDT_updated = True
        
    def KF_init(self, bbox3D):
        #Kalman filter
        self.kf = KalmanFilter(dim_x=10, dim_z=7)       
        # There is no need to use EKF here as the measurement and state are in the same space with linear relationship

        # state x dimension 10: x, y, z, theta, l, w, h, dx, dy, dz
        # constant velocity model: x' = x + dx, y' = y + dy, z' = z + dz 
        # while all others (theta, l, w, h, dx, dy, dz) remain the same
        self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix, dim_x * dim_x
                              [0,1,0,0,0,0,0,0,1,0],
                              [0,0,1,0,0,0,0,0,0,1],
                              [0,0,0,1,0,0,0,0,0,0],  
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,0,0,1]])     

        # measurement function, dim_z * dim_x, the first 7 dimensions of the measurement correspond to the state
        self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      
                              [0,1,0,0,0,0,0,0,0,0],
                              [0,0,1,0,0,0,0,0,0,0],
                              [0,0,0,1,0,0,0,0,0,0],
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0]])

        # measurement uncertainty, uncomment if not super trust the measurement data due to detection noise
        # self.kf.R[0:,0:] *= 10.   

        # initial state uncertainty at time 0
        # Given a single data, the initial velocity is very uncertain, so giv a high uncertainty to start
        self.kf.P[7:, 7:] *= 1000. 	
        self.kf.P *= 10.

        # process uncertainty, make the constant velocity part more certain
        self.kf.Q[7:, 7:] *= 0.01

        # initialize data
        self.kf.x[:7] = bbox3D.reshape((7, 1))
        
        if(Box3D.label_format.lower()=="kitti"):
            rot=-bbox3D[3] # kitti的rot是多一個負號
        else:
            rot=bbox3D[3]
            
        self.kf.x[-3] = np.cos(rot)*self.initial_speed
        self.kf.x[-1] = np.sin(rot)*self.initial_speed

        
def KF_predict(kf,time=1):
    """ predict the state of the kalman filter
    """
    new_kf = copy.copy(kf)
    for t in range(time):
        new_kf.predict()
    # new_kf.predict()
    return new_kf
        
def POT_append_downsample(pcd_all, pcd):
    if(pcd_all is None):
        pcd_all = pcd
    else:
        #OLD METHOD			
        # old_pcds = random_drop(old_pcds, alpha)
        # old_pcds = np.row_stack((old_pcds,pcd))	
        # old_pcds = old_pcds[random_sample(len(old_pcds),4096)]
        #NEW METHOD
        pcd_all = np.row_stack((pcd_all,pcd))	
    pcd_sample = pcd_all[random_sample(len(pcd_all),4096)]
    return pcd_all,pcd_sample