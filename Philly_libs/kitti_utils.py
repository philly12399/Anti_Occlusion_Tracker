import numpy as np


def move_to_frame_axis(oxts, calib, coord, frame, det):
    return move_to_axis(oxts, calib, coord, frame, det, mode="frame")

def move_to_origin_axis(oxts, calib, coord, frame, det):
    return move_to_axis(oxts, calib, coord, frame, det, mode="origin")
import pdb
import copy
#origin mode or frame mode    
def move_to_axis(oxts, calib, coord, frame, det, mode="origin"):
    assert mode == "origin" or mode == "frame"
    assert coord == "lidar" or coord == "camera"
    
    det_xyz = np.array(det[:3]).reshape((1, -1))
    det_rot = det[3] 
    # rect_to_imu
    if(coord == "camera"):
        det_xyz = calib.rect_to_imu(det_xyz)
        det_rot = calib.rect_to_velo_rot(det_rot)
    
    # translation
    if mode == "origin":    
        T = copy.deepcopy(oxts[frame])
        T0 = inverse_transform(copy.deepcopy(oxts[0]))
        T = np.matmul(T0,T)
    elif mode == "frame":   
        T = inverse_transform(copy.deepcopy(oxts[frame]))
        T0 = copy.deepcopy(oxts[0])
        T = np.matmul(T,T0)
   
    xyz = np.append(det_xyz, [1.0])
    xyz_axis = np.matmul(T,xyz)[:3].reshape((1, -1))  
    
    #rotation,didn't consider the rotation of origin, oxts seems like absolute rot,relative translation
    R = T[:3,:3]
    rot = np.array([np.cos(det_rot),np.sin(det_rot),0])
    rot_axis = np.matmul(R,rot)[:2]

    rot_axis = np.arctan2(rot_axis[1],rot_axis[0])
    # pdb.set_trace()
    #imu_to_rect
    if(coord == "camera"):
        xyz_axis = calib.imu_to_rect(xyz_axis)
        rot_axis = calib.velo_to_rect_rot(rot_axis)
    # pdb.set_trace()
    return np.append(xyz_axis.flatten(),rot_axis).tolist()

def EXP(oxts,frame):
    
    T = oxts[frame]
    T_inv = inverse_transform(oxts[frame])
    
    R0 = oxts[0][:3,:3]
    R0_T= R0.T
    
    R = np.matmul(R0_T,T[:3,:3])
    R_inv = np.matmul(T_inv[:3,:3],R0)
    print("\n")
    print(frame)
    print(oxts[frame])
    print(np.matmul(R,R_inv))
    return

def inverse_transform(T):

    R = T[:3, :3]
    t = T[:3, 3]

    R_inv = R.T
    t_inv = -R_inv.dot(t)

    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

import pdb

def KITTI_FOV_filter(calib, pts_rect, img_shape=(375, 1242)):
    # pts_rect = calib.lidar_to_rect(points[:, 0:3])
    # print(pts_rect.shape)
    # pts_img, pts_rect_depth = calib.project_rect_to_image(pts_rect)
    
    pts_img = calib.project_rect_to_image(pts_rect)
    
    # b_reshaped = pts_rect_depth[:, np.newaxis]
    # p1=np.hstack((pts_img,b_reshaped))
    # print(p1)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    return val_flag_merge

    # pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    # return pts_valid_flag