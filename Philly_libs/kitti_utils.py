import numpy as np


def move_to_frame_axis(oxts, calib, coord, frame, det):
    return move_to_axis(oxts, calib, coord, frame, det, mode="frame")

def move_to_origin_axis(oxts, calib, coord, frame, det):
    return move_to_axis(oxts, calib, coord, frame, det, mode="origin")
import pdb
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
    if mode == "origin":    T = oxts[frame]
    elif mode == "frame":   T = inverse_transform(oxts[frame])
    xyz = np.append(det_xyz, [1.0])
    xyz_axis = np.matmul(T,xyz)[:3].reshape((1, -1))  
    
    #rotation,didn't consider the rotation of origin, oxts seems like absolute rot,relative translation
    R = T[:3,:3]
    rot = np.array([np.cos(det_rot),np.sin(det_rot),0])
    rot_axis = np.matmul(R,rot)[:2]
    rot_axis = np.arctan2(rot_axis[1],rot_axis[0])
    
    #imu_to_rect
    if(coord == "camera"):
        xyz_axis = calib.imu_to_rect(xyz_axis)
        rot_axis = calib.velo_to_rect_rot(rot_axis)
    return np.append(xyz_axis.flatten(),rot_axis).tolist()

    
def inverse_transform(T):

    R = T[:3, :3]
    t = T[:3, 3]

    R_inv = R.T
    t_inv = -R_inv.dot(t)

    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv