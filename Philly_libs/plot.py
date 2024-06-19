import numpy as np
import numpy as np
import open3d as o3d

def draw_pcd_and_bbox(pcd, box):  
    pcd = pcd[:,:3]
    
    p= o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(pcd)
    b = o3d.geometry.OrientedBoundingBox()
    # b.center = [0,0,0]
    b.center = [box['x'],box['y'],box['z']]    
    b.extent = [box['l'],box['w'],box['h']]
    R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz((0, 0,box['roty']))
    b.rotate(R, b.center)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(p)
    vis.add_geometry(b)
    vis.get_render_option().background_color = np.asarray([0, 0, 0]) # 設置一些渲染屬性
    vis.run()
    vis.destroy_window()
    # o3d.visualization.draw_geometries([p,b], width=800, height=500)    
    return

def draw_pcd(pcd):
    l = [make_pcd(pcd)]
    draw_all(l)
    return

def draw_pcd_and_bbox_v2(pcd, box):
    l = [make_pcd(pcd),make_bbox(box)]
    draw_all(l)
    return

def make_pcd(pcd):
    pcd = pcd[:,:3] 
    p= o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(pcd)
    return p

def make_bbox(box):
    b = o3d.geometry.OrientedBoundingBox()
    # b.center = [0,0,0]    
    b.center = [box['x'],box['y'],box['z']]    
    b.extent = [box['l'],box['w'],box['h']]
    if 'roty' not in box:
        box['roty'] = 0
    R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz((0, 0,box['roty']))
    b.rotate(R, b.center)
    return b

import pdb
def draw_all(l=[]):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for i in l:        
        vis.add_geometry(i)
    vis.get_render_option().background_color = np.asarray([0, 0, 0]) # 設置一些渲染屬性
    vis.run()
    vis.destroy_window()
    return