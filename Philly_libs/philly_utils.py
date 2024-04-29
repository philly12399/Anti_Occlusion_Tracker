import os
import open3d as o3d
def pcd_info_seq_preprocess(pcd_info, pcd_db_seq_root, min_frame, max_frame, class_map, cat):
	cat_low = cat.lower()    
	pcd_info_seq=[[] for i in range(min_frame, max_frame + 1)]
	frame_cnt = [0 for i in range(min_frame, max_frame + 1)]
	for pp in pcd_info:
		#print(pp['obj']['frame_id'],pp['obj_det_idx'])
		fid = pp['obj']['frame_id']-min_frame
		frame_seq = pcd_info_seq[fid]
		if(pp['obj']['obj_type'] in class_map):
			pp['obj']['obj_type'] = class_map[pp['obj']['obj_type']]
		if(pp['obj']['obj_type']!=cat_low):
			frame_cnt[fid]+=1
			continue
		pp['mae_dense_path'] = os.path.join(pcd_db_seq_root, pp['mae_dense_path'])
		##Put info to correspond frame, some pcd is not consequence, so pad with None
		while(pp['obj_det_idx']>frame_cnt[fid]):
			frame_seq.append(None)			
		frame_seq.append(pp)
		frame_cnt[fid]+=1
	return pcd_info_seq



    
# def drawbox(vis,box,drawpts=False,color=[0,0,0]):
#     b = o3d.geometry.OrientedBoundingBox()
#     b.center = [box['x'],box['y'],box['z']]
#     b.extent = [box['l'],box['w'],box['h']]
#     b.color = color
#     vis.add_geometry(b)
#     if(drawpts and len(box["pts"]) > 0 ):
#         p = o3d.geometry.PointCloud()
#         p.points = o3d.utility.Vector3dVector(box["pts"])
#         vis.add_geometry(p)
    
def rank_list(input_list): #自己在原本list是第幾小
    ranked_list = sorted(range(len(input_list)), key=lambda x: input_list[x])
    return [ranked_list.index(i) for i in range(len(input_list))]

def in_bbox(point, bbox):
    x = point[0]
    y = point[1]
    z = point[2]
    x_min = bbox.x - bbox.l / 2
    x_max = bbox.x + bbox.l / 2
    y_min = bbox.y - bbox.w / 2
    y_max = bbox.y + bbox.w / 2
    z_min = bbox.z - bbox.h / 2
    z_max = bbox.z + bbox.h / 2
    return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z >= z_min) & (z <= z_max)

def draw_pts(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.visualization.draw_geometries([pcd,], width=800, height=500)