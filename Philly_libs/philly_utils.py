import os
import open3d as o3d
from  Philly_libs.philly_io import *
import pdb
import copy
#frame_det_idx 是detection中符合cat的那些obj在diff0(gt)的det idx,因為我是用diff0去create gtdb
def pcd_info_seq_preprocess(pcd_info, pcd_db_seq_root, frame_num, frame_det_idx): 
    pcd_info_seq=[[] for i in range(frame_num)]
    # print(len(frame_det_idx),len(pcd_info_seq)) #frame_det_idx的總數<=pcd_info_seq的總數,因為可能最後幾frame沒有detection
    for pp in pcd_info:
        fid = pp['obj']['frame_id']
        if(fid>=len(frame_det_idx)):
            break
        frame_seq = pcd_info_seq[fid]
        if(pp is None or pp['obj_det_idx'] not in frame_det_idx[fid]):
            continue
        pp['mae_dense_path'] = os.path.join(pcd_db_seq_root, pp['mae_dense_path'])	
        frame_seq.append(pp)
    
    for i in range(frame_num):
        assert len(pcd_info_seq[i]) == len(frame_det_idx[i])
        for j in range(len(frame_det_idx[i])):
            id=pcd_info_seq[i][j]['obj_det_idx']
            assert id == frame_det_idx[i][j]
        
    return pcd_info_seq

def load_dense_byinfo(pcd_info_frame):
    pcd_frame = []
    for p in pcd_info_frame:
        if(p is None or not p['valid']):
            pcd_frame.append(None)
        else:
            dense_pts = read_points_from_bin(p['mae_dense_path'], unique=True)
            pcd_frame.append(dense_pts)
    return pcd_frame

    
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

# def draw_pts(pts):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pts)
#     o3d.visualization.draw_geometries([pcd,], width=800, height=500)
    
def make_bbox(box):
    b = o3d.geometry.OrientedBoundingBox()
    # b.center = [0,0,0]    
    b.center = [box['x'],box['y'],box['z']]    
    b.extent = [box['l'],box['w'],box['h']]
    R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz((0, 0,box['roty']))
    b.rotate(R, b.center)
    return b

def random_sample(n,c):
    randomi = np.arange(n)
    np.random.shuffle(randomi)
    return randomi[:c]

def random_drop(arr,r):
    assert r<=1 and r>=0
    r=1-r
    randomi = random_sample(len(arr),int(len(arr)*r))
    return arr[randomi]
    
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

def interpolate_bbox(box1,box2,start,end,id,info, output_kf_cls):
    num=end-start+1
    output = []
    if(num<2): assert False
    
    info_kf = copy.deepcopy(info)
    info_kf[1]+=10
    
    size=(box2[:3]+box1[:3])/2
    frame=list(range(start,end+1))
    boxes = interpolate(box1[3:],box2[3:],num)
    for i,b in enumerate(boxes):
        boxes[i] = np.append(size,b,axis=0)
    boxes[0],boxes[-1] = box1,box2
    for i,b in enumerate(boxes):
        if(i==0): continue
        if(output_kf_cls and i!= len(boxes)-1):
            output.append(np.concatenate((b, [id], info_kf, [frame[i]])).reshape(1, -1))
        else:
            output.append(np.concatenate((b, [id], info, [frame[i]])).reshape(1, -1))
    # pdb.set_trace()

    return output

def interpolate(a,b,t):
    return [a + i * (b - a) / (t - 1) for i in range(t)] 

class DictToObj:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)