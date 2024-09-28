import os
import pickle
import numpy as np
import open3d as o3d 
import math
from Philly_libs.NDTPDF import PDF
from Philly_libs.philly_utils import in_bbox
from Philly_libs.plot import *
import time
import pdb
def test(track_root='./output_bytrackid/car_mark_all_rotxy'):
    np.random.seed(0)
    track_path = {}
    dirlist=sorted(os.listdir(track_root))
    for i in range(len(dirlist)):
        path1 = os.path.join(track_root, dirlist[i])
        if(not os.path.isdir(path1)):
            pkl=dirlist.pop(i) 
    with open(os.path.join(track_root, pkl), 'rb') as file:
        info = pickle.load(file)
    vis = o3d.visualization.Visualizer()
    # vis.get_render_option().background_color = np.asarray([0, 0, 0])
    
    for t in dirlist:
        track_path[t] = []
        tpath = os.path.join(track_root, t)
        frame_num = (int(sorted(os.listdir(tpath))[-1][:6])+1)  
        for f in range(frame_num):            
            pcd_path=os.path.join(tpath, str(f).zfill(6)+'_dense_points.txt')
            track_path[t].append(pcd_path)
        # one track only
        break
    
    VOXEL_SIZE = 0.5
    DENSITY = 100
    NUM_SAMPLES = int((VOXEL_SIZE**3)*DENSITY)
    PDF_FLAG=False    
    MIN_PTS_VOXEL = 5
    track_buffer = {}
    voxel_of_track = {}
    rep = [7,19,30,29,20,39]
    
    for ti, tid in enumerate(track_path):

        t_info = info[tid]
        frames = track_path[tid]
        track_buffer[tid] = []
        l=[]  
        allpts=[]
        for fi, f in enumerate(frames):
            pcd = read_vis_points(f)
            bbox = t_info[fi]['obj']['box3d']
            valid_voxel, invalid_voxel, voxel = voxelize(pcd, bbox, VOXEL_SIZE, overlap=True, min_pts = MIN_PTS_VOXEL)
            track_buffer[tid].append({'voxel':valid_voxel,'bbox':bbox})
            vis.create_window()
            # drawbox(vis,bbox,color = [1,0,0])
            p_mean = []
            pts = []
            pdf = []

            for vi,v in enumerate(valid_voxel):
                # drawbox(vis,v)
                p_mean.append(v['mean'])  
                # random sample     
                # if(PDF_FLAG):
                #     samples = np.random.multivariate_normal(v['mean'], v['cov'], NUM_SAMPLES)
                #     pdf.append(samples.tolist())       
                #     mvn = scipy.stats.multivariate_normal(mean=v['mean'], cov=v['cov'], allow_singular=True)
                #     mvn_sample = mvn.pdf(samples)
                #     neg_log_psi = -np.average(np.log(mvn_sample))
                #     ndt_score = -np.average(mvn_sample)                          
            allpts+=p_mean 
            # if(PDF_FLAG):                      
            #     pdf = np.array(pdf).reshape(-1,3) 
            #     print(f"Sample {NUM_SAMPLES} points per voxel. {len(valid_voxel)}/{len(invalid_voxel) + len(valid_voxel)} voxels are valid.")
            #     print(pdf.shape)
            #     print(f"NDT score: {global_ndt_score}, Neg log psi: {global_neg_log_psi}")
        # print(len(allpts))
        # p = o3d.geometry.PointCloud()            
        # p.points = o3d.utility.Vector3dVector(allpts)
        # vis.add_geometry(p)
        # vis.run()
        # vis.destroy_window() 
        v0,_,_ = voxelize(allpts, track_buffer[tid][rep[ti]]['bbox'], VOXEL_SIZE, overlap=True, min_pts = MIN_PTS_VOXEL)
        voxel_of_track[tid] = v0
    
    
    
    
    
    # ### in same track, with mean of voxel
    # for ti, tid in enumerate(track_path):   
    #     buf = track_buffer[tid] 
    #     s=[]
    #     for fi,f in enumerate(buf):
    #         score = NDT_score(f,{'voxel':v0,'bbox':buf[rep[ti]]['bbox']})
    #         s.append(score)
    #     print(s)
    #     pdb.set_trace()
    #     # trackbuf.append(track_buffer[tid][rep[ti]])
    # exit()
    # ###different track 
    # s=[]
    # for fi in range(len(trackbuf)): 
    #     for fj in range(len(trackbuf)): 
    #         score = NDT_score(trackbuf[fi],trackbuf[fj])
    #         s.append(score)  
    # s = np.array(s).reshape(len(trackbuf),len(trackbuf))
    # for si in range(len(s)):
    #     print(s[si])
    #     print(rank_list(s[si]))
    # pdb.set_trace()
    # exit()
    
    ###in same track
    for ti, tid in enumerate(track_path):
        buf = track_buffer[tid]
        ref = buf[7]
        s=[]
        for fi in range(len(buf)): 
            for fj in range(len(buf)): 
                # print(fi,fj)        
                score = NDT_score(buf[fi],ref)
                # print(score)
                s.append(score)  
            print(f"frame {fi}")
            print(s)
        # print(NDT_score(buf[1],buf[-1]))

        # s = np.array(s).reshape(len(buf),len(buf))
        # for si in range(len(s)):
        #     print(s[si])
        #     print(rank_list(s[si]))
        
        pdb.set_trace()
        break
        

def NDT_voxelize(pcd, det, cfg=None,draw=False): # 只用到det lwh
    TT=[time.time()]
    TT.append(time.time())
    if(pcd is None):
        return None,None,None
    
    voxel_size, overlap, min_pts_voxel, noise = 0.5, True, 5, 0.05
    if(cfg is not None):
        voxel_size, overlap, min_pts_voxel, noise = cfg['voxel_size'], cfg['overlap'], cfg['min_pts_voxel'], cfg['noise']
    # move to orign 
     
    box = NDT_Voxel(0,0,0,det.l,det.w,det.h)
    pcd = [p for p in pcd if in_bbox(p,box)]
    # pdb.set_trace()
    
    if(draw):
        box_dict = {
            'x':0,'y': 0, 'z':0,'l':det.l,'w':det.w, 'h':det.h, 'roty':0
        }
        draw_pcd_and_bbox_v2(np.array(pcd),box_dict)
        
    # pdb.set_trace()
    origin = 0 #第一個voxel的中心在原點
    # origin = voxel_size/2 #第一個voxel中心在原點往右上平移voxelsize/2
    if(overlap):
        stride = voxel_size/2
        scalar=8
    else:
        stride = voxel_size
        scalar=1
        
    l, w, h = box.l, box.w, box.h
    ln, wn, hn = math.ceil(l / (stride))+1, math.ceil(w / (stride))+1, math.ceil(h / (stride))+1
    voxels = []
    
    for i in range(0, ln):
        for j in range(0, wn):
            for k in range(0, hn):
                # 起點voxel中心在原點,如果想讓左下在原點就要+voxel_size/2,並把ln,wn,hn -1
                v = {
                    'x': i * (stride) - l / 2 + origin,
                    'y': j * (stride) - w / 2 + origin,
                    'z': k * (stride) - h / 2 + origin,
                    'l': voxel_size,
                    'w': voxel_size,
                    'h': voxel_size,
                }
                voxels.append(NDT_Voxel(v['x'],v['y'],v['z'],v['l'],v['w'],v['h'],voxel_size,min_pts_voxel,noise))
    TT.append(time.time())
    
    #regular
    incnt=0
    for p in pcd:
        
        i,j,k = int((p[0]+l/2+ (voxel_size/2-origin))/(stride)), int((p[1]+w/2+ (voxel_size/2-origin))/(stride)), int((p[2]+h/2+ (voxel_size/2-origin))/(stride))   
        idx0 = i*wn*hn + j*hn + k
        if(not overlap):
            if(in_bbox(p,voxels[idx0])):
                voxels[idx0].pts.append(p)
                incnt+=1
                continue   
        else:   
            direct = voxels[idx0].get_direction(p)
            direct = [[0,direct[0]],[0,direct[1]],[0,direct[2]]]
            for di in direct[0]:
                for dj in direct[1]:
                    for dk in  direct[2]:  
                        idx = int((i+di)*wn*hn + (j+dj)*hn + (k+dk))
                        if(in_bbox(p,voxels[idx])):  
                            voxels[idx].pts.append(p)
                            incnt+=1 

    assert scalar*len(pcd) ==  incnt
    TT.append(time.time())

    ##statistic
    valid_voxel = []
    invalid_voxel = []
    for v in voxels:     
        v.calculate_NDT() 
        if(v.valid):
            valid_voxel.append(v)
        else:
            invalid_voxel.append(v)
    TT.append(time.time())
    # print(f"allocate_PTS_voxel_time:{TT[3]-TT[2]}s, with {len(pcd)} points")
    # print(f"Voxel_init_time:{TT[2]-TT[1]}s, allocate_PTS_voxel_time:{TT[3]-TT[2]}s, calculate_NDT_time:{TT[4]-TT[3]}s")
    return valid_voxel, invalid_voxel, voxels


def NDT_score(a, b, mixed_pdf=True): #-sum(pdf)
    # pdb.set_trace()
    a_array = np.array([(va.x, va.y, va.z) for va in a])
    b_array = np.array([(vb.x, vb.y, vb.z) for vb in b])
    pairs = []
    #find closest match voxel pair
    for i, va in enumerate(a_array):
        dist = np.sum((b_array - va)**2, axis=1)
        closest_index = np.argmin(dist)
        min_dist = dist[closest_index]
        pairs.append((i,closest_index))
    global_ndt_score = 0
    pts_cnt=0
    for (i,j) in pairs:
        score = NDT_voxel_score(a[i], b[j], mixed_pdf)
        global_ndt_score += score
        pts_cnt += len(a[i].pts)
    # print(f"pts in voxel:{pts_cnt}/{4096*8}")
    ##avg per pts, and scalar to expand difference
    scalar = 4096*8
    avg_ndt_score_per_pts = scalar*global_ndt_score/pts_cnt
    return avg_ndt_score_per_pts
        
def NDT_voxel_score(a, b, mixed_pdf=True):
    if(mixed_pdf):
        pdf_scores = b.NDTpdf.mixed_pdf(a.pts)
    else:
        pdf_scores = b.NDTpdf.pdf(a.pts)
    ndt_score = -np.sum(pdf_scores)
    mean_scores = np.mean(pdf_scores)
    assert ndt_score != -np.inf
    return ndt_score

def adjust_covariance_eigenvalues(covariance_matrix):
    # return covariance_matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    min_eigenvalue = np.min(eigenvalues)
    max_eigenvalue = np.max(eigenvalues)
    if min_eigenvalue < 0.001 * max_eigenvalue:
        min_eigenvalue = 0.001 * max_eigenvalue
    else: 
        return covariance_matrix        
    adjusted_eigenvalues = np.diag(np.maximum(eigenvalues, min_eigenvalue))
    adjusted_covariance_matrix = np.matmul(np.matmul(eigenvectors, adjusted_eigenvalues), np.linalg.inv(eigenvectors))

    return adjusted_covariance_matrix

class NDT_Voxel:
    def __init__(self, x,y,z,l,w,h,voxel_size=0.5,min_pts_voxel=5,noise=0.05):
        self.x = x
        self.y = y
        self.z = z
        self.l = l
        self.w = w
        self.h = h
        self.pts = []
        self.voxel_size = voxel_size
        self.min_pts_voxel = min_pts_voxel
        self.noise = noise
        self.valid = True
        self.mean = None
        self.cov = None
        self.NDTpdf = None
        
    def calculate_NDT(self):
        self.pts = np.array(self.pts)    
        if(len(self.pts) >= self.min_pts_voxel):
            #防止矩陣退化成singular，如果解出來是complex就invalid
            cov = adjust_covariance_eigenvalues(np.cov(self.pts,rowvar=False))      
            if(np.iscomplexobj(cov)):
                self.valid = False
                return
            
            self.mean = np.mean(self.pts,0)
            self.cov = cov
            # self.cov_inv = np.linalg.pinv(self.cov) 
            self.NDTpdf = PDF(self.mean, self.cov, self.voxel_size, self.noise)
            # self.pdf = scipy.stats.multivariate_normal(mean=self.mean, cov=self.cov, allow_singular=True).pdf
        else:
            self.valid = False
    def __str__(self):
        return f"x:{self.x}, y:{self.y}, z:{self.z}, l:{self.l}, w:{self.w}, h:{self.h}, pts:{len(self.pts)}, mean:{self.mean}, cov:{self.cov}"
   
    def get_direction(self, p):
        return np.sign([p[0]-self.x,p[1]-self.y,p[2]-self.z])


def draw_NDT_voxel(vs, random=True,NUM_SAMPLES=100,drawbox=False):
    if(drawbox):
        pts = []
        box=[]
        for v in vs:
            pts.extend(v.pts.tolist())
            bb=make_bbox({
                'x':v.x,'y':v.y,'z':v.z,'l':v.l,'w':v.w,'h':v.h
            })
            box.append(bb)
            # draw_all(box)

        pts = make_pcd(np.array(pts))
        # draw_pcd(pts)
        box.append(pts)
        draw_all(box)
        return
    
    if(random):
        random_sample_pts = []
        for v in vs:
            samples = np.random.multivariate_normal(v.mean, v.cov, NUM_SAMPLES)
            random_sample_pts.append(samples.tolist())       
        random_sample_pts = np.array(random_sample_pts).reshape(-1,3) 
        draw_pcd(random_sample_pts)
    else:
        pts = []
        for v in vs:
            pts.extend(v.pts.tolist())
        pts = np.array(pts)    
        draw_pcd(pts)
    
    return 