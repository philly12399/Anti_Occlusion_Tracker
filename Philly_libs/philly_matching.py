import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment
from AB3DMOT_libs.dist_metrics import iou, dist3d, dist_ground, m_distance
from Philly_libs.NDT import NDT_score
import copy
from AB3DMOT_libs.box import Box3D
INVALID_VALUE=1e10

def compute_bbox_affinity(dets, trks, metric, trk_mask=[]): #BIGGER BETTER
    # compute affinity matrix
    global INVALID_VALUE
    aff_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
    for d, det in enumerate(dets):
        for t, trk in enumerate(trks):
            if(trk is None):
                aff_matrix[d, t] = -INVALID_VALUE
                continue
            if(t in trk_mask):
                aff_matrix[d, t] = -INVALID_VALUE
                continue
            # choose to use different distance metrics
            if 'iou' in metric:    	  dist_now = iou(det, trk, metric)            
            # elif metric == 'm_dis':   dist_now = -m_distance(det, trk, trk_inv_inn_matrices[t])
            elif metric == 'euler':   dist_now = -m_distance(det, trk, None)
            elif metric == 'dist_2d': dist_now = -dist_ground(det, trk)              	
            elif metric == 'dist_3d': dist_now = -dist3d(det, trk) 
            elif metric == 'angle':   dist_now = compute_angle_aff(det, trk)       				
            else: assert False, 'error'
            aff_matrix[d, t] = dist_now

    return aff_matrix
import pdb
import time

def multiframe_bbox_affinity(aff_history, history, weight):
    d, t = aff_history[0].shape
    aff_sum = np.zeros((d, t), dtype=np.float32)
    aff_weight = np.zeros((d, t), dtype=np.float32)
    for di in range(d):
        for ti in range(t):
            for h in range(history):
                v = aff_history[h][di][ti]
                if(v != -INVALID_VALUE): #有交集,GIOU3D沒交集會<0
                    aff_sum[di][ti] += weight[h]*v
                    aff_weight[di][ti] += weight[h]
                    
    # no zero weight        
    mask = (aff_weight == 0)
    aff_weight[mask] = 1
    
    aff_sum = np.divide(aff_sum, aff_weight)
    return aff_sum
            
def compute_pcd_affinity(NDT_Voxels, track_buf): #SMALLER BETTER
    #initialize value ,if invalid pair, the value will remain this
    global INVALID_VALUE
    # compute affinity matrix
    aff_matrix = np.full((len(NDT_Voxels), len(track_buf)), INVALID_VALUE, dtype=np.float32)

    for v, voxels in enumerate(NDT_Voxels):
        if(voxels is None):
            continue
        for t, trkbuf in enumerate(track_buf):
            if(trkbuf.NDT_updated):
                rep_of_trk = trkbuf.NDT_of_track
            else:
                rep_of_trk = None
            if(rep_of_trk is None): continue
            aff_matrix[v, t] = NDT_score(voxels, rep_of_trk)
    return aff_matrix

def pcd_affinity_postprocess(aff_matrix, dist, angle, max_dist = 4.0, max_angle=20):
    global INVALID_VALUE
    v,t = aff_matrix.shape
    for i in range(v):
        for j in range(t):
            if(abs(dist[i][j]) > max_dist):
                aff_matrix[i][j] = INVALID_VALUE
            elif(abs(angle[i][j]) > np.deg2rad(max_angle)):
                aff_matrix[i][j] = INVALID_VALUE
    return aff_matrix

def greedy_matching(cost_matrix):
    # association in the greedy manner
    # refer to https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking/blob/master/main.py

    num_dets, num_trks = cost_matrix.shape[0], cost_matrix.shape[1]

    # sort all costs and then convert to 2D
    distance_1d = cost_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_trks, index_1d % num_trks], axis=1)

    # assign matches one by one given the sorting, but first come first serves
    det_matches_to_trk = [-1] * num_dets
    trk_matches_to_det = [-1] * num_trks
    matched_indices = []
    for sort_i in range(index_2d.shape[0]):
        det_id = int(index_2d[sort_i][0])
        trk_id = int(index_2d[sort_i][1])

        # if both id has not been matched yet
        if trk_matches_to_det[trk_id] == -1 and det_matches_to_trk[det_id] == -1:
            trk_matches_to_det[trk_id] = det_id
            det_matches_to_trk[det_id] = trk_id
            matched_indices.append([det_id, trk_id])

    return np.asarray(sorted(matched_indices))

def optimize_matching(matrix, algm='greedy'): #find min
    
    if algm == 'hungar':
        row_ind, col_ind = linear_sum_assignment(matrix)      	# hougarian algorithm, find min cost
        matched_indices = np.stack((row_ind, col_ind), axis=1)
    elif algm == 'greedy':
        matched_indices = greedy_matching(matrix) 				# greedy matching
    else: assert False, 'error'
    cost = 0
    for m1,m2 in matched_indices:
        cost += matrix[m1][m2]
    return matched_indices, cost

def data_association(dets, trks, NDT_Voxels, trk_buf, trk_mask, metric, threshold, algm='greedy', history = 1, NDT_flag=False, NDT_thres = {}):   
    """
    Assigns detections to tracked object

    dets:  a list of Box3D object (detected)
    trks:  a list of Box3D object (predicted)

    Returns 3 lists of matches, unmatched_dets and unmatched_trks, and total cost, and affinity matrix
    """

    # if there is no item in either row/col, skip the association and return all as unmatched
    aff_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
    if len(trks) == 0: 
        return np.empty((0, 2), dtype=int), np.arange(len(dets)), [], 0, aff_matrix,{}
    if len(dets) == 0: 
        return np.empty((0, 2), dtype=int), [], np.arange(len(trks)), 0, aff_matrix,{}		
    #  pred bbox x past time(-1,-2...) -> past time(-1,-2...) x pred bbox
    trks_T = [[row[j] for row in trks] for j in range(len(trks[0]))]
    ##Stage 1 
    # compute affinity matrix of past time 
    aff_matrix_history = [compute_bbox_affinity(dets, trks_T[h], metric,trk_mask) for h in range(history)]
    max_mat = np.max(np.array(aff_matrix_history), axis = 0)
    aff_matrix = aff_matrix_history[0]
    ## association based on the affinity matrix
    matched_indices, cost_bbox = optimize_matching(-aff_matrix, algm)
    # save for unmatched objects
    unmatched_dets = []
    for d, det in enumerate(dets):
        if (d not in matched_indices[:, 0]): unmatched_dets.append(d)
    unmatched_trks = []
    for t, trk in enumerate(trks):
        if (t not in matched_indices[:, 1]): unmatched_trks.append(t)
    # filter out matches with low affinity
    matches = []
    for m in matched_indices:
        if (aff_matrix[m[0], m[1]] < threshold):
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else: matches.append(m.reshape(1, 2))
            
    unmatched_dets = sorted(unmatched_dets)
    unmatched_trks = sorted(unmatched_trks)
    
    ##Stage 2 NDT
    stage2_stat={}
    if(NDT_flag): #if use NDT
        ## First collect unmatched det NDT, exclude None
        unmatched_det_NDT = [NDT_Voxels[i] for i in unmatched_dets]
        valid_det_NDT = [[],[]] # for original id and data
        for i,v in enumerate(unmatched_det_NDT):
            if v!=None:
                valid_det_NDT[0].append(unmatched_dets[i])
                valid_det_NDT[1].append(v)
        
        ## Then collect unmatched trk 
        unmatched_trkbuf = [trk_buf[j] for j in unmatched_trks]
        valid_trkbuf = [[],[]] # for original id and data
        ## if we have "valid det" and "unmatched trk"
        if(len(valid_det_NDT[0])>0 and len(unmatched_trkbuf)>0):   
            ## Do trk.update_NDT 
            for i,t in enumerate(unmatched_trkbuf):
                t.update_NDT()
                ## if NDT update success, add to valid_trkbuf
                if t.NDT_updated==True:
                    valid_trkbuf[0].append(unmatched_trks[i])
                    valid_trkbuf[1].append(t)

            #if we have "valid trkbuf"
            if(len(valid_trkbuf[0])>0):                
                det1 = [dets[i] for i in valid_det_NDT[0]]
                ##FIXED HISTORY = 1, so i use trksT [0]
                trks1 = [trks_T[0][j] for j in valid_trkbuf[0]] 
                trk_buf_bbox=[Box3D.array2bbox(trk_buf[j].bbox[-1]) for j in valid_trkbuf[0]] 
                
                dist = compute_bbox_affinity(det1, trks1, "dist_2d") 
                angle = compute_bbox_affinity(det1, trk_buf_bbox, "angle")
                pcd_affinity_matrix = compute_pcd_affinity(valid_det_NDT[1], valid_trkbuf[1])
                pcd_affinity_matrix_filted = pcd_affinity_postprocess(copy.deepcopy(pcd_affinity_matrix), dist, angle,NDT_thres['max_dist'], NDT_thres['max_angle'])
                matched_indices_NDT, _ = optimize_matching(pcd_affinity_matrix_filted, 'hungar')
                # print(pcd_affinity_matrix)  
                # print(matched_indices_NDT)   
                # pdb.set_trace() 
                pair=[]
                for m in matched_indices_NDT:
                    d1=valid_det_NDT[0][m[0]]
                    t1=valid_trkbuf[0][m[1]]
                    if (pcd_affinity_matrix_filted[m[0], m[1]] < NDT_thres['NDT_score']):
                        matches.append(np.array([d1,t1]).reshape(1, 2))
                        unmatched_dets.remove(d1)
                        unmatched_trks.remove(t1)
                        pair.append([m[0],m[1]])

                stage2_stat['aff'] = pcd_affinity_matrix
                stage2_stat['pair'] = pair
                stage2_stat['NDT_det_index'] = valid_det_NDT[0] 
                NDT_trk_index =[trk_buf[i].id for i in valid_trkbuf[0]]
                stage2_stat['NDT_trk_index'] = NDT_trk_index
                stage2_stat['dist'] = dist
            else:
                stage2_stat['log']= f"{len(valid_trkbuf[0])} valid NDT trkbuf ; {len(unmatched_trks)} unmatched trk"
        else:
            stage2_stat['log']=""
            if(len(valid_det_NDT[0])==0):
                stage2_stat['log'] = f"{len(valid_det_NDT[0])} valid NDT det ; {len(unmatched_dets)} unmatched det\n "
                stage2_stat['unmatched_det'] = True
            if(len(unmatched_trkbuf)==0):
                if(stage2_stat['log']!=""):
                   stage2_stat['log']+= f"\n{len(unmatched_trks)} unmatched trk" 
                else:
                    stage2_stat['log']= f"{len(unmatched_trks)} unmatched trk"
            
    if len(matches) == 0: 
        matches = np.empty((0, 2),dtype=int)
    else: matches = np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_dets), np.array(unmatched_trks), cost_bbox, aff_matrix, stage2_stat

def compute_vector_angle(a, b): 
    # compute the angle of vector a->b
    if(Box3D.label_format=='kitti'):
        angle = angle_normalize(-np.arctan2((b.z-a.z),(b.x-a.x)))  #加負是因為kitti的roty定義是反的(逆時針是負)
    elif(Box3D.label_format=='wayside'):
        angle = angle_normalize(np.arctan2((b.z-a.z),(b.x-a.x)))  
        
    return angle

def compute_angle_aff(det, trk,): ##trk head(前進方向) , bbox angle(associate的連線), close enough
    angle = compute_vector_angle(trk,det)
    return abs(angle_normalize(angle)-angle_normalize(trk.ry))

def angle_normalize(theta):
    if theta >= np.pi: theta -= np.pi * 2    # make the theta still in the range
    if theta < -np.pi: theta += np.pi * 2
    return theta