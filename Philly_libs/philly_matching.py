import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment
from AB3DMOT_libs.dist_metrics import iou, dist3d, dist_ground, m_distance
from Philly_libs.NDT import NDT_score
def compute_bbox_affinity(dets, trks, metric):
    # compute affinity matrix

    aff_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
    for d, det in enumerate(dets):
        for t, trk in enumerate(trks):

            # choose to use different distance metrics
            if 'iou' in metric:    	  dist_now = iou(det, trk, metric)            
            # elif metric == 'm_dis':   dist_now = -m_distance(det, trk, trk_inv_inn_matrices[t])
            elif metric == 'euler':   dist_now = -m_distance(det, trk, None)
            elif metric == 'dist_2d': dist_now = -dist_ground(det, trk)              	
            elif metric == 'dist_3d': dist_now = -dist3d(det, trk)              				
            else: assert False, 'error'
            aff_matrix[d, t] = dist_now

    return aff_matrix
import pdb
import time

def compute_pcd_affinity(NDT_Voxels, track_buf):
    #initialize value ,if invalid pair, the value will remain this
    INVALID_VALUE=1e10
    # compute affinity matrix
    aff_matrix = np.full((len(NDT_Voxels), len(track_buf)), INVALID_VALUE, dtype=np.float32)

    for v, voxels in enumerate(NDT_Voxels):
        if(voxels is None):
            continue
        for t, trkbuf in enumerate(track_buf):
            if(trkbuf.NDT_updated):
                rep_of_trk = trkbuf.NDT_of_track
                # rep_of_trk = trkbuf.NDT_voxels[-1]
            else:
                rep_of_trk = None
            if(rep_of_trk is None):
                continue
            aff_matrix[v, t] = NDT_score(voxels, rep_of_trk)
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
  
def data_association(dets, trks, NDT_voxels, trk_buf, metric, threshold, algm='greedy'):   
    """
    Assigns detections to tracked object

    dets:  a list of Box3D object (detected)
    trks:  a list of Box3D object (predicted)

    Returns 3 lists of matches, unmatched_dets and unmatched_trks, and total cost, and affinity matrix
    """

    # if there is no item in either row/col, skip the association and return all as unmatched
    aff_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
    if len(trks) == 0: 
        return np.empty((0, 2), dtype=int), np.arange(len(dets)), [], 0, aff_matrix
    if len(dets) == 0: 
        return np.empty((0, 2), dtype=int), [], np.arange(len(trks)), 0, aff_matrix		            
    
    # compute affinity matrix
    aff_matrix = compute_bbox_affinity(dets, trks, metric)
    pcd_affinity_matrix = compute_pcd_affinity(NDT_voxels, trk_buf)
    # association based on the affinity matrix
    
    matched_indices,_ = optimize_matching(-aff_matrix, algm)
    # print(matched_indices)
    pcd_matched_indices, pcd_cost = optimize_matching(pcd_affinity_matrix, 'hungar')
    # print(matched_indices,pcd_matched_indices)
    
    # print(pcd_affinity_matrix)
    # print(pcd_matched_indices1,c1)
    # print(pcd_matched_indices2,c2)
    
    # else:
    # 	cost_list, hun_list = best_k_matching(-aff_matrix, hypothesis)

    # compute total fdcost
    cost = 0
    for row_index in range(matched_indices.shape[0]):
        cost -= aff_matrix[matched_indices[row_index, 0], matched_indices[row_index, 1]]

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
    if len(matches) == 0: 
        matches = np.empty((0, 2),dtype=int)
    else: matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_dets), np.array(unmatched_trks), cost, aff_matrix
