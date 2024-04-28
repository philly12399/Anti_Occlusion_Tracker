import os
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