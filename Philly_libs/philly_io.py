import pickle
import numpy as np
def read_pkl(path): 
    with open(path, 'rb') as file:
        data = pickle.load(file)
        return data
    
def write_pkl(data, outpath):  
    with open(outpath, 'wb') as file:
        pickle.dump(data, file)
    return

def pkl_to_txt(path,out):
    data = read_pkl(path)
    f=open(out,'w')
    f.write(str(data))
    return data

def read_points_from_bin(bin_file, unique=False):
    gt_points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)[:,:3]
    if(unique):
        gt_points = np.unique(gt_points,axis=0)
    return gt_points

def read_vis_points(pcd_path):
    arr=[]
    with open(pcd_path, 'r') as file:
        data = file.readlines()
        for d in data:
            d = d.replace('\n','').split(';')
            arr.append([float(d[0]), float(d[1]), float(d[2])])    
    arr = np.array(arr)
    #unique
    arr = np.unique(arr,axis=0)
    return arr