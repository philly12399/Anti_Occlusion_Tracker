import pickle
import numpy as np
def read_pkl(path): 
    with open(path, 'rb') as file:
        data = pickle.load(file)
        return data

def read_points_from_bin(bin_file):
    gt_points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)[:,:3]
    return gt_points
