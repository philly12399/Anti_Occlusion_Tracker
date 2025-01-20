import os
import Philly_libs.philly_io as io_utils
from Philly_libs.NDT import *

def draw_NDT_trackid(EXP_PATH,mem=None):
    merged_track_exp_path=os.path.join(EXP_PATH,f"car_all") 
    trackid_exp_path=os.path.join(EXP_PATH,"car")
    mode_list=["merge","frame"]     
    cnt=0
    while True:
        mode = mode_list[cnt]
        print(f"mode: {mode}, (switch/s to change mode)")
        in1 = input("track id:")
        try:
            if(in1=='q'):    break
            elif(in1=='switch' or in1 == 's'):
                cnt=(cnt+1)%len(mode_list)
                print(f"switch to {mode_list[cnt]} mode")
                continue
            else:   
                if(mode=="merge"):
                    trackid = str(in1).zfill(4)
                    print(f"draw trackid:{trackid}")
                    dense_path =os.path.join(merged_track_exp_path,f"{trackid}_NDT.pkl")
                    
                elif(mode=="frame"): 
                    in_frame = in1.split(' ')
                    trackid = str(in_frame[0]).zfill(4)
                    frameid = str(in_frame[1]).zfill(4)
                    if(mem!=None):
                        frameid = str(mem[trackid][int(in_frame[1])]).zfill(4)
                    dense_path =os.path.join(trackid_exp_path,f"{trackid}_{frameid}.pkl")
                print(dense_path)    
                dense_NDT = io_utils.read_pkl(dense_path)  
                draw_NDT_voxel(dense_NDT,random=False)            
        except:
            print("input error, continue")
            continue
def draw_single_NDT(path):
    print(path)
    dense_path =os.path.join(path)
    dense_NDT = io_utils.read_pkl(dense_path)  
    draw_NDT_voxel(dense_NDT,random=False,drawbox=True)     
    return 

def draw_NDT_dir(root):
    for r in sorted(os.listdir(root)):
        path = os.path.join(root,r)
        draw_single_NDT(path)
    return

if __name__ == '__main__':
    seq=1
    seq = str(seq).zfill(4)
    EXP_PATH="/home/philly12399/philly_ssd/ab3dmot/NDT_pkl/det/demo_gtdets_car_cache/"
    EXP_PATH = os.path.join(EXP_PATH,seq)  
    # merged_mem = io_utils.read_pkl("/home/philly12399/philly_ssd/NDT_EXP/0021/analysis/merged_member.pkl")
    # draw_NDT_trackid(EXP_PATH,mem=merged_mem)   
    draw_NDT_dir(EXP_PATH)
    
    