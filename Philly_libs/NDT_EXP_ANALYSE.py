import numpy as np
import open3d as o3d
import os
# import click
import sys
import pdb
import Philly_libs.philly_io as io_utils
import Philly_libs.plot as plot
from Philly_libs.NDT import *
from Philly_libs.philly_utils import DictToObj

import matplotlib.pyplot as plt
import seaborn as sns

def density_graph(avg_diff,avg_same,mode): 
    if(isinstance(avg_diff,dict)):
        values_A = list(avg_diff.values())
        values_B = list(avg_same.values())
    else:
        values_A = avg_diff
        values_B = avg_same
    # 設置圖表風格
    sns.set(style="whitegrid")

    # 創建畫布
    plt.figure(figsize=(20, 12))
    
    # 繪製 KDE 圖

    sns.kdeplot(values_A, color='blue', shade=True, label='Different',common_norm=False)
    sns.kdeplot(values_B, color='green', shade=True, label='Same',common_norm=False)

    # 添加圖例和標題
    plt.legend(fontsize=20)

    plt.title(f'Density Plot of {mode} NDT Score Distributions', fontsize=20)
    plt.xlabel('NDT Score', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # 顯示圖表
    plt.show()


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import itertools
def pca(vec,mode):
    track_vec = [np.array(value) for value in vec.values()]
    X = np.vstack(track_vec)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # 2. 進行 PCA
    pca = PCA(n_components=2)  # 你可以選擇保留的主成分數量，這裡保留2個主成分
    X_pca = pca.fit_transform(X_scaled)

    # 輸出結果
    print("原始數據的形狀:", X.shape)
    print("降維後的數據形狀:", X_pca.shape)
    truck=['11', '20', '22', '27', '29', '32', '48', '49', '54', '55', '56']
    shift=[51,14,16,56,35,50,22]
    start_idx = 0  # 起始索引

    plt.figure(figsize=(20, 12))

    cnt = 0
    used={}
    for track_name, track_data in vec.items():
        # 根據每個 track 的數量劃分數據
        
        num_points = len(track_data)
        end_idx = start_idx + num_points
        
        # 繪製每個 track 的數據，顏色不同
        if(str(int(track_name)) in truck):
            clr = 'blue'
            lb='Truck'
        else:
            clr = 'red'
            lb='Car'
        if(lb not in used):
            plt.scatter(X_pca[start_idx:end_idx, 0], X_pca[start_idx:end_idx, 1], color=clr,label=lb,s=25)
            used[lb]=1
        else:
            plt.scatter(X_pca[start_idx:end_idx, 0], X_pca[start_idx:end_idx, 1], color=clr,s=25)
        if(int(track_name) in shift):
            print(track_name)
            plt.text(X_pca[start_idx:end_idx, 0], X_pca[start_idx:end_idx, 1], f'{int(track_name)}',color='black', fontsize=14, ha='left',va='bottom')
        else:
            plt.text(X_pca[start_idx:end_idx, 0], X_pca[start_idx:end_idx, 1], f'{int(track_name)}',color='black', fontsize=14, ha='right',va='bottom')
        start_idx = end_idx  # 更新下一個 track 的起始索引
        cnt += 1
        
    plt.title(f'PCA Visualization of {mode} NDT Score',fontsize=20)
    plt.xlabel('Principal Component 1',fontsize=16)
    plt.ylabel('Principal Component 2',fontsize=16)
    plt.legend(fontsize=20,markerscale=2)
    plt.show()
    plt.savefig("test.svg")
    
def generate_random_array(N, x):
    # 生成全為0的陣列
    array = np.zeros(N, dtype=int)
    
    # 隨機選擇 x 個位置將其設為 1
    indices = np.random.choice(N, x, replace=False)
    array[indices] = 1
    
    return array
def frame_merge_all(PATH):
    frame_merge = io_utils.read_pkl(os.path.join(PATH,"frame_merge_score_all_frame.pkl"))
    same=[]
    diff=[]
    for i,trackid_i in enumerate(frame_merge):
        # same[trackid_i]=[]
        # diff[trackid_i]=[]
        for frame in frame_merge[trackid_i]:
            for si in range(len(frame)):
                if(i==si):
                    # same[trackid_i].append(frame[si])   
                    same.append(frame[si])           
                else:
                    # diff[trackid_i].append(frame[si]) 
                    diff.append(frame[si])   
    print(len(same),len(diff))
    density_graph(diff,same,"Frame-Merged")

def merge_merge_all(PATH):
    merge_merge = io_utils.read_pkl(os.path.join(PATH,"map_merge_merge_score.pkl"))
    same=[]
    diff=[]
    for trackid_i in merge_merge:
        for trackid_j in merge_merge:
                score = merge_merge[trackid_i][trackid_j]
                if(trackid_i==trackid_j):
                    # same[trackid_i].append(frame[si])   
                    same.append(score)           
                else:
                    # diff[trackid_i].append(frame[si]) 
                    diff.append(score)   
    density_graph(diff,same,"Merged-Merged")
def pca_old(vec):
    track_vec = [np.array(value) for value in vec.values()]
    X = np.vstack(track_vec)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. 進行 PCA
    pca = PCA(n_components=2)  # 你可以選擇保留的主成分數量，這裡保留2個主成分
    X_pca = pca.fit_transform(X_scaled)

    # 輸出結果
    print("原始數據的形狀:", X.shape)
    print("降維後的數據形狀:", X_pca.shape)
    print("主成分解釋的方差比例:", pca.explained_variance_ratio_)
    print("主成分:", pca.components_)
    num_colors=10
    colors = [
        '#FF0000',  # 紅色
        '#FF7F00',  # 橙色
        '#0000FF',  # 藍色
        '#4B0082',  # 靛藍色
        '#9400D3',  # 紫色
        'yellow',
        'green',
        'black',
        'pink',
        'brown'
    ]    
    # markers = ['o', 's', 'D','x'] 
    start_idx = 0

    plt.figure(figsize=(20, 12))
    np.random.seed(1)
    rand_x = generate_random_array(len(vec), 10)
    cnt = 0
    cnt1=0
    legend_handles = [] 
    for track_name, track_data in vec.items():
        num_points = len(track_data)
        end_idx = start_idx + num_points
        
        if(rand_x[cnt] == 1):
            clr = colors[cnt1%len(colors)]
            cnt1+=1
            # mrk=markers[cnt%len(markers)]
            scatter=plt.scatter(X_pca[start_idx:end_idx, 0], X_pca[start_idx:end_idx, 1], 
                        color=clr, label=track_name, s=15) 
            legend_handles.append((scatter, track_name))
        start_idx = end_idx
        cnt += 1
    mode="Frame-Merged"
    plt.title(f'PCA Visualization of {mode} NDT Score',fontsize=20)
    plt.xlabel('Principal Component 1',fontsize=16)
    plt.ylabel('Principal Component 2',fontsize=16)
    legend_handles = sorted(legend_handles, key=lambda x: x[1])  # 按標籤排序
    plt.legend([handle[0] for handle in legend_handles], 
               [handle[1] for handle in legend_handles], fontsize=16)
    plt.show()
       
if __name__ == '__main__':
    PATH="/home/philly12399/philly_ssd/NDT_EXP/0021/analysis/"
    modes=["merge_merge","frame_merge"]
    # for mode in modes:
    #     member=list(io_utils.read_pkl(os.path.join(PATH,"merged_member.pkl")).keys())
    #     avg_diff=io_utils.read_pkl(os.path.join(PATH,"avg_"+mode+"_diff.pkl"))
    #     avg_same=io_utils.read_pkl(os.path.join(PATH,"avg_"+mode+"_same.pkl"))
    #     map_score=io_utils.read_pkl(os.path.join(PATH,"map_"+mode+"_score.pkl"))
    #     # density_graph(avg_diff,avg_same,mode)
    #     violin_graph(avg_diff,avg_same,mode,norm=False)
        
    frame_merge_all(PATH)
    # merge_merge_all(PATH)
    
    # merge_merge = io_utils.read_pkl(os.path.join(PATH,"map_merge_merge_score.pkl"))
    # for key in merge_merge.keys():
    #     arr = [merge_merge[key][k] for k in merge_merge[key]]
    #     merge_merge[key] = [arr]
    # pca(merge_merge,mode="Merged-Merged")
    
    
    # frame_merge = io_utils.read_pkl(os.path.join(PATH,"frame_merge_score_all_frame.pkl"))
    # pca_old(frame_merge)
    
