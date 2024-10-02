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

def density_graph(avg_diff,avg_same,mode,norm=True): 
    values_A = list(avg_diff.values())
    values_B = list(avg_same.values())
    # 計算 A 和 B 的均值和標準差
    mean_A, std_A = np.mean(values_A), np.std(values_A)
    mean_B, std_B = np.mean(values_B), np.std(values_B)

    # 生成基於均值和標準差的常態分布數據
    # 使用與原數據大小一致的 N 生成新數據
    generated_A = np.random.normal(loc=mean_A, scale=std_A, size=10000)
    generated_B = np.random.normal(loc=mean_B, scale=std_B, size=10000)
    # 設置圖表風格
    sns.set(style="whitegrid")

    # 創建畫布
    plt.figure(figsize=(10, 6))

    # 繪製 KDE 圖
    if(norm):
        sns.kdeplot(generated_A, color='blue', shade=True, label='Diff Distribution',common_norm=False)
        sns.kdeplot(generated_B, color='green', shade=True, label='Same Distribution',common_norm=False)
    else:
        sns.kdeplot(values_A, color='blue', shade=True, label='Diff Distribution',common_norm=False)
        sns.kdeplot(values_B, color='green', shade=True, label='Same Distribution',common_norm=False)

    # 添加圖例和標題
    plt.legend()
    if(norm):
        plt.title(f'Density Plot of Diff/Same Normal Distributions of {mode}')
    else:
        plt.title(f'Density Plot of Diff/Same Distributions of {mode}')
        
    plt.xlabel('NDT Score')
    plt.ylabel('Density')

    # 顯示圖表
    plt.show()

import plotly.express as px
import pandas as pd

def violin_graph(avg_diff,avg_same,mode,norm=False): 
    values_A = list(avg_diff.values())
    values_B = list(avg_same.values())
    # 計算 A 和 B 的均值和標準差
    mean_A, std_A = np.mean(values_A), np.std(values_A)
    mean_B, std_B = np.mean(values_B), np.std(values_B)

    # 生成基於均值和標準差的常態分布數據
    # 使用與原數據大小一致的 N 生成新數據
    generated_A = np.random.normal(loc=mean_A, scale=std_A, size=100)
    generated_B = np.random.normal(loc=mean_B, scale=std_B, size=100)
    if(norm):
        values_A=list(generated_A) 
        values_B=list(generated_B) 
    
    # 構建一個 DataFrame 給 Plotly 繪圖使用
    df = pd.DataFrame({
    'Group': ['Diff'] * len(values_A) + ['Same'] * len(values_B),
    'NDT Score': values_A + values_B
    })

    # 使用 Plotly Express 繪製箱型圖，並顯示所有數據點
    if(norm):
        title=f"Violin Plot of Diff/Same Normal Distributions of {mode}"
    else:
        title=f"Violin Plot of Diff/Same Distributions of {mode}"

    fig = px.violin(df, x='Group', y='NDT Score', color='Group', box=False, points="all", title=title)
    # 顯示p圖表
    fig.write_image(f"./{mode}_violin.png")
    
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import itertools
def pca(vec):
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
    # colors = itertools.cycle(plt.cm.rainbow(np.linspace(0, 1, len(vec))))  # 生成不同顏色
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = itertools.cycle(color_list)
    # markers = itertools.cycle(('o', 's', 'v', '^', '<', '>', 'd', 'p', '*', 'h'))  # 不同符號 (marker)
    start_idx = 0  # 起始索引

    plt.figure(figsize=(8, 6))
    np.random.seed(0)
    rand_x = generate_random_array(len(vec), 10)
    # rand_x = [1 for i in range(10)] + [0 for i in range(100)]
    # rand_x = [1 for i in range(100)] 
    cnt = 0
    
    for track_name, track_data in vec.items():
        # 根據每個 track 的數量劃分數據
        
        num_points = len(track_data)
        end_idx = start_idx + num_points
        
        # 繪製每個 track 的數據，顏色不同
        if(rand_x[cnt]==1):
            clr = next(colors)
            plt.scatter(X_pca[start_idx:end_idx, 0], X_pca[start_idx:end_idx, 1], color=clr,label=track_name)
            plt.text(X_pca[start_idx:end_idx, 0], X_pca[start_idx:end_idx, 1], f'{int(track_name)}',color=clr, fontsize=12, ha='right')
        start_idx = end_idx  # 更新下一個 track 的起始索引
        cnt += 1
        
    plt.title('PCA Visualization of Tracks')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()
    
    
def generate_random_array(N, x):
    # 生成全為0的陣列
    array = np.zeros(N, dtype=int)
    
    # 隨機選擇 x 個位置將其設為 1
    indices = np.random.choice(N, x, replace=False)
    array[indices] = 1
    
    return array

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
    frame_merge = io_utils.read_pkl(os.path.join(PATH,"frame_merge_score_all_frame.pkl"))
    # pca(frame_merge)
    
    merge_merge = io_utils.read_pkl(os.path.join(PATH,"map_merge_merge_score.pkl"))
    for key in merge_merge.keys():
        arr = [merge_merge[key][k] for k in merge_merge[key]]
        merge_merge[key] = [arr]
    pca(merge_merge)
    