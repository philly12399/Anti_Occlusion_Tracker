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

def box_graph(avg_diff,avg_same,mode,norm=False): 
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
        
if __name__ == '__main__':
    PATH="/home/philly12399/philly_ssd/NDT_EXP/0021/analysis/"
    modes=["merge_merge","frame_merge"]
    for mode in modes:
        member=list(io_utils.read_pkl(os.path.join(PATH,"merged_member.pkl")).keys())
        avg_diff=io_utils.read_pkl(os.path.join(PATH,"avg_"+mode+"_diff.pkl"))
        avg_same=io_utils.read_pkl(os.path.join(PATH,"avg_"+mode+"_same.pkl"))
        map_score=io_utils.read_pkl(os.path.join(PATH,"map_"+mode+"_score.pkl"))
        # density_graph(avg_diff,avg_same,mode)
        box_graph(avg_diff,avg_same,mode,norm=False)
        
    