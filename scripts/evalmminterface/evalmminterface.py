import argparse
import torch
from tqdm import tqdm
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
# 绘图库
from matplotlib import pyplot as plt
import numpy as np

import sys, os
# 获取脚本所在的目录
current_file_directory = os.path.dirname(os.path.abspath(__file__))
# 设置为 python 脚本当前所在的目录
os.chdir(current_file_directory)
sys.path.append("/d/lsy/pythonworkspace/llava_test/utils")

from utils.metrics import id_correlation
from utils.intrinsic_dimension import estimate_id

# def expectation_compute(corrnorm_list: List[float], bin_start: float, bin_end: float, bin_width: float):
#     bins = np.arange(bin_start, bin_end + bin_width, bin_width)
#     hist, bin_edges = np.histogram(corrnorm_list, bins=bins)

#     #compute expectation numeric
#     expectation_numeric: float = 0.0
#     for i, hist_value in enumerate(hist):
#         frequency = hist_value / len(corrnorm_list)
#         expectation_numeric += frequency * (bin_edges[i] + (bin_width / 2))
    
#     return expectation_numeric

# def draw_corr_columnar_distribution(bin_start_list: List[float], bin_end_list: List[float], bin_width_list: List[float], data_list: List[List[float]], name_list: List[str], color_list: List[str], xlabel_list: List[str], batch_size: int):
#     fig, axs = plt.subplots(2, 3, figsize=(25, 20))  # 创建一个包含 2 行 3 列子图的图形
#     fig.tight_layout(pad=6.0)  # 子图之间的间距
#     for i, ax in enumerate(axs.flat):  # axs.flat 会将二维子图展平，便于迭代
#         if i < len(data_list):
#             bins = np.arange(bin_start_list[i], bin_end_list[i] + bin_width_list[i], bin_width_list[i])  # 使用numpy的histogram函数来分割数据
#             hist, bin_edges = np.histogram(data_list[i], bins=bins)
            
#             # 计算各项指标的期望
#             expectation_numeric = expectation_compute(data_list[i], bin_start_list[i], bin_end_list[i], bin_width_list[i])
#             # 绘制柱状图
#             ax.bar(bin_edges[:-1], hist, width=bin_width_list[i], align='edge', edgecolor='black', color=color_list[i])
            
#             # 设置标签
#             ax.set_xlabel(xlabel_list[i])
#             ax.set_ylabel('Frequency')
#             ax.set_title(f'Distribution of {name_list[i]} with batch_size={batch_size}')
            
#             xScale = [f'{bin_edges[j]:.2f}-{bin_edges[j+1]:.2f}' for j in range(len(bin_edges)-1)]  # 设置x轴刻度
#             ax.set_xticks(bin_edges[:-1])
#             ax.set_xticklabels(xScale, rotation=45)  # 将标签旋转45度，防止重叠
#             ax.set_xlim(bin_edges[0], bin_edges[-1])  # 设置x轴的范围
            
#             # 在每个柱子上方显示频率数值
#             for j in range(len(hist)):
#                 if hist[j] > 0:  # 只对频率大于0的柱子标识
#                     ax.text(bin_edges[j] + bin_width_list[i] / 2, hist[j], str(hist[j]), ha='center', va='bottom', fontsize=9, color='black')
            
#             ax.legend([f'{name_list[i]} freq\nExpectation: {expectation_numeric:.2f}'], loc='upper right', fontsize=14)  # 添加图例

#     fig.delaxes(axs[1, 2])  # 删除右下角的空白子图
#     plt.show()




def eval_mm_interface(args: argparse.Namespace) -> None:
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--textRepPath", type=str, default="../../representation/LLaVA-CC3M-Pretrain-595K/oldextra/stage1/trainjson_ck2000_qwen2.5_3B_Instruct_clipvL14_model_text.pt")
    parser.add_argument("--imageRepPath", type=str, default="../../representation/LLaVA-CC3M-Pretrain-595K/oldextra/stage1/trainjson_ck2000_qwen2.5_3B_Instruct_clipvL14_model_image.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--bin_start_list", type=float, nargs="+", default=[0.0, 0.0, 0.0, 0.0, 0.0])
    parser.add_argument("--bin_end_list", type=float, nargs="+", default=[1.0, 5.0, 40.0, 1.0, 30.0])
    parser.add_argument("--bin_width_list", type=float, nargs="+", default=[0.05, 1.0, 4.0, 0.05, 3.0])
    parser.add_argument("--name_list", type=str, nargs="+", default=["idcor", "text_id", "image_id", "p", "merge_id"])
    parser.add_argument("--color_list", type=str, nargs="+", default=["red", "green", "yellow", "blue", "pink"])
    parser.add_argument("--xlabel_list", type=str, nargs="+", default=["idcor with text_embeds and image_embeds Interval",
                                                                       "id for text_embeds Interval",
                                                                       "id for image_embeds Interval",
                                                                       "pvalues with corr of text_embeds and image_embeds Interval",
                                                                       "merge_id with text_embeds and image_embeds Interval"])
    args = parser.parse_args()
