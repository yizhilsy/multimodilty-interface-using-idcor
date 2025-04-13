import torch
from tqdm import tqdm
from utils.metrics import id_correlation
from utils.intrinsic_dimension import estimate_id
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

# 绘图库
from matplotlib import pyplot as plt
import numpy as np
import os

def analyze_corr_rep(textRepPath: str, imageRepPath: str, batch_size: int, id_alg: str, shuffleN: int,
                     bin_start_list: List[int], bin_end_list: List[int], bin_width_list: List[int], name_list: List[str], color_list: List[str], xlabel_list: List[str],
                     save_folder: str = None):
    # integral_corr = integralRepcorr_compute(textRepPath=textRepPath, imageRepPath=imageRepPath, id_alg=id_alg, shuffleN=shuffleN)
    corr: CorrelationResult = correlation_compute(textRepPath, imageRepPath, batch_size, id_alg, shuffleN)
    text_id_list = corr.text_id_list
    image_id_list = corr.image_id_list
    idcor_list = corr.idcor_list
    p_list = corr.p_list
    merge_id_list = corr.merge_id_list
    data_list = [idcor_list, text_id_list, image_id_list, p_list, merge_id_list]
    figure_name = textRepPath.split("/")[-1].rsplit('_text.pt', 1)[0]
    # 绘制柱状图并打印
    draw_corr_columnar_distribution(bin_start_list, bin_end_list, bin_width_list, data_list, 
                                    name_list, color_list, xlabel_list, batch_size, figure_name, save_folder)

"""
    NOTE 计算在验证多模态对齐的数据集上不同模态表征的总体corr numeric（一种非切分计算指标分布的方法）
    HACK 当验证多模态对齐的数据集很庞大时, 会要求难以承受的显存和内存开销, 因此会导致显存/内存溢出
"""
def integralRepcorr_compute(textRepPath: str, imageRepPath: str, id_alg: str, shuffleN: int) -> Any:
    text_reps = torch.load(textRepPath)
    image_reps = torch.load(imageRepPath)
    integral_corr = id_correlation(text_reps.to(torch.float32), image_reps.to(torch.float32), shuffleN, id_alg)
    return integral_corr

@dataclass
class CorrelationResult:
    text_id_list: List[float]
    image_id_list: List[float]
    idcor_list: List[float]
    p_list: List[float]
    merge_id_list: List[float]

def correlation_compute(textRepPath: str, imageRepPath: str, batch_size: int, id_alg: str, shuffleN: int) -> CorrelationResult:
    text_reps = torch.load(textRepPath)
    image_reps = torch.load(imageRepPath)
    text_reps_batches = torch.split(text_reps, batch_size)
    image_reps_batches = torch.split(image_reps, batch_size)
    text_id_list, image_id_list, idcor_list, p_list, merge_id_list = [], [], [], [], []
    
    # 以batch_size为一批的大小，计算text_embeds与image_embeds之间的corr
    for batch, (text_embeds, image_embeds) in tqdm(enumerate(zip(text_reps_batches, image_reps_batches))):
        corr = id_correlation(text_embeds.to(torch.float32), image_embeds.to(torch.float32), shuffleN, id_alg)
        text_id_list.append(corr['id1'])
        image_id_list.append(corr['id2'])
        idcor_list.append(corr['corr'])
        p_list.append(corr['p'])
        merge_id_list.append(corr['id'])

    return CorrelationResult(
        text_id_list=text_id_list,
        image_id_list=image_id_list,
        idcor_list=idcor_list,
        p_list=p_list,
        merge_id_list=merge_id_list
    )

def draw_corr_columnar_distribution(bin_start_list: List[float], bin_end_list: List[float], bin_width_list: List[float], data_list: List[List[float]],
                                    name_list: List[str], color_list: List[str], xlabel_list: List[str], batch_size: int, figure_name: str, save_folder: str = None):
    fig, axs = plt.subplots(2, 3, figsize=(25, 20))  # 创建一个包含 2 行 3 列子图的图形
    fig.tight_layout(pad=6.0)  # 子图之间的间距
    for i, ax in enumerate(axs.flat):  # axs.flat 会将二维子图展平，便于迭代
        if i < len(data_list):
            bins = np.arange(bin_start_list[i], bin_end_list[i] + bin_width_list[i], bin_width_list[i])  # 使用numpy的histogram函数来分割数据
            hist, bin_edges = np.histogram(data_list[i], bins=bins)
            
            # 计算各项指标的期望
            expectation_numeric = expectation_compute(data_list[i], bin_start_list[i], bin_end_list[i], bin_width_list[i])
            # 绘制柱状图
            ax.bar(bin_edges[:-1], hist, width=bin_width_list[i], align='edge', edgecolor='black', color=color_list[i])
            
            # 设置标签
            ax.set_xlabel(xlabel_list[i])
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {name_list[i]} with batch_size={batch_size}')
            
            xScale = [f'{bin_edges[j]:.2f}-{bin_edges[j+1]:.2f}' for j in range(len(bin_edges)-1)]  # 设置x轴刻度
            ax.set_xticks(bin_edges[:-1])
            ax.set_xticklabels(xScale, rotation=45)  # 将标签旋转45度，防止重叠
            ax.set_xlim(bin_edges[0], bin_edges[-1])  # 设置x轴的范围
            
            # 在每个柱子上方显示频率数值
            for j in range(len(hist)):
                if hist[j] > 0:  # 只对频率大于0的柱子标识
                    ax.text(bin_edges[j] + bin_width_list[i] / 2, hist[j], str(hist[j]), ha='center', va='bottom', fontsize=9, color='black')
            
            ax.legend([f'{name_list[i]} freq\nExpectation: {expectation_numeric:.2f}'], loc='upper right', fontsize=12)  # 添加图例

    fig.delaxes(axs[1, 2])  # 删除右下角的空白子图
    plt.show()
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, figure_name + ".png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def expectation_compute(corrnorm_list: List[float], bin_start: float, bin_end: float, bin_width: float):
    bins = np.arange(bin_start, bin_end + bin_width, bin_width)
    hist, bin_edges = np.histogram(corrnorm_list, bins=bins)

    #compute expectation numeric
    expectation_numeric: float = 0.0
    for i, hist_value in enumerate(hist):
        frequency = hist_value / len(corrnorm_list)
        expectation_numeric += frequency * (bin_edges[i] + (bin_width / 2))
    
    return expectation_numeric