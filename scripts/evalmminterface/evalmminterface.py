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

sys.path.append("/home/lsy/workspace/llava_test")  # 添加 llava_test/utils 到路径

# custom package
from utils.metrics import id_correlation
from utils.intrinsic_dimension import estimate_id
from analyze_statistics.analyzeMMRep import analyze_corr_rep, expectation_compute

def eval_mm_interface(args: argparse.Namespace) -> None:
    analyze_corr_rep(args.textRepPath, args.imageRepPath, args.batch_size, args.id_alg, args.shuffleN,
                     args.bin_start_list, args.bin_end_list, args.bin_width_list, args.name_list, args.color_list, args.xlabel_list, args.save_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--textRepPath", type=str, default="../../representation/LLaVA-CC3M-Pretrain-595K/oldextra/stage1/trainjson_ck2000_qwen2.5_3B_Instruct_clipvL14_model_text.pt")
    parser.add_argument("--imageRepPath", type=str, default="../../representation/LLaVA-CC3M-Pretrain-595K/oldextra/stage1/trainjson_ck2000_qwen2.5_3B_Instruct_clipvL14_model_image.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--id_alg", type=str, default="towNN")
    parser.add_argument("--shuffleN", type=int, default=100)
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
    parser.add_argument("--save_folder", type=str, default="../../result_imgs/oldextraalg/LLaVA-CC3M-Pretrain-595K/finetune")
    args = parser.parse_args()
    eval_mm_interface(args=args)
