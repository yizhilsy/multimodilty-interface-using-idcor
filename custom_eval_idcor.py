import transformers
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Trainer,
    TrainingArguments,
    AutoProcessor
)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from PIL import Image
from peft import PeftModel

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import evaluate
import os, sys
import argparse

"""
    custom package
"""

# 导入读取数据集和处理数据集为向量的工具类
from show_llava.origindata import LlavaDataset, TrainLLavaModelCollator
from show_llava.data import N24News_LlavaDataset
from show_llava.util import print_trainable_parameters

# 导入IdCor_LlavaForConditionalGeneration
from IdCor_LlavaForConditionalGeneration import IdCor_LlavaForConditionalGeneration

# 导入计算id, idcor的package
from utils import intrinsic_dimension, metrics, utils
from utils.metrics import id_correlation
from utils.intrinsic_dimension import estimate_id

# 获取脚本所在的目录
current_file_directory = os.path.dirname(os.path.abspath(__file__))
# 设置为 python 脚本当前所在的目录
os.chdir(current_file_directory)

# 指定gpu
device = "cuda:0"
# 初始化日志
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='./result_model/stage1/[v2.CC3M-Pretrain-595K]qwen2.5_3B_Instruct_clipvL14/checkpoint-5000', help='model_name_or_path')
parser.add_argument('--lora_name_or_path', type=str, default=None, help='lora_name_or_path')
parser.add_argument('--bert_name_or_path', type=str, default='./google-bert/bert-base-uncased', help='bert_name_or_path')
parser.add_argument('--data_path', type=str, default='/d/lsy/shared_data/liuhaotian/LLaVA-CC3M-Pretrain-595K', help='data_path')
parser.add_argument('--output_representation_name', type=str, default='alpha_qwen2.5_3B_Instruct_clipvL14_model', help='output_representation_name')
parser.add_argument('--device', type=str, default='cuda:0', help='select device')
parser.add_argument('--dataset', type=str, default='LLaVA-CC3M-Pretrain-595k', help='dataset')

# 指定要训练的模型路径及训练参数工具类
@dataclass
class ModelArguments:
    # 基础模型路径（ lora 微调基于的初始模型）
    model_name_or_path: Optional[str] = field(default="./qwen2.5_3B_Instruct_clipvL14_model/model001")
    # lora 微调结果路径
    lora_name_or_path: Optional[str] = field(default="./output_model_lora_show/[epoch4-5]qwen2.5_3B_Instruct_clipvL14")
    # bert 预训练模型路径
    bert_name_or_path: Optional[str] = field(default="./google-bert/bert-base-uncased")

def load_model_processor(modelargs: ModelArguments):
    # 读取模型
    model = IdCor_LlavaForConditionalGeneration.from_pretrained(
        modelargs.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device,
        pretrained_bert_name_or_path = modelargs.bert_name_or_path
    )
    # 加载processor处理器
    processor = LlavaProcessor.from_pretrained(modelargs.model_name_or_path)

    if modelargs.lora_name_or_path is None:
        pass
    else:
        model = PeftModel.from_pretrained(model, modelargs.lora_name_or_path)   # 合并lora微调模型
    return model, processor

# 指定数据集路径工具类
@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    # source_length: int = field(default=128)
    # target_length: int = field(default=512)

def load_dataset_collator(processor, dataargs: DataArguments):
    llava_dataset = None
    if args.dataset == "LLaVA-CC3M-Pretrain-595k":
        llava_dataset = LlavaDataset(
            dataargs.data_path  # "/d/lsy/shared_data/liuhaotian/LLaVA-CC3M-Pretrain-595K"
        )
    elif args.dataset == "N24News":
        llava_dataset = N24News_LlavaDataset(
            dataargs.data_path  # "/d/lsy/shared_data/N24News"
        )
    data_collator = TrainLLavaModelCollator(processor, -100)
    return llava_dataset, data_collator

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # 将命令行参数解析成 dataclass 对象
    args = parser.parse_args()
    print(args)
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        lora_name_or_path=args.lora_name_or_path,
        bert_name_or_path=args.bert_name_or_path
    )
    data_args = DataArguments(
        data_path=args.data_path
    )
    device = args.device

    # 创建保存text_embeds和image_embeds的文件夹
    if not os.path.exists(f'./representation/{args.dataset}'):
        os.makedirs(f'./representation/{args.dataset}', exist_ok=True)

    model, processor = load_model_processor(model_args)
    eval_dataset, data_collator = load_dataset_collator(processor, data_args)
    dataloader_params = {
        "batch_size": 128,
        "collate_fn": data_collator,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": 4,
        "shuffle": False
    }
    eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
    model.eval()
    with torch.no_grad():
        all_text_embeds = []
        all_image_embeds = []
        for steps, inputs in tqdm(enumerate(eval_dataloader)):
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            human_input = inputs["human_input"]

            # 调用IdCor_LlavaForConditionalGeneration加载的模型新增的extra_imageAndtext_embeddings方法
            image_and_text_embeddings = model.extra_imageAndtext_embeddings(
                human_input=human_input,
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                vision_feature_layer=-2,
                vision_feature_select_strategy="default"
            )

            image_embeds = image_and_text_embeddings.image_embeds
            text_embeds = image_and_text_embeddings.text_embeds
            
            # test 粗略计算一下idcor
            corr = id_correlation(text_embeds.to(torch.float32), image_embeds.to(torch.float32), 100, 'twoNN')
            logging.info(f"corr: {corr}")

            # 保存每个batch_size的image_embeds和text_embeds
            all_text_embeds.append(text_embeds.detach().cpu())
            all_image_embeds.append(image_embeds.detach().cpu())

    # 将所有的image_embeds和text_embeds拼接起来
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    
    logging.info(f"all_text_embeds shape: {all_text_embeds.shape}")
    logging.info(f"all_image_embeds shape: {all_image_embeds.shape}")

    torch.save(all_text_embeds, f'./representation/{args.dataset}/{args.output_representation_name}_text.pt')
    torch.save(all_image_embeds, f'./representation/{args.dataset}/{args.output_representation_name}_image.pt')
