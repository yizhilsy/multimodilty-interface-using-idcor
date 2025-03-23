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
from show_llava.finetune_data import finetune_SupervisedDataset, TrainLLavaModelCollator
from show_llava.data import N24News_LlavaDataset
from show_llava.util import print_trainable_parameters

# 导入IdCor_LlavaForConditionalGeneration
from IdCor_nobert_LlavaForConditionalGeneration import IdCor_nobert_LlavaForConditionalGeneration

# 导入计算id, idcor的package
from utils import intrinsic_dimension, metrics, utils
from utils.metrics import id_correlation
from utils.intrinsic_dimension import estimate_id

# 获取脚本所在的目录
current_file_directory = os.path.dirname(os.path.abspath(__file__))
# 设置为 python 脚本当前所在的目录
os.chdir(current_file_directory)

# 初始化日志
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='./qwen2.5_3B_Instruct_clipvL14_model/model001', help='model_name_or_path')
parser.add_argument('--lora_name_or_path', type=str, default=None, help='lora_name_or_path')
parser.add_argument('--data_path', type=str, default='/home/lsy/shared_data/liuhaotian/LLaVA-CC3M-Pretrain-595K/chat.json', help='data_path for conversation data')
parser.add_argument('--image_folder', type=str, default='/home/lsy/shared_data/liuhaotian/LLaVA-CC3M-Pretrain-595K/images_dl', help='multi-modal image folder')
parser.add_argument('--output_representation_name', type=str, default='trainjson_ck2000_qwen2.5_3B_Instruct_clipvL14_model', help='output_representation_name')
parser.add_argument('--device', type=str, default='cuda:0', help='select device')
parser.add_argument('--dataset', type=str, default='LLaVA-CC3M-Pretrain-595K', help='dataset')
parser.add_argument('--model_max_q_length', type=int, default=768, help='model_max_q_length')
parser.add_argument('--model_max_a_length', type=int, default=512, help='model_max_a_length')
parser.add_argument('--subversion', type=str, default='v1', help='version sub the dataset(dataset/subversion)')

# 指定要训练的模型路径及训练参数工具类
@dataclass
class ModelArguments:
    # 基础模型路径（ lora 微调基于的初始模型）
    model_name_or_path: Optional[str] = field(default="./qwen2.5_3B_Instruct_clipvL14_model/model001")
    # lora 微调结果路径
    lora_name_or_path: Optional[str] = field(default="./result_model/stage2/[v2.finetune_sqa]/trainjson_ck2000_qwen2.5_3B_Instruct_clipvL14")

# 指定数据集路径工具类
@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training conversation data."}
    )
    image_folder: str = field(
        default=None, metadata={"help": "Path to the image folder."}
    )
    # source_length: int = field(default=128)
    # target_length: int = field(default=512)

# TrainLLavaModelCollator数据处理类需要指定的处理参数
@dataclass
class ProcessArguments:
    model_max_q_length: int = field(default=768, metadata={"help": "model_max_q_length"})
    model_max_a_length: int = field(default=512, metadata={"help": "model_max_a_length"})

def load_model_processor(modelargs: ModelArguments, device: str):
    # 读取模型
    model = IdCor_nobert_LlavaForConditionalGeneration.from_pretrained(
        modelargs.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device,
    )
    # 加载processor处理器
    processor = LlavaProcessor.from_pretrained(modelargs.model_name_or_path)

    if modelargs.lora_name_or_path is None:
        pass
    else:
        model = PeftModel.from_pretrained(model, modelargs.lora_name_or_path)   # Loading LoRA weights
        model = model.merge_and_unload()    # Merging LoRA weights
    return model, processor

def load_dataset_collator(processor, dataargs: DataArguments, processargs: ProcessArguments):
    llava_dataset = None
    if args.dataset == "LLaVA-CC3M-Pretrain-595K":
        llava_dataset = finetune_SupervisedDataset(
            dataargs.data_path,  # "/d/lsy/shared_data/liuhaotian/LLaVA-CC3M-Pretrain-595K"
            dataargs.image_folder
        )
    elif args.dataset == "N24News":
        llava_dataset = N24News_LlavaDataset(
            dataargs.data_path  # "/d/lsy/shared_data/N24News"
        )
    data_collator = TrainLLavaModelCollator(processor, -100, processargs.model_max_q_length, processargs.model_max_q_length)
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
    )
    data_args = DataArguments(
        data_path=args.data_path,
        image_folder=args.image_folder
    )
    process_args = ProcessArguments(
        model_max_q_length=args.model_max_q_length,
        model_max_a_length=args.model_max_a_length
    )
    device = args.device

    # 创建保存text_embeds和image_embeds的文件夹
    if not os.path.exists(f'./representation/{args.dataset}/{args.subversion}'):
        os.makedirs(f'./representation/{args.dataset}/{args.subversion}', exist_ok=True)

    model, processor = load_model_processor(modelargs=model_args, device=device)
    eval_dataset, data_collator = load_dataset_collator(processor, data_args, process_args)
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

            # 调用IdCor_LlavaForConditionalGeneration加载的模型新增的extra_imageAndtext_embeddings方法
            image_and_text_embeddings = model.extra_imageAndtext_embeddings(
                processor=processor,
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

    torch.save(all_text_embeds, f'/d/lsy/pythonworkspace/llava_test/representation/{args.dataset}/{args.subversion}/{args.output_representation_name}_text.pt')
    torch.save(all_image_embeds, f'/d/lsy/pythonworkspace/llava_test/representation/{args.dataset}/{args.subversion}/{args.output_representation_name}_image.pt')
