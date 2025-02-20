"""
    借助transformers库中的Trainer类，实现对于训练好的llava模型在特定数据集上计算idcor数值的计算
    eval_idcor_batch_size = 256
"""

import transformers
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Trainer,
    TrainingArguments,
    AutoProcessor
)
from PIL import Image
from peft import PeftModel
import torch
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import evaluate
import os, sys

# custom package

# 导入读取数据集和处理数据集为向量的工具类
from show_llava.data import LlavaDataset, TrainLLavaModelCollator
from show_llava.util import print_trainable_parameters
# 导入计算id, idcor的package
from utils import (
    intrinsic_dimension, metrics, utils
)

# 获取脚本所在的目录
current_file_directory = os.path.dirname(os.path.abspath(__file__))
# 设置为 python 脚本当前所在的目录
os.chdir(current_file_directory)

# 指定gpu
device = "cuda:0"
# 初始化日志
logger = logging.getLogger(__name__)

# 指定要训练的模型路径及训练参数工具类
@dataclass
class ModelArguments:
    # 基础模型路径（ lora 微调基于的初始模型）
    model_name_or_path: Optional[str] = field(default="./qwen2.5_3B_Instruct_clipvL14_model/model001")
    # lora 微调结果路径
    lora_name_or_path: Optional[str] = field(default="./output_model_lora_show/[epoch4-5]qwen2.5_3B_Instruct_clipvL14")

def load_model_processor(modelargs: ModelArguments):
    # 读取模型
    model = LlavaForConditionalGeneration.from_pretrained(
        modelargs.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device

    )
    # 合并lora微调模型
    model = PeftModel.from_pretrained(model, modelargs.lora_name_or_path)
    # 加载processor处理器
    processor = LlavaProcessor.from_pretrained(modelargs.model_name_or_path)
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
    llava_dataset = LlavaDataset(
        dataargs.data_path  # "data/liuhaotian/LLaVA-CC3M-Pretrain-595K"
    )
    data_collator = TrainLLavaModelCollator(processor, -100)

    return llava_dataset, data_collator

# args init
model_args: ModelArguments = ModelArguments(
    model_name_or_path="./qwen2.5_3B_Instruct_clipvL14_model/model001",
    lora_name_or_path="./output_model_lora_show/[epoch4-5]qwen2.5_3B_Instruct_clipvL14"
)

data_args: DataArguments = DataArguments(
    data_path="/home/lsy/shared_data/liuhaotian/LLaVA-CC3M-Pretrain-595K"
)

train_args: TrainingArguments = TrainingArguments(
    output_dir="./eval_idcor_checkpoints",
    per_device_eval_batch_size=8,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=16,
    save_strategy="no",
    save_total_limit=3,
    learning_rate=4e-4,
    include_for_metrics="inputs"
)

acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def eval_idcor_metric(eval_predict):
    predictions, labels, inputs = eval_predict
    print(inputs)

def evaluate(modeling_args: ModelArguments, dataing_args: DataArguments, training_args: TrainingArguments):
     # 将命令行参数解析成 dataclass 对象
    # parser = transformers.HfArgumentParser(
    #     (ModelArguments, DataArguments, TrainingArguments)
    # )
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model, processor = load_model_processor(modeling_args)
    eval_dataset, data_collator = load_dataset_collator(processor, dataing_args)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=eval_idcor_metric
    )

    trainer.evaluate()

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    evaluate(modeling_args=model_args, dataing_args=data_args, training_args=train_args)


