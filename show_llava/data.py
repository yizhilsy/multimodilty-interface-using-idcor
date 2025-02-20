from dataclasses import dataclass
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Any, Tuple
from transformers import AutoProcessor
from torch import Tensor
from dataclasses import dataclass

class LlavaDataset(Dataset):
    # 构造函数
    def __init__(self, dataset_dir:str) -> None:
        super().__init__()
        self.chat_data, self.image_dir = self.build_dataset(dataset_dir)

    def build_dataset(self, data_dir:str) -> tuple[List[Dict[str, Any]], Path]:
        data_dir = Path(data_dir)
        chat_file = data_dir.joinpath("chat.json")
        image_dir = data_dir.joinpath("images_dl")
        chat_data = pd.read_json(path_or_buf=chat_file).to_dict(orient="records")
        return chat_data, image_dir
    
    def __len__(self) -> int:
        return len(self.chat_data)
    
    def __getitem__(self, index) -> tuple[str, str, Path]:
        cur_data = self.chat_data[index]
        human_input = cur_data['conversations'][0]['value']
        gpt_output = cur_data['conversations'][1]['value']
        image_path = self.image_dir.joinpath(cur_data.get('image'))

        return (human_input, gpt_output, image_path)

@dataclass
class QaImageOutput:
    q_input_ids: torch.Tensor
    pixel_values: torch.Tensor
    a_input_ids: torch.Tensor

def build_qaimage(processor: AutoProcessor, q_text: str, a_text: str, image_path: Path) -> QaImageOutput:
    # instruction or input or question
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": q_text},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # image
    raw_image = Image.open(fp=image_path)
    # 生成Question部分的向量
    inputs = processor(text=prompt, images=raw_image, return_tensors="pt")
    # 生成Answer部分的向量
    a_input_ids = processor.tokenizer(
        a_text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )["input_ids"]
    return QaImageOutput(
        q_input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        a_input_ids=a_input_ids,
    )

# 定义 collator 函数
class TrainLLavaModelCollator:
    # 构造函数
    def __init__(self, processor: AutoProcessor, IGNORE_INDEX: int) -> None:
        self.processor = processor
        self.ignore_index = IGNORE_INDEX
    
    # 拼接单个样本的 q_input_ids 及 a_input_ids
    def convert_one_piece(self,
                          q_input_ids: torch.Tensor,
                          a_input_ids: torch.Tensor) -> None:
        input_ids = torch.concat(tensors=[
            q_input_ids,
            a_input_ids,
            torch.tensor(data=self.processor.tokenizer.eos_token_id).reshape(1, -1)
        ], axis=1)
        labels = torch.concat([
            torch.full_like(input=q_input_ids, fill_value=self.ignore_index),
            a_input_ids,
            torch.tensor(data=self.processor.tokenizer.eos_token_id).reshape(1, -1)
        ], axis=1)
        return input_ids, labels
    
    def __call__(self, features:List) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        input_ids_list = []
        labels_list = []
        pixel_values = []
        max_input_len_list = []

        for feature in features:
            # 1. 调用 build_qaimage 函数将单个样本转换为张量
            qaimage_output = build_qaimage(
                processor=self.processor,
                q_text=feature[0],
                a_text=feature[1],
                image_path=feature[2]
            )
            # 2. 将单个样本的 q_input_ids 及 a_input_ids 张量拼接
            temp_input_ids, temp_labels = self.convert_one_piece(
                q_input_ids=qaimage_output.q_input_ids,
                a_input_ids=qaimage_output.a_input_ids
            )
            input_ids_list.append(temp_input_ids)
            labels_list.append(temp_labels)
            pixel_values.append(qaimage_output.pixel_values)
            max_input_len_list.append(temp_input_ids.shape[1])
        
        # 对齐 input_ids 和 labels
        max_input_len = max(max_input_len_list)
        final_input_ids = torch.concat([    # 将所有对齐到最大长度后的 input_ids 拼接起来组成 final_input_ids
            torch.concat([  # 遍历每个 input_ids 将它们对齐到最大长度
                torch.full(size=(1, max_input_len - max_input_len_list[index]), fill_value=self.processor.tokenizer.pad_token_id),
                value
            ], axis=1)
            for index, value in enumerate(iterable=input_ids_list)
        ])

        final_labels = torch.concat([    # 将所有对齐到最大长度后的 labels 拼接起来组成 final_labels
            torch.concat([  # 遍历每个 labels 将它们对齐到最大长度
                torch.full(size=(1, max_input_len - max_input_len_list[index]), fill_value=self.ignore_index),
                value
            ], axis=1)
            for index, value in enumerate(iterable=labels_list)
        ])

        # 按照 dim=0 维拼接所有的 pixel_values
        final_pixel_values = torch.concat(pixel_values, axis=0)
        attention_mask = torch.ones_like(final_input_ids)
        # 因对齐而造成的填充部分 attention_mask 置 0
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0
        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "pixel_values": final_pixel_values,
            "attention_mask": attention_mask
        }
