from dataclasses import dataclass
import pandas as pd
import torch
import math
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Optional
from transformers import AutoProcessor
from torch import Tensor
from dataclasses import dataclass
from .constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, MODEL_MAX_LENGTH, MODEL_MAX_Q_LENGTH, MODEL_MAX_A_LENGTH

from dataclasses import dataclass
from torch import Tensor

# 适用于单轮对话的finetune数据集
class ExtraRepDataset(Dataset):
    """ Dataset for supervised fine-tuning in Llava stage2 train """
    
    def __init__(self, data_path: str, image_folder: str) -> None:
        super().__init__()
        self.data_path = data_path
        self.image_folder = Path(image_folder)
        self.chat_data = self.build_dataset(self.data_path)

    def build_dataset(self, data_path: str) -> List[Dict[str, Any]]:
        data_path = Path(data_path)
        chat_data = pd.read_json(path_or_buf=data_path).to_dict(orient="records")
        return chat_data
    
    def __len__(self) -> int:
        return len(self.chat_data)
    
    def __getitem__(self, index) -> tuple[str, str, Path]:
        cur_data = self.chat_data[index]
        human_input = cur_data['conversations'][0]['value']
        gpt_output = cur_data['conversations'][1]['value']
        image_path = None
        # TODO: 检查是否存在键值image，以及是否为None或Nan
        if 'image' in cur_data and cur_data['image'] is not None and not (isinstance(cur_data['image'], float) and math.isnan(cur_data['image'])):
            image_path = self.image_folder.joinpath(cur_data.get('image'))
        return (human_input, gpt_output, image_path)

@dataclass
class QaImageOutput:
    q_input_ids: torch.Tensor
    pixel_values: torch.Tensor
    a_input_ids: torch.Tensor

def preprocess_multimodal(q_text: str):
    if DEFAULT_IMAGE_TOKEN in q_text:
        q_text = q_text.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        q_text = DEFAULT_IMAGE_TOKEN + '\n' + q_text
        q_text = q_text.strip()
    return q_text

def build_qaimage(processor: AutoProcessor, q_text: str, a_text: str, image_path: Path) -> QaImageOutput:
    is_multimodal = image_path is not None
    if is_multimodal: # adjust <image> position to the begin for instruction or input or question
        q_text = preprocess_multimodal(q_text)
    else:   # 非多模态数据添加默认的图片占位符，后续将mask掉
        q_text = DEFAULT_IMAGE_TOKEN + q_text
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": q_text},
    ]
    # 应用模板后将会应用speaker角色及start/end signal
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # 生成Question部分的向量
    raw_image = None
    inputs = None
    if is_multimodal:  # 若原始数据中有图片, 则读取图片
        raw_image = Image.open(fp=image_path)
        inputs = processor(text=prompt, images=raw_image, return_tensors="pt")
    else:   # 若原始数据中没有图片, 生成一个占位图片，并在后续的处理中通过mask忽略掉占位图片
        crop_size = processor.image_processor.crop_size
        raw_image = torch.zeros(3, crop_size['height'], crop_size['width'])
        inputs = processor(text=prompt, images=raw_image, return_tensors="pt", do_rescale=False)
   
    # 生成Answer部分的向量
    # BEGIN_SIGNAL = '<|im_start|>'
    # END_SIGNAL = '<|im_end|>'
    # a_text = a_text + END_SIGNAL + '\n' + BEGIN_SIGNAL
    a_input_ids = processor.tokenizer(
        a_text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )["input_ids"]
    return QaImageOutput(
        q_input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        a_input_ids=a_input_ids
    )

# 定义处理模型各个模态数据的 collator 函数, 用于抽取模型各个模态的特征向量
class ExtraLLavaModelRepCollator:
    def __init__(self, processor: AutoProcessor, MY_IGNORE_INDEX: int, MY_MODEL_MAX_Q_LENGTH: int, MY_MODEL_MAX_A_LENGTH: int) -> None:
        self.processor = processor
        self.ignore_index = MY_IGNORE_INDEX if MY_IGNORE_INDEX is not None else IGNORE_INDEX
        self.model_max_q_length = MY_MODEL_MAX_Q_LENGTH if MY_MODEL_MAX_Q_LENGTH is not None else MODEL_MAX_Q_LENGTH
        self.model_max_a_length = MY_MODEL_MAX_A_LENGTH if MY_MODEL_MAX_A_LENGTH is not None else MODEL_MAX_A_LENGTH
        self.model_max_length = self.model_max_q_length + self.model_max_a_length

    # 拼接单个样本的 q_input_ids 及 a_input_ids
    def convert_one_piece(self,
                          q_input_ids: torch.Tensor,
                          a_input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        # 滑动窗口寻找'<|im_end|>\n'出现的位置
        def SlidingWindow(q_input_ids: torch.Tensor, STOP_SIGNAL: str) -> List[int]:
            stop_input_ids = self.processor.tokenizer(STOP_SIGNAL, return_tensors="pt")["input_ids"].squeeze()
            indices = []
            for i in range(len(q_input_ids) - 1):
                if q_input_ids[i].item() == stop_input_ids[0].item() and q_input_ids[i+1].item() == stop_input_ids[1].item():
                    indices.append(i)
            return indices

        input_ids = torch.concat(tensors=[
            q_input_ids,
            a_input_ids,
            torch.tensor(data=self.processor.tokenizer.eos_token_id).reshape(1, -1)
        ], axis=1)

        STOP_SIGNAL = '<|im_end|>\n'
        indices = SlidingWindow(q_input_ids=q_input_ids.squeeze(), STOP_SIGNAL=STOP_SIGNAL)
        # FIXME: 目前掩码掉system对话中的STOP_SIGNAL, 但不掩码掉user对话中的STOP_SIGNAL
        indices = indices[1:] # system对话中包含<|im_end|>\n，system对话全ignore

        labels = torch.concat([ # 遵循 llava 源码的写法，不ignore每个对话<STOP>作用的位置
            torch.full_like(input=q_input_ids, fill_value=self.ignore_index),
            a_input_ids,
            torch.tensor(data=self.processor.tokenizer.eos_token_id).reshape(1, -1)
        ], axis=1)
        
        for idx in indices: # <|im_end|>\n位置不掩码，模型学习对话如何结束
            labels[:, idx:idx+2] = q_input_ids[:, idx:idx+2]

        return input_ids, labels
    
    def __call__(self, features:List) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        q_input_ids_list = []
        a_input_ids_list = []
        pixel_values = []   # 存储原始图像的像素的张量
        max_q_input_len_list = []
        max_a_input_len_list = []

        for feature in features:
            # 1. 调用 build_qaimage 函数将单个样本转换为张量
            qaimage_output = build_qaimage(
                processor=self.processor,
                q_text=feature[0],
                a_text=feature[1],
                image_path=feature[2]
            )
            
            # 2. 根据 model_max_q_length 和 model_max_a_length 对 q_input_ids 和 a_input_ids 进行左截断
            qaimage_output.q_input_ids = qaimage_output.q_input_ids[:, :self.model_max_q_length]
            qaimage_output.a_input_ids = qaimage_output.a_input_ids[:, :self.model_max_a_length]

            # 3. 将截断后的 q_input_ids 和 a_input_ids 存入结果列表
            q_input_ids_list.append(qaimage_output.q_input_ids)
            a_input_ids_list.append(qaimage_output.a_input_ids)
            pixel_values.append(qaimage_output.pixel_values)
            max_q_input_len_list.append(qaimage_output.q_input_ids.shape[1])
            max_a_input_len_list.append(qaimage_output.a_input_ids.shape[1])
        
        # 对齐 q_input_ids 和 a_input_ids
        max_q_input_len = max(max_q_input_len_list)
        max_a_input_len = max(max_a_input_len_list)

        final_q_input_ids = torch.concat([    # 将所有对齐到最大长度后的 q_input_ids 拼接起来组成 final_q_input_ids
            torch.concat([  # 遍历每个 q_input_ids 将它们对齐到最大长度
                torch.full(size=(1, max_q_input_len - max_q_input_len_list[index]), fill_value=self.processor.tokenizer.pad_token_id),
                value
            ], axis=1)
            for index, value in enumerate(iterable=q_input_ids_list)
        ])

        final_a_input_ids = torch.concat([    # 将所有对齐到最大长度后的 labels 拼接起来组成 final_labels
            torch.concat([  # 遍历每个 labels 将它们对齐到最大长度
                torch.full(size=(1, max_a_input_len - max_a_input_len_list[index]), fill_value=self.processor.tokenizer.pad_token_id),
                value
            ], axis=1)
            for index, value in enumerate(iterable=a_input_ids_list)
        ])

        # 按照 dim=0 维拼接所有的 pixel_values
        final_pixel_values = torch.concat(pixel_values, axis=0)
        
        # 构造标识final_q_input_ids和final_a_input_ids张量中非pad_token_id的部分
        q_attention_mask = torch.ones_like(final_q_input_ids)
        a_attention_mask = torch.ones_like(final_a_input_ids)
        # 因对齐而造成的pad填充部分 attention_mask 置 0
        q_attention_mask[final_q_input_ids == self.processor.tokenizer.pad_token_id] = 0
        a_attention_mask[final_a_input_ids == self.processor.tokenizer.pad_token_id] = 0

        return {
            "q_input_ids": final_q_input_ids,
            "a_input_ids": final_a_input_ids,
            "pixel_values": final_pixel_values,
            "q_attention_mask": q_attention_mask,
            "a_attention_mask": a_attention_mask
        }