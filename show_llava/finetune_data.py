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

class finetune_SupervisedDataset(Dataset):
    """ Dataset for supervised fine-tuning in Llava stage2 train """
    
    def __init__(self, data_path: str, image_path: str) -> None:
        super().__init__()
        self.data_path = data_path
        self.image_path = image_path
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
        image_path = Path(self.image_path).joinpath(cur_data.get('image'))
        return (human_input, gpt_output, image_path)