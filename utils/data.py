import pandas as pd
from datasets import load_dataset, Dataset
from PIL import Image
import torch
import os
# 获取脚本所在的目录
current_file_directory = os.path.dirname(os.path.abspath(__file__))
# 设置为脚本所在的目录
os.chdir(current_file_directory)
# 验证当前工作目录
print("data.py当前工作目录:", os.getcwd())

# 获取HF_TOKEN环境变量
hf_token = os.environ.get("HF_TOKEN")
print("从环境变量获取到的hugging face token:", hf_token)

def get_data(dataset_name):
    print("get dataset:" + dataset_name)
    if dataset_name=='coco':
        dataset = load_dataset('json', data_files='./data/annotations/captions_val2014.json', split='all')['annotations'][0]
        dataset=pd.DataFrame.from_dict(dataset).drop_duplicates(subset='image_id')
        dataset = Dataset.from_pandas(dataset)
        def collate(modality, samples):
            if modality=='image':
                images = [Image.open(f'./data/val2014/COCO_val2014_'+str(sample["image_id"]).zfill(12)+'.jpg').convert("RGB") for sample in samples]
                return images
            else:
                texts = [sample['caption'] for sample in samples] 
                return texts
        return dataset, collate
    elif dataset_name=='imagenet':
        dataset=load_dataset('imagefolder', data_dir='/home/lsy/shared_data/imagenet-1k/val', split='validation', streaming=True)
        # 打印 dataset 的类型
        print(type(dataset))

        def collate(modality, samples):
            images = [sample['image'].convert("RGB") for sample in samples]
            labels = torch.tensor([sample['label'] for sample in samples])
            return images, labels
        return dataset, collate
    elif dataset_name=='imagenet_sketch':
        dataset = load_dataset("imagenet_sketch", split="train")

        def collate(modality, samples):
            images = [sample['image'].convert("RGB") for sample in samples]
            labels = torch.tensor([sample['label'] for sample in samples])
            return images, labels
        return dataset, collate
    
    elif dataset_name=='N24News':
        dataset = load_dataset('json', data_files='/home/lsy/shared_data/N24News/news/nytimes.json', split='all')
        print(len(dataset))

        def collate(modality, samples):
            if modality=='image':
                images = [Image.open(f'/home/lsy/shared_data/N24News/imgs/{sample["image_id"]}.jpg').convert("RGB") for sample in samples]
                return images
            else:
                texts = [sample['caption'] for sample in samples] 
                return texts
        return dataset, collate