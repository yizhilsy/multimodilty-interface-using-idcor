import subprocess
import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import json, os
import argparse

# 获取脚本所在的目录
current_file_directory = os.path.dirname(os.path.abspath(__file__))
# 设置为 python 脚本当前所在的目录
os.chdir(current_file_directory)

@dataclass
class ExtraConfig:
    model_name_or_path: str = field(
        default='./qwen2.5_3B_Instruct_clipvL14_model/model001', metadata={"help": "Base model path"}
    )
    lora_name_or_path: str = field(
        default=None, metadata={"help": "Lora model path"}
    )
    data_path: str = field(
        default='/home/lsy/shared_data/liuhaotian/LLaVA-CC3M-Pretrain-595K/chat.json', metadata={"help": "data_path for conversation data"}
    )
    image_folder: str = field(
        default='/home/lsy/shared_data/liuhaotian/LLaVA-CC3M-Pretrain-595K/images_dl', metadata={"help": "multi-modal image folder"}
    )
    output_representation_name: str = field(
        default='trainjson_ck2000_qwen2.5_3B_Instruct_clipvL14_model', metadata={"help": "output_representation_name"}
    )
    device: str = field(
        default='cuda:0', metadata={"help": "select device"}
    )
    dataset: str = field(
        default='LLaVA-CC3M-Pretrain-595K', metadata={"help": "dataset"}
    )
    model_max_q_length: int = field(
        default=768, metadata={"help": "model_max_q_length"}
    )
    model_max_a_length: int = field(
        default=512, metadata={"help": "model_max_a_length"}
    )
    subversion: str = field(
        default='v1', metadata={"help": "version sub the dataset(dataset/subversion)"}
    )
    subsubstatus: str = field(
        default='finetune', metadata={"help": "status sub the subversion(dataset/subversion/subsubstatus)"}
    )
    extra_batch_size: int = field(
        default=128, metadata={"help": "batch_size for extraing representation"}
    )


def run_extra_rep(extra_config: ExtraConfig):
    command = []
    if extra_config.lora_name_or_path is not None:  # 有lora微调适配参数
        command = [
            "python", "../../extra_mm_representations.py",
            "--model_name_or_path", extra_config.model_name_or_path,
            "--lora_name_or_path", extra_config.lora_name_or_path,
            "--data_path", extra_config.data_path,
            "--image_folder", extra_config.image_folder,
            "--output_representation_name", extra_config.output_representation_name,
            "--device", extra_config.device,
            "--dataset", extra_config.dataset,
            "--model_max_q_length", str(extra_config.model_max_q_length),
            "--model_max_a_length", str(extra_config.model_max_a_length),
            "--subversion", extra_config.subversion,
            "--subsubstatus", extra_config.subsubstatus,
            "--extra_batch_size", str(extra_config.extra_batch_size)
        ]
    else:   # 无lora微调适配参数
        command = [
            "python", "../../extra_mm_representations.py",
            "--model_name_or_path", extra_config.model_name_or_path,
            "--data_path", extra_config.data_path,
            "--image_folder", extra_config.image_folder,
            "--output_representation_name", extra_config.output_representation_name,
            "--device", extra_config.device,
            "--dataset", extra_config.dataset,
            "--model_max_q_length", str(extra_config.model_max_q_length),
            "--model_max_a_length", str(extra_config.model_max_a_length),
            "--subversion", extra_config.subversion,
            "--subsubstatus", extra_config.subsubstatus,
            "--extra_batch_size", str(extra_config.extra_batch_size)
        ]
    print(f"Running extra_representations command: {command}")
    subprocess.run(command) # 子进程在实际工作脚本所在的目录下执行command命令

def ThreadPool_Execute(num_workers: int, extraconfig_list: List[ExtraConfig]):
    # 使用线程池并行执行num_workers个任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for extra_config in extraconfig_list:
            future = executor.submit(run_extra_rep, extra_config)   # 向线程池提交模型推理任务
            futures.append(future)  # 追加到任务状态列表
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-configs", type=str, default="./eval-configs2k_to_50k.json", help="Path to the eval config's json file.")
    args = parser.parse_args()
    with open(args.extra_configs) as extra_config_file:
        configs = json.load(extra_config_file)

    data_path = configs['data-path']
    image_folder = configs['image-folder']
    dataset = configs['dataset']
    subversion = configs['subversion']
    subsubstatus = configs['subsubstatus']
    extra_batch_size = configs['extra_batch_size']
    models = configs['models']
    
    extraconfig_list: List[ExtraConfig] = []
    for model in models:
        extraconfig_list.append(ExtraConfig(
                                          model_name_or_path=model['model_name_or_path'],
                                          lora_name_or_path=model['lora_name_or_path'] if 'lora_name_or_path' in model else None,
                                          data_path=data_path,
                                          image_folder=image_folder,
                                          output_representation_name=model['output_representation_name'],
                                          device=model['device'],
                                          dataset=dataset,
                                          model_max_q_length=model['model_max_q_length'],
                                          model_max_a_length=model['model_max_a_length'],
                                          subversion=subversion,
                                          subsubstatus=subsubstatus,
                                          extra_batch_size=extra_batch_size
                                          ))
    
    ThreadPool_Execute(1, extraconfig_list=extraconfig_list)
    

    

        
