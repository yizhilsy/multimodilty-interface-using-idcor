import subprocess
import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

import json, os, time
import argparse

@dataclass
class ExtraConfig:
    model_name_or_path: str = field(
        default='./qwen2.5_3B_Instruct_clipvL14_model/model001', metadata={"help": "Base model path"}
    )
    lora_name_or_path: str = field(
        default=None, metadata={"help": "Lora model path"}
    )
    bert_name_or_path: str = field(
        default=None, metadata={"help": "Bert model path"}
    )
    data_path: str = field(
        default='/home/lsy/shared_data/liuhaotian/LLaVA-CC3M-Pretrain-595K/chat.json', metadata={"help": "data_path for conversation data"}
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
    subversion: str = field(
        default='v1', metadata={"help": "version sub the dataset(dataset/subversion)"}
    )


def run_extra_rep(extra_config: ExtraConfig):
    command = []
    if extra_config.lora_name_or_path is not None:
        command = [
            "python", "/d/lsy/pythonworkspace/llava_test/custom_eval_idcor.py",
            "--model_name_or_path", extra_config.model_name_or_path,
            "--lora_name_or_path", extra_config.lora_name_or_path,
            "--bert_name_or_path", extra_config.bert_name_or_path,
            "--data_path", extra_config.data_path,
            "--output_representation_name", extra_config.output_representation_name,
            "--device", extra_config.device,
            "--dataset", extra_config.dataset,
            "--subversion", extra_config.subversion
        ]
    else:
        command = [
            "python", "/d/lsy/pythonworkspace/llava_test/custom_eval_idcor.py",
            "--model_name_or_path", extra_config.model_name_or_path,
            "--bert_name_or_path", extra_config.bert_name_or_path,
            "--data_path", extra_config.data_path,
            "--output_representation_name", extra_config.output_representation_name,
            "--device", extra_config.device,
            "--dataset", extra_config.dataset,
            "--subversion", extra_config.subversion
        ]

    print(f"Running extra_representations command: {command}")
    time.sleep(16)
    subprocess.run(command)

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
    parser.add_argument("--extra-configs", type=str, default="./oldextra-configs2k_to_50k.json", help="Path to the eval config's json file.")
    args = parser.parse_args()
    with open(args.extra_configs) as extra_config_file:
        configs = json.load(extra_config_file)

    data_path = configs['data-path']
    dataset = configs['dataset']
    subversion = configs['subversion']
    models = configs['models']
    
    extraconfig_list: List[ExtraConfig] = []
    for model in models:
        extraconfig_list.append(ExtraConfig(
                                          model_name_or_path=model['model_name_or_path'],
                                          lora_name_or_path=model['lora_name_or_path'] if 'lora_name_or_path' in model else None,
                                          bert_name_or_path=model['bert_name_or_path'],
                                          data_path=data_path,
                                          output_representation_name=model['output_representation_name'],
                                          device=model['device'],
                                          dataset=dataset,
                                          subversion=subversion
                                          ))
    
    ThreadPool_Execute(2, extraconfig_list=extraconfig_list)
    

    

        
