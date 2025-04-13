import subprocess
import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import json, os
import argparse

@dataclass
class MMConfig:
    textRepPath: str = field(
        default="../../representation/LLaVA-CC3M-Pretrain-595K/oldextra/stage1/trainjson_ck6000_qwen2.5_3B_Instruct_clipvL14_model_text.pt", metadata={"help": "Path to the text rep."}
    )
    imageRepPath: str = field(
        default="../../representation/LLaVA-CC3M-Pretrain-595K/oldextra/stage1/trainjson_ck6000_qwen2.5_3B_Instruct_clipvL14_model_image.pt", metadata={"help": "Path to the image rep."}
    )
    batch_size: int = field(
        default=256, metadata={"help": "batch size for the hist evaluation."}
    )
    id_alg: str = field(
        default="twoNN", metadata={"help": "algorithm for intrinsic dimension estimation."}
    )
    shuffleN: int = field(
        default=100, metadata={"help": "number of shuffles."}
    )
    bin_start_list: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0],
        metadata={"help": "List of bin start values."}
    )
    bin_end_list: List[float] = field(
        default_factory=lambda: [1.0, 5.0, 40.0, 1.0, 30.0],
        metadata={"help": "List of bin end values."}
    )
    bin_width_list: List[float] = field(
        default_factory=lambda: [0.05, 1.0, 4.0, 0.05, 3.0],
        metadata={"help": "List of bin width values."}
    )
    name_list: List[str] = field(
        default_factory=lambda: ["idcor", "text_id", "image_id", "p", "merge_id"],
        metadata={"help": "List of names for the bins."}
    )
    color_list: List[str] = field(
        default_factory=lambda: ["red", "green", "yellow", "blue", "pink"],
        metadata={"help": "List of colors for the bins."}
    )
    xlabel_list: List[str] = field(
        default_factory=lambda: ["idcor with text_embeds and image_embeds Interval",
                                 "id for text_embeds Interval",
                                 "id for image_embeds Interval",
                                 "pvalues with corr of text_embeds and image_embeds Interval",
                                 "merge_id with text_embeds and image_embeds Interval"],
        metadata={"help": "List of x-axis labels for the bins."}
    )
    save_folder: str = field(
        default="../../result_imgs/oldextraalg/LLaVA-CC3M-Pretrain-595K", metadata={"help": "Path to save the result images."}
    )

def run_evalmm(mm_config: MMConfig):
    command = [
        "python", "./evalmminterface.py",
        "--textRepPath", mm_config.textRepPath,
        "--imageRepPath", mm_config.imageRepPath,
        "--batch_size", str(mm_config.batch_size),
        "--id_alg", mm_config.id_alg,
        "--shuffleN", str(mm_config.shuffleN),
        "--bin_start_list", *map(str, mm_config.bin_start_list),
        "--bin_end_list", *map(str, mm_config.bin_end_list),
        "--bin_width_list", *map(str, mm_config.bin_width_list),
        "--name_list", *mm_config.name_list,
        "--color_list", *mm_config.color_list,
        "--xlabel_list", *mm_config.xlabel_list,
        "--save_folder", mm_config.save_folder
    ]
    print(f"Running evalmm command: {command}")
    subprocess.run(command)

def ThreadPool_Execute(num_workers: int, mmconfig_list: List[MMConfig]):
    # 使用线程池并行执行num_workers个任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for mm_config in mmconfig_list:
            future = executor.submit(run_evalmm, mm_config)   # 向线程池提交模型推理任务
            futures.append(future)  # 追加到任务状态列表
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mm-configs", type=str, default="./eval_mm_rep.json", help="Path to the mm config's json file.")
    args = parser.parse_args()
    with open(args.mm_configs) as mm_config_file:
        configs = json.load(mm_config_file)
    
    batch_size = configs['batch_size']
    id_alg = configs['id_alg']
    shuffleN = configs['shuffleN']
    bin_start_list = configs['bin_start_list']
    bin_end_list = configs['bin_end_list']
    bin_width_list = configs['bin_width_list']
    name_list = configs['name_list']
    color_list = configs['color_list']
    xlabel_list = configs['xlabel_list']
    save_folder = configs['save_folder']
    mm_rep_paths = configs['mm_rep_paths']
    
    mmconfig_list: List[MMConfig] = []
    for mm_rep_path in mm_rep_paths:
        mmconfig_list.append(MMConfig(
            textRepPath=mm_rep_path['textRepPath'],
            imageRepPath=mm_rep_path['imageRepPath'],
            batch_size=batch_size,
            id_alg=id_alg,
            shuffleN=shuffleN,
            bin_start_list=bin_start_list,
            bin_end_list=bin_end_list,
            bin_width_list=bin_width_list,
            name_list=name_list,
            color_list=color_list,
            xlabel_list=xlabel_list,
            save_folder=save_folder
        ))
    
    ThreadPool_Execute(1, mmconfig_list=mmconfig_list)
    

    

        
