import subprocess
import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import json, os
import argparse

@dataclass
class EvalConfig:
    base_dir: str = field(
        default="/home/lsy/workspace/papercode/LLaVA/playground/data/eval/scienceqa", metadata={"help": "Base directory for evaluation."}
    )
    result_file: str = field(
        default="../../../playground/eval/scienceqa/answers/finetune3epochs/[new_prompt]llava-trainjson_ck2000-qwen3BInstruct-clipvL14.jsonl", metadata={"help": "Path to the result file."}
    )
    output_file: str = field(
        default="../../../playground/eval/scienceqa/answers/finetune3epochs/[new_prompt]llava-trainjson_ck2000-qwen3BInstruct-clipvL14_output.jsonl", metadata={"help": "Path to the output file."}
    )
    output_result: str = field(
        default="../../../playground/eval/scienceqa/answers/finetune3epochs/[new_prompt]llava-trainjson_ck2000-qwen3BInstruct-clipvL14_result.json", metadata={"help": "Path to the output result file."}
    )


def run_eval_statistics(eval_config: EvalConfig):
    command = [
        "python", "/home/lsy/workspace/papercode/LLaVA/llava/eval/eval_science_qa.py",
        "--base-dir", eval_config.base_dir,
        "--result-file", eval_config.result_file,
        "--output-file", eval_config.output_file,
        "--output-result", eval_config.output_result
    ]
    print(f"Running eval_statistics command: {command}")
    subprocess.run(command)

def ThreadPool_Execute(num_workers: int, evalconfig_list: List[EvalConfig]):
    # 使用线程池并行执行num_workers个任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for eval_config in evalconfig_list:
            future = executor.submit(run_eval_statistics, eval_config)   # 向线程池提交模型推理任务
            futures.append(future)  # 追加到任务状态列表
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-configs", type=str, default="./eval-configs2k_to_50k.json", help="Path to the eval config's json file.")
    args = parser.parse_args()
    with open(args.eval_configs) as eval_config_file:
        configs = json.load(eval_config_file)

    models = configs['models']
    
    evalconfig_list: List[EvalConfig] = []
    for model in models:
        evalconfig_list.append(EvalConfig(base_dir=model['base-dir'],
                                          result_file=model['result-file'],
                                          output_file=model['output-file'],
                                          output_result=model['output-result']))
    
    ThreadPool_Execute(4, evalconfig_list=evalconfig_list)
    

    

        
