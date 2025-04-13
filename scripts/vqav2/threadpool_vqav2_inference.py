import subprocess
import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import json, os
import argparse

@dataclass
class InferenceConfig:
    model_path: str = field(
        default="../../../qwen2.5_3B_Instruct_clipvL14_model/model001", metadata={"help": "Path to the base model."}
    )
    lora_path: str = field(
        default="../../../result_model/stage2/[v2.finetune_sqa]/trainjson_ck2000_qwen2.5_3B_Instruct_clipvL14", metadata={"help": "Path to the lora parameter."}
    )
    device: str = field(
        default="cuda:0", metadata={"help": "Device to run the model on."}
    )
    question_file: str = field(
        default="../../../playground/eval/scienceqa/llava_test_CQM-A.json", metadata={"help": "Path to the question file."}
    )
    image_folder: str = field(
        default="../../../playground/eval/scienceqa/images/test", metadata={"help": "Path to the image folder."}
    )
    answers_file: str = field(
        default="../../../playground/eval/scienceqa/answers/finetune3epochs/[new_prompt]llava-trainjson_ck2000-qwen3BInstruct-clipvL14.jsonl", metadata={"help": "Path to the answers file to store the inference result."}
    )
    temperature: float = field(
        default=0, metadata={"help": "model generate temperature."}
    )
    num_chunks: int = field(
        default=1, metadata={"help": "number of chunks to split the test dataset."}
    )
    chunk_idx: int = field(
        default=0, metadata={"help": "index of the chunk to inference."}
    )
    num_beams: int = field(
        default=1, metadata={"help": "number of beams for beam search."}
    )
    max_new_tokens: int = field(
        default=1024, metadata={"help": "max new tokens to generate."}
    )


def run_inference(inference_config: InferenceConfig):
    command = [
        "python", "/home/lsy/workspace/papercode/LLaVA/llava/eval/model_vqa_loader_exp0.py",
        "--model-path", inference_config.model_path,
        "--lora-path", inference_config.lora_path,
        "--device", inference_config.device,
        "--question-file", inference_config.question_file,
        "--image-folder", inference_config.image_folder,
        "--answers-file", inference_config.answers_file,
        "--num-chunks", str(inference_config.num_chunks),
        "--chunk-idx", str(inference_config.chunk_idx),
        "--temperature", str(inference_config.temperature),
        "--num_beams", str(inference_config.num_beams),
        "--max_new_tokens", str(inference_config.max_new_tokens)
    ]
    print(f"Running inference command: {command}")
    subprocess.run(command)

def ThreadPool_Execute(num_workers: int, inferenceconfig_list: List[InferenceConfig]):
    # 使用线程池并行执行num_workers个任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for inference_config in inferenceconfig_list:
            future = executor.submit(run_inference, inference_config)   # 向线程池提交模型推理任务
            futures.append(future)  # 追加到任务状态列表
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer-configs", type=str, default="./infer_configs.json", help="Path to the inference config's json file.")
    args = parser.parse_args()
    with open(args.infer_configs) as infer_config_file:
        configs = json.load(infer_config_file)
    
    question_file = configs['question-file']
    image_folder = configs['image-folder']
    num_chunks = configs['num-chunks']
    chunk_idx = configs['chunk-idx']
    temperature = configs['temperature']
    num_beams = configs['num_beams']
    max_new_tokens = configs['max_new_tokens']
    models = configs['models']
    
    inferenceconfig_list: List[InferenceConfig] = []
    for model in models:
        inferenceconfig_list.append(InferenceConfig(model_path=model['model-path'],
                                                    lora_path=model['lora-path'],
                                                    device=model['device'],
                                                    question_file=question_file,
                                                    image_folder=image_folder,
                                                    answers_file=model['answers-file'],
                                                    temperature=temperature,
                                                    num_chunks=num_chunks,
                                                    chunk_idx=chunk_idx,
                                                    num_beams=num_beams,
                                                    max_new_tokens=max_new_tokens))
    
    ThreadPool_Execute(1, inferenceconfig_list=inferenceconfig_list)