from transformers import AutoModelForCausalLM, AutoProcessor, LlavaForConditionalGeneration 
from PIL import Image
from peft import PeftModel
import torch
import os

# 获取脚本所在的目录
current_file_directory = os.path.dirname(os.path.abspath(__file__))
# 设置为 python 脚本当前所在的目录
os.chdir(current_file_directory)

device = "cuda:0"

# 模型路径
model_name_or_path = "./result_model/stage1/[v2.CC3M-Pretrain-595K]qwen2.5_3B_Instruct_clipvL14/checkpoint-5000"
mini_model_name_or_path = "./mini_model/model001"

# 加载 base 基础模型
model = LlavaForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=mini_model_name_or_path,
    device_map=device,
    torch_dtype=torch.bfloat16
)

processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=model_name_or_path)

# 构造轮次型对话数据
prompt_text = "<image>\nWhat's the content of the image?"
prompt_text_next="Summarize the visual content of the image.\n<image>"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt_text},
]

# 应用对话模板
prompt = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

url = "data/oldwhite.jpeg"
image = Image.open(fp=url)

inputs = processor(images=image, text=prompt, return_tensors="pt")

for temp_key in inputs.keys():
    inputs[temp_key] = inputs[temp_key].to(device)

# Generate
generate_ids = model.generate(**inputs, 
                                max_new_tokens=500,
                                repetition_penalty=1.2
                                )
# 将模型生成的文本 token 序列 解码成可读的字符串形式
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(response)