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

# 基础模型路径（ lora 微调的初始模型）
base_model_path = "./qwen2.5_3B_Instruct_clipvL14_model/model001"
# lora 微调模型路径
lora_model_path = "./output_model_lora_show/qwen2.5_3B_Instruct_clipvL14"

# 加载 base 基础模型
model = LlavaForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=base_model_path,
    device_map=device,
    torch_dtype=torch.bfloat16
)

# 加载 lora 适配器
model = PeftModel.from_pretrained(model, lora_model_path)

processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=base_model_path)

prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
url = "data/australia.jpg"
image = Image.open(fp=url)

inputs = processor(images=image, text=prompt, return_tensors="pt")

for temp_key in inputs.keys():
    inputs[temp_key] = inputs[temp_key].to(device)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=50)
# 将模型生成的文本 token 序列 解码成可读的字符串形式
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(response)