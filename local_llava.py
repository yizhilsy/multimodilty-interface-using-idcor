from PIL import Image
import requests
import torch
import os
# 导入本地可编辑 transformers 库
import transformers
from transformers import AutoProcessor, LlavaForConditionalGeneration

# 获取脚本所在的目录
current_file_directory = os.path.dirname(os.path.abspath(__file__))
# 设置为 python 脚本当前所在的目录
os.chdir(current_file_directory)

# 验证当前工作目录
print("当前工作目录:", os.getcwd())

model_name_or_path = "llava-1.5-7b-hf"
device = "cuda:0"

model = LlavaForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path,
    device_map=device,
    torch_dtype=torch.bfloat16
)

# image_processor: CLIPImageProcessor, tokenizer: LlamaTokenizerFast
processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=model_name_or_path)

prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
prompt2 = "USER: <image>\nWhat animals are in the image and what are they doing? What is next to these animals? What color are these animals? ASSISTANT:"
url = "data/000000039769.jpg"
image = Image.open(fp=url)

inputs = processor(images=image, text=prompt2, return_tensors="pt")

# inputs['input_ids', 'attention_mask', 'pixel_values']，前两个键对应文本输入，第三个键对应图像输入
for temp_key in inputs.keys():
    inputs[temp_key] = inputs[temp_key].to(device)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=50)
# 将模型生成的文本 token 序列 解码成可读的字符串形式
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(response)