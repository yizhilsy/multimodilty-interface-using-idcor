## 训练任务

### stage1

* model: `Qwen2.5-3B-Instruc + clipvL14` freeze vision_tower and language model and train mlp on dataset: `LLaVA-CC3M-Pretrain-595K` for 1 epoch, learing rate 2e-3, batch size 128
