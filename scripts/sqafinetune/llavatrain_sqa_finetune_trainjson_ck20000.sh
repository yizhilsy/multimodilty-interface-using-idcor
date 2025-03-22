# --nnodes 1 --nproc_per_node 4 --master_port 25641

deepspeed --include localhost:0,1 ../../finetune_show.py \
    --deepspeed ../../ds_zero2_no_offload.json \
    --model_name_or_path ../../result_model/stage1/[v3.CC3M-Pretrain-595K]qwen2.5_3B_Instruct_clipvL14/checkpoint-20000 \
    --model_max_q_length 768 \
    --model_max_a_length 512 \
    --train_type lora_finetune \
    --data_path /d/lsy/shared_data/liuhaotian/LLaVA-Finetune/ScienceQA/llava_train_QCM-LEA.json \
    --image_folder /d/lsy/shared_data/liuhaotian/LLaVA-Finetune/ScienceQA/images/train \
    --eval_data_path /d/lsy/shared_data/liuhaotian/LLaVA-Finetune/ScienceQA/llava_val_QCM-LEA.json \
    --eval_image_folder /d/lsy/shared_data/liuhaotian/LLaVA-Finetune/ScienceQA/images/val \
    --remove_unused_columns false \
    --bf16 true \
    --fp16 false \
    --dataloader_pin_memory True \
    --dataloader_num_workers 6 \
    --dataloader_persistent_workers True \
    --output_dir ../../result_model/stage2/scienceqa/[v2.finetune_sqa]/trainjson_ck20000_qwen2.5_3B_Instruct_clipvL14 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --report_to "tensorboard" \
    --learning_rate 2e-4 \
    --logging_steps 8 \
    --logging_dir "../../train_log/stage2/[v2.finetune_sqa]trainjson_ck20000_qwen2.5_3B_Instruct_clipvL14"
# --model_max_length 2048

# --save_strategy "steps" \
# --save_steps 10 \
# --save_steps 1000 \
# --save_strategy "epoch" \
