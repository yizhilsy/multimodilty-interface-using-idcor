# --nnodes 1 --nproc_per_node 4 --master_port 25641

deepspeed --include localhost:0,1 run_show.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path ./qwen2.5_3B_Instruct_clipvL14_model/model001 \
    --train_type freeze_vision_and_language \
    --data_path /d/lsy/shared_data/liuhaotian/LLaVA-CC3M-Pretrain-595K \
    --remove_unused_columns false \
    --bf16 true \
    --fp16 false \
    --dataloader_pin_memory True \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers True \
    --output_dir ./result_model/stage1/[v2.CC3M-Pretrain-595K]qwen2.5_3B_Instruct_clipvL14 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --evaluation_strategy "steps" \
    --eval_steps 5000 \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 8 \
    --report_to "tensorboard" \
    --learning_rate 2e-3 \
    --logging_steps 8 \
    --logging_dir "./train_log/[v2.CC3M-Pretrain-595K]qwen2.5_3B_Instruct_clipvL14"
# --model_max_length 2048

# --save_strategy "steps" \
# --save_steps 10 \
# --save_steps 1000 \
# --save_strategy "epoch" \
