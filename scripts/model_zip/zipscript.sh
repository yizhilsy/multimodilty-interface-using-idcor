#!/bin/bash

model_folder_list=("checkpoint-2000 checkpoint-4000 checkpoint-6000 checkpoint-8000 checkpoint-10000"
                   "checkpoint-12000 checkpoint-14000 checkpoint-16000 checkpoint-18000 checkpoint-20000")

for model_folders in "${model_folder_list[@]}"; do
    echo "开始压缩 $model_folders..."

    first_checkpoint=$(echo "$model_folders" | awk '{print $1}')
    # 执行压缩命令
    zip -r $model_folders &> ./${first_checkpoint}.log
    # 检查zip命令是否成功
    if [ $? -eq 0 ]; then
        echo "成功压缩 $model_folders"
    else
        echo "压缩 $model_folders 失败"
        exit 1 # 如果某次压缩失败，停止执行后续压缩
    fi
done

echo "所有压缩任务完成"