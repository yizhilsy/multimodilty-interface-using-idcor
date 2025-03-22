#!/bin/bash

scripts=("llavatrain_sqa_finetune1.sh" "llavatrain_sqa_finetune2.sh" "llavatrain_sqa_finetune3.sh" "llavatrain_sqa_finetune4.sh" "llavatrain_sqa_finetune5.sh")

for script in "${scripts[@]}"; do
    echo "开始执行 $script..."
    
    # 执行脚本
    sh $script &> terminal_log/$script.log
    
    # 检查当前脚本是否执行成功
    if [ $? -eq 0 ]; then
        echo "$script 执行成功"
    else
        echo "$script 执行失败，停止执行后续脚本"
        exit 1  # 如果某个脚本执行失败，停止执行后续脚本
    fi
done

echo "所有脚本执行完毕"