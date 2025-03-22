#!/bin/bash

scripts=("llavatrain_sqa_finetune_trainjson_ck10000.sh" "llavatrain_sqa_finetune_trainjson_ck12000.sh"
        "llavatrain_sqa_finetune_trainjson_ck14000.sh" "llavatrain_sqa_finetune_trainjson_ck16000.sh" "llavatrain_sqa_finetune_trainjson_ck18000.sh"
        "llavatrain_sqa_finetune_trainjson_ck20000.sh" "llavatrain_sqa_finetune_trainjson_ck22000.sh" "llavatrain_sqa_finetune_trainjson_ck24000.sh"
        "llavatrain_sqa_finetune_trainjson_ck26000.sh" "llavatrain_sqa_finetune_trainjson_ck28000.sh" "llavatrain_sqa_finetune_trainjson_ck30000.sh"
        "llavatrain_sqa_finetune_trainjson_ck32000.sh" "llavatrain_sqa_finetune_trainjson_ck34000.sh" "llavatrain_sqa_finetune_trainjson_ck36000.sh"
        "llavatrain_sqa_finetune_trainjson_ck38000.sh" "llavatrain_sqa_finetune_trainjson_ck40000.sh" "llavatrain_sqa_finetune_trainjson_ck42000.sh"
        "llavatrain_sqa_finetune_trainjson_ck44000.sh" "llavatrain_sqa_finetune_trainjson_ck46000.sh" "llavatrain_sqa_finetune_trainjson_ck48000.sh"
        "llavatrain_sqa_finetune_trainjson_ck50000.sh" )

for script in "${scripts[@]}"; do
    echo "开始执行 $script..."
    
    # 执行脚本
    sh $script &> ../../terminal_log/${script%.sh}.log
    
    # 检查当前脚本是否执行成功
    if [ $? -eq 0 ]; then
        echo "$script 执行成功"
    else
        echo "$script 执行失败，停止执行后续脚本"
        exit 1  # 如果某个脚本执行失败，停止执行后续脚本
    fi
done

echo "所有脚本执行完毕"