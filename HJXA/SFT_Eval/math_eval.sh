#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
# 锁定使用 GPU
export CUDA_VISIBLE_DEVICES=2

export HF_DATASETS_TRUST_REMOTE_CODE=1
# --- 1. 配置区域 ---
# 在这里指定你的任务列表，用空格隔开
TASKS_ARRAY=("gsm8k,mmlu") # ,minerva_math,mmlu,commonsense_qa") # 数学，常识，数学，代码，逻辑 minerva_math
shots=2



BASE_DIR= # "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Train/LLaMA-Factory/model/COE_PT_Main/Qwen3_0_6B_Base"

LOG_ROOT= # "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/eval_logs"

BASE_NAME=$(basename "${BASE_DIR}")



# --- 2. 逻辑处理 (自动转换) ---
# 将数组转换为逗号分隔的字符串，用于 lm_eval 命令
TASK_LIST=$(IFS=,; echo "${TASKS_ARRAY[*]}")
# 将数组转换为下划线分隔的字符串，用于创建文件夹名
TASK_DIR_NAME=$(IFS=_; echo "${TASKS_ARRAY[*]}")

TARGET_LOG_DIR="${LOG_ROOT}/${TASK_DIR_NAME}/${BASE_NAME}/shots_${shots}"
RESULTS_DIR="./results/${TASK_DIR_NAME}/${BASE_NAME}/shots_${shots}"

mkdir -p "$TARGET_LOG_DIR"
mkdir -p "$RESULTS_DIR"

echo "================================================================"
echo "当前指定任务: $TASK_LIST"
echo "日志保存目录: $TARGET_LOG_DIR"
echo "================================================================"

# --- 3. 循环处理模型 ---
for model_path in "$BASE_DIR"/*/; do
    # 获取文件夹名
    model_name=$(basename "${model_path%/}")
    log_file="${TARGET_LOG_DIR}/${model_name}_shots_${shots}.log"
    
    # 检查该任务组合下的日志是否已存在
    if [ -f "$log_file" ]; then
        echo "跳过模型: $model_name (任务组合的日志已存在)"
        continue
    fi

    echo "正在评测模型: $model_name"
    
    # 执行评测
    # 如果你想用 vLLM 加速，把 --model hf 改为 vllm
    #         --limit 1500 \
    lm_eval --model vllm \
        --model_args "pretrained=${model_path},tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,enable_thinking=False" \
        --tasks "$TASK_LIST" \
        --device cuda:0 \
        --batch_size auto \
        --num_fewshot "$shots" \
        --log_samples \
        --output_path "$RESULTS_DIR" \
        --apply_chat_template \
        2>&1 | tee "$log_file"

    echo "模型 $model_name 评测完成。"
    echo "----------------------------------------------------------------"
done

echo "所有任务检查/评测已完成！"