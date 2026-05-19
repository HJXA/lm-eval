#!/bin/bash
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=1
export PATH="/ruilab/jxhe/miniconda3/envs/lmeval/bin:$PATH"

# 锁定使用 GPU
export CUDA_VISIBLE_DEVICES=0

# =========================
# 1. 配置区域
# =========================

TASKS_ARRAY=("ai2_arc,hellaswag,openbookqa,piqa,commonsense_qa,mmlu_continuation")
shots=0

# 直接填写多个模型路径
# 注意：路径应该指向包含 config.json / model.safetensors / tokenizer.json 的 HF 模型目录
# 不要指向 global_stepxxxxx 这种子目录
MODEL_PATHS=(
# "/ruilab2/hjxa/ms-swift/output/PT_LLaMA_0.5B/v8-20260511-151925/checkpoint-350000"
"/ruilab2/hjxa/checkpoints/qwen/Qwen3/0.6B/Qwen3-0.6B-Base"
# "/path/to/model_2"
# "/path/to/model_3"
)

LOG_ROOT="./lm-eval/eval_logs"
RESULT_ROOT="./lm-eval/results"

# vLLM 参数
TP_SIZE=1
GPU_MEM=0.4
DTYPE="auto"

# =========================
# 2. 任务字符串处理
# =========================

TASK_LIST=$(IFS=,; echo "${TASKS_ARRAY[*]}")
TASK_DIR_NAME=$(echo "$TASK_LIST" | tr ',' '_')

echo "================================================================"
echo "当前指定任务: $TASK_LIST"
echo "shots: $shots"
echo "================================================================"

# =========================
# 3. 遍历给定模型路径
# =========================

for model_path in "${MODEL_PATHS[@]}"; do

    # 去掉结尾斜杠
    model_path="${model_path%/}"

    if [ -z "$model_path" ]; then
        continue
    fi

    if [ ! -d "$model_path" ]; then
        echo "❌ 跳过：模型路径不存在: $model_path"
        continue
    fi

    if [ ! -f "$model_path/config.json" ]; then
        echo "❌ 跳过：该目录下没有 config.json，不是有效 HF 模型目录:"
        echo "   $model_path"
        continue
    fi

    # 直接以最后一级目录名命名
    model_name=$(basename "$model_path")
    save_model_name="${model_name}"

    TARGET_LOG_DIR="${LOG_ROOT}/${TASK_DIR_NAME}/${save_model_name}/shots_${shots}"
    RESULTS_DIR="${RESULT_ROOT}/${TASK_DIR_NAME}/${save_model_name}/shots_${shots}"

    mkdir -p "$TARGET_LOG_DIR"
    mkdir -p "$RESULTS_DIR"

    log_file="${TARGET_LOG_DIR}/${save_model_name}_${shots}.log"

    if [ -f "$log_file" ]; then
        echo "跳过模型: $save_model_name，日志已存在: $log_file"
        continue
    fi

    echo "==============================================================="
    echo "正在评测模型:"
    echo "  model_path: $model_path"
    echo "  model_name: $save_model_name"
    echo "  log_file:   $log_file"
    echo "  result_dir: $RESULTS_DIR"
    echo "==============================================================="

    lm_eval --model vllm \
        --model_args "pretrained=${model_path},tensor_parallel_size=${TP_SIZE},dtype=${DTYPE},gpu_memory_utilization=${GPU_MEM},enable_thinking=False" \
        --tasks "$TASK_LIST" \
        --device cuda:0 \
        --batch_size auto \
        --num_fewshot "$shots" \
        --log_samples \
        --output_path "$RESULTS_DIR" \
        2>&1 | tee "$log_file"

    echo "模型 $save_model_name 评测完成。"
    echo "---------------------------------------------------------------"

done

echo "🎉 所有给定模型路径已全部完成！"