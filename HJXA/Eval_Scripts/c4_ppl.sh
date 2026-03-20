#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=1
echo $HF_ENDPOINT

# 锁定使用 GPU
export CUDA_VISIBLE_DEVICES=4

# --- 1. 配置区域 ---
# 
# 任务列表
TASKS_ARRAY=("c4")
shots=0

# ✅ 可以写多个 base_dir
BASE_DIRS=(
# "/data/jxhe/LLM/checkpoints/OLMo_checkpoints/little_sets"  # PT 要带 little_sets，INS 不要
# "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Train/LLaMA-Factory/model/olmo_general_sft_cpt_models"
"/ruilab/jxhe/CoE_Monitor/checkpoints/pt_models/PT_HJXA_Llama_104M_Minimind_no_packing_no_padding_free/little_sets"
# "/ruilab/jxhe/CoE_Monitor/checkpoints/pt_models/PT_HJXA_Llama_104M_Minimind/little_sets"
# "/ruilab/jxhe/CoE_Monitor/checkpoints/pt_models/PT_HJXA_Llama_5M_no_packing_no_padding_free/little_sets"
# "/ruilab/jxhe/CoE_Monitor/checkpoints/pt_models/PT_HJXA_Llama_5M/little_sets"
# "/ruilab/jxhe/CoE_Monitor/checkpoints/pt_models/padding_free_bug/PT_HJXA_Llama_104M_Minimind_bug/little_sets"
# "/ruilab/jxhe/CoE_Monitor/checkpoints/pt_models/padding_free_bug/PT_HJXA_Llama_55M_bug/little_sets"
# "/ruilab/jxhe/CoE_Monitor/checkpoints/pt_models/padding_free_bug/PT_HJXA_Llama_5M_bug/little_sets"
)

LOG_ROOT="./lm-eval/eval_logs"
RESULT_ROOT="./lm-eval/results"

# --- 2. 任务字符串处理 ---
TASK_LIST=$(IFS=,; echo "${TASKS_ARRAY[*]}")
TASK_DIR_NAME=$(IFS=_; echo "${TASKS_ARRAY[*]}")

echo "================================================================"
echo "当前指定任务: $TASK_LIST"
echo "================================================================"

# --- 3. 遍历多个 BASE_DIR ---
for BASE_DIR in "${BASE_DIRS[@]}"; do

    if [ -z "$BASE_DIR" ]; then
        continue
    fi

    BASE_NAME=$(basename "$BASE_DIR")

    # 判断是否以 little_sets 结尾
    if [[ "$BASE_DIR" == */little_sets ]]; then
        # 如果是，取两级：父目录/little_sets
        PARENT_NAME=$(basename "$(dirname "$BASE_DIR")")
        FULL_BASE_NAME="${PARENT_NAME}/${BASE_NAME}"
        USE_CHAT_TEMPLATE=false
    else
        # 如果不是，只取最后结尾一级
        FULL_BASE_NAME="${BASE_NAME}"
        USE_CHAT_TEMPLATE=true
    fi

    TARGET_LOG_DIR="${LOG_ROOT}/${TASK_DIR_NAME}/${FULL_BASE_NAME}/shots_${shots}"
    RESULTS_DIR="${RESULT_ROOT}/${TASK_DIR_NAME}/${FULL_BASE_NAME}/shots_${shots}"

    mkdir -p "$TARGET_LOG_DIR"
    mkdir -p "$RESULTS_DIR"

    echo "==============================================================="
    echo "当前 BASE_DIR: $BASE_DIR"
    echo "日志目录: $TARGET_LOG_DIR"
    echo "==============================================================="

    # --- 4. 遍历当前 BASE_DIR 下的所有模型 ---
    for model_path in "$BASE_DIR"/*/; do
        model_name=$(basename "${model_path%/}")
        log_file="${TARGET_LOG_DIR}/${model_name}_${shots}.log"

        if [ -f "$log_file" ]; then
            echo "跳过模型: $model_name (日志已存在)"
            continue
        fi

        echo "正在评测模型: $model_name"


        lm_eval --model vllm \
            --model_args "pretrained=${model_path},tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.5,enable_thinking=False" \
            --tasks "$TASK_LIST" \
            --device cuda:0 \
            --batch_size auto \
            --num_fewshot "$shots" \
            --log_samples \
            --output_path "$RESULTS_DIR" \
            2>&1 | tee "$log_file"


        echo "模型 $model_name 评测完成。"
        echo "---------------------------------------------------------------"
    done

done

echo "🎉 所有 BASE_DIR 已全部完成！"