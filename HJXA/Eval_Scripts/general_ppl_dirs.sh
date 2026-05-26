#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=1
export PATH="/ruilab/jxhe/miniconda3/envs/lmeval/bin:$PATH"

# 锁定使用 GPU
export CUDA_VISIBLE_DEVICES=4

# --- 1. 配置区域 ---
#
# 任务列表
TASKS_ARRAY=("ai2_arc,hellaswag,openbookqa,piqa,commonsense_qa,mmlu_continuation")
shots=0

# ✅ 可以写多个 base_dir
BASE_DIRS=(
# "/data/jxhe/LLM/checkpoints/OLMo_checkpoints/little_sets"  # PT 要带 little_sets，INS 不要
# "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Train/LLaMA-Factory/model/olmo_general_sft_cpt_models"
# "/ruilab/jxhe/CoE_Monitor/checkpoints/pt_models/PT_Pythia_14M/little_sets"
"/ruilab2/hjxa/ms-swift/output/SFT/llama-0.5B-350B-general"
# "/ruilab2/hjxa/checkpoints/pt/llama-0.5B-350B/little_sets"


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
        USE_CHAT_TEMPLATE=false
    fi

    TARGET_LOG_DIR="${LOG_ROOT}/${TASK_DIR_NAME}/${FULL_BASE_NAME}/shots_${shots}"
    RESULTS_DIR="${RESULT_ROOT}/${TASK_DIR_NAME}/${FULL_BASE_NAME}/shots_${shots}"

    mkdir -p "$TARGET_LOG_DIR"
    mkdir -p "$RESULTS_DIR"

    echo "==============================================================="
    echo "当前 BASE_DIR: $BASE_DIR"
    echo "日志目录: $TARGET_LOG_DIR"
    echo "==============================================================="

    # --- 4. 自动寻找模型目录 ---
    # 找到所有包含 config.json 的模型目录
    mapfile -t ALL_PATHS < <(
      find "$BASE_DIR" -type f -name "config.json" \
        | sed 's#/config.json##' \
        | sort -V
    )

    if [ "${#ALL_PATHS[@]}" -eq 0 ]; then
        echo "警告: 在 $BASE_DIR 下没有找到包含 config.json 的模型目录"
        continue
    fi

    # 按版本目录分组，每组只保留最后一个 checkpoint（步数最大）
    # MODEL_PATHS=()
    # prev_version_dir=""
    # last_in_group=""
    # for mp in "${ALL_PATHS[@]}"; do
    #     version_dir=$(dirname "$mp")
    #     if [ "$version_dir" != "$prev_version_dir" ] && [ -n "$prev_version_dir" ]; then
    #         MODEL_PATHS+=("$last_in_group")
    #     fi
    #     last_in_group="$mp"
    #     prev_version_dir="$version_dir"
    # done
    # if [ -n "$last_in_group" ]; then
    #     MODEL_PATHS+=("$last_in_group")
    # fi

    MODEL_PATHS=("${ALL_PATHS[@]}")

    # --- 5. 遍历模型 ---
    for model_path in "${MODEL_PATHS[@]}"; do

        rel_path="${model_path#$BASE_DIR/}"

        # 把路径里的 / 替换成 __，避免日志文件名冲突
        model_name=$(echo "$rel_path" | sed 's#/#__#g')

        log_file="${TARGET_LOG_DIR}/${model_name}_${shots}.log"

        if [ -f "$log_file" ]; then
            echo "跳过模型: $model_name (日志已存在)"
            continue
        fi

        echo "正在评测模型: $model_name"

        EXTRA_ARGS=""
        if [ "$USE_CHAT_TEMPLATE" = true ]; then
            EXTRA_ARGS="--apply_chat_template"
        fi

        lm_eval --model vllm \
            --model_args "pretrained=${model_path},tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.4,enable_thinking=False" \
            --tasks "$TASK_LIST" \
            --device cuda:0 \
            --batch_size auto \
            --num_fewshot "$shots" \
            --log_samples \
            --output_path "$RESULTS_DIR" \
            $EXTRA_ARGS \
            2>&1 | tee "$log_file"


        echo "模型 $model_name 评测完成。"
        echo "---------------------------------------------------------------"
    done

done

echo "🎉 所有 BASE_DIR 已全部完成！"