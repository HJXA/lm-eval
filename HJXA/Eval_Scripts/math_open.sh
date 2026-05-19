#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

# 锁定使用 GPU
export CUDA_VISIBLE_DEVICES=3

# --- 1. 配置区域 ---

TASKS_ARRAY=("gsm8k,minerva_math")
shots=0

# ✅ 可以写多个 base_dir
BASE_DIRS=(
  # PT little_sets，例如里面是 v*/checkpoint-*
#   "/ruilab2/hjxa/checkpoints/pt/llama-0.5B-350B/little_sets"

  # SFT 分级目录，例如 checkpoint-200000/v0-xxx/checkpoint-1000
  "/ruilab2/hjxa/ms-swift/output/SFT/llama-0.5B-350B-math-full"
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

  # 判断是否以 little_sets 结尾：PT 模型，不用 chat template
  if [[ "$BASE_DIR" == */little_sets ]]; then
    PARENT_NAME=$(basename "$(dirname "$BASE_DIR")")
    FULL_BASE_NAME="${PARENT_NAME}/${BASE_NAME}"
    USE_CHAT_TEMPLATE=false
  else
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
  echo "结果目录: $RESULTS_DIR"
  echo "是否使用 chat template: $USE_CHAT_TEMPLATE"
  echo "==============================================================="

  # --- 4. 自动寻找模型目录 ---
  # 判定标准：目录下有 config.json
  mapfile -t MODEL_PATHS < <(
    find "$BASE_DIR" -type f -name "config.json" \
      | sed 's#/config.json##' \
      | sort
  )

  if [ "${#MODEL_PATHS[@]}" -eq 0 ]; then
    echo "警告: 在 $BASE_DIR 下没有找到包含 config.json 的模型目录"
    continue
  fi

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

    echo "---------------------------------------------------------------"
    echo "正在评测模型: $model_name"
    echo "模型路径: $model_path"
    echo "日志文件: $log_file"

    EXTRA_ARGS=""
    if [ "$USE_CHAT_TEMPLATE" = true ]; then
      EXTRA_ARGS="--apply_chat_template"
    fi

    lm_eval --model vllm \
      --model_args "pretrained=${model_path},tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.25,enable_thinking=False" \
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