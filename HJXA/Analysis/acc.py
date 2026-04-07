import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm


def log_message(message):
    tqdm.write(str(message), end="\n")


def safe_configure_torch_threads(num_threads=1, interop_threads=1):
    try:
        torch.set_num_threads(num_threads)
    except RuntimeError as e:
        log_message(f"[Torch Thread Config] skip set_num_threads({num_threads}): {e}")

    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError as e:
        log_message(f"[Torch Thread Config] skip set_num_interop_threads({interop_threads}): {e}")


safe_configure_torch_threads(1, 1)



# =========================
# 1. 全局配置
# =========================

RANKS = list(range(4))

EXCLUDED_TASKS = ["c4"]


# 全局变量：1 Step 对应多少 token（默认 1M）
TOKENS_PER_STEP = 1_000_000
# 统一换算到 Billion token（B）
TOKENS_PER_STEP_GLOBAL = TOKENS_PER_STEP / 1e9
READ_MAX_WORKERS = min(16, max(4, (os.cpu_count() or 8)))

# 平滑配置：Time-weighted EMA
ENABLE_TIME_WEIGHTED_EMA = True
TW_EMA_ALPHA = 0.1

# 任务分组 (来自第二段代码)
TASK_GROUPS = {
    "C4": {
        "title": "C4",
        "tasks": ["c4"]
    },
    "commonsense": {
        "title": "Commonsense Reasoning",
        "tasks": ["hellaswag", "piqa", "commonsense_qa"]
    },
    "science": {
        "title": "Scientific Reasoning",
        "tasks": ["arc_challenge", "arc_easy", "openbookqa"]
    },
    "mmlu": {
        "title": "MMLU (Multi-domain Knowledge)",
        "tasks": ["mmlu_continuation"]
    },
    "Ifeval": {
        "title": "Ifeval",
        "tasks": ["ifeval"]

    },
    "Math":{
        "title": "Math",
        "tasks": ["gsm8k", "minerva_math"]
    }
}

FILE_NAME_MAP = {
    "commonsense": "CommonSense.png",
    "science": "Science.png",
    "mmlu": "MMLU.png",
    "Ifeval": "Ifeval.png",
    "Math": "Math.png",
    "C4": "C4.png",
}

TASKS = {
    "arc_challenge": {
        "eval_base": "./lm-eval/results/ai2_arc,hellaswag,openbookqa,piqa,commonsense_qa,mmlu_continuation",
        "metric": "acc,none"
    },
    "arc_easy": {
        "eval_base": "./lm-eval/results/ai2_arc,hellaswag,openbookqa,piqa,commonsense_qa,mmlu_continuation",
        "metric": "acc,none"
    },
    "commonsense_qa": {
        "eval_base": "./lm-eval/results/ai2_arc,hellaswag,openbookqa,piqa,commonsense_qa,mmlu_continuation",
        "metric": "acc,none"
    },
    "hellaswag": {
        "eval_base": "./lm-eval/results/ai2_arc,hellaswag,openbookqa,piqa,commonsense_qa,mmlu_continuation",
        "metric": "acc,none"
    },
    "openbookqa": {
        "eval_base": "./lm-eval/results/ai2_arc,hellaswag,openbookqa,piqa,commonsense_qa,mmlu_continuation",
        "metric": "acc,none"
    },
    "piqa": {
        "eval_base": "./lm-eval/results/ai2_arc,hellaswag,openbookqa,piqa,commonsense_qa,mmlu_continuation",
        "metric": "acc,none"
    },
    "mmlu_continuation": {
        "eval_base": "./lm-eval/results/ai2_arc,hellaswag,openbookqa,piqa,commonsense_qa,mmlu_continuation",
        "metric": "acc,none"
    },
    "ifeval": {
        "eval_base": "./lm-eval/results/ifeval",
        "pt_eval_base": "./lm-eval/results/ifeval",
        "metric": "inst_level_strict_acc,none"
    },
    "gsm8k": {
        "eval_base": "./lm-eval/results/gsm8k,minerva_math",
        "pt_eval_base": "./lm-eval/results/gsm8k,minerva_math",
        "metric": "exact_match,flexible-extract"
    },
    "minerva_math": {
        "eval_base": "./lm-eval/results/gsm8k,minerva_math",
        "pt_eval_base": "./lm-eval/results/gsm8k,minerva_math",
        "metric": "math_verify,none"
    },
    "c4": {
        "eval_base": "./lm-eval/results/c4",
        "metric": "byte_perplexity,none",
    }
}



# =========================
# 2. 模型统一配置
# =========================

ASI_CACHE = {}

RUN_TYPE = "general"  # "general" / "math"

SAVE_DIR = f"./lm-eval/HJXA/Analysis/SFT_Analysis/Acc_Figures/{RUN_TYPE}"
os.makedirs(SAVE_DIR, exist_ok=True)

MODELS = {
    # ================= MiniMind =================
    # "MiniMind": {
    #     "root": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Train/LLaMA-Factory/coe_train_result/COE_PT_Main/My_PT/PT_MiniMind2_100M",
    #     "general_sft_folder": "pt_minimid_general_sft_cpt_models",
    #     "math_sft_folder": "pt_minimid_openmath_sft_cpt_models",
    #     "eval_pattern": r'tokens_([0-9.]+)B',
    #     "coe_pattern": r'(\d+)B',
    #     "pt_folder": "PT_MiniMind2_100M/little_sets",
    #     "pt_pattern": r'tokens_([0-9.]+)B',
    # },

    "PT_Pythia_14M": {
        "root": "/ruilab/jxhe/CoE_Monitor/ms-swift/coe_train_result/PT_Pythia_14M",
        "eval_pattern": r'checkpoint-([0-9.]+)',
        "coe_pattern": r'(\d+)B',
        "pt_folder": "PT_Pythia_14M/little_sets",
        "pt_pattern": r'checkpoint-([0-9.]+)',
    },
    
}


if RUN_TYPE == "general":
    TASKS.pop("gsm8k", None)
    TASKS.pop("minerva_math", None)
    TASK_GROUPS.pop("Math", None)
elif RUN_TYPE == "math":
    TASKS.pop("ifeval", None)
    TASK_GROUPS.pop("Ifeval", None)

# =========================
# 3. 数据提取统一函数
# =========================
def extract_task_score(json_path, task_name):
    """根据任务配置提取对应的指标分数"""
    with open(json_path, "r") as f:
        data = json.load(f)

    results = data.get("results", {})
    if task_name not in results:
        return None
        
    res = results[task_name]
    
    # 获取该任务预设的指标 Key
    target_metric = TASKS.get(task_name, {}).get("metric", "acc,none")
    
    # 尝试提取指定指标
    score = res.get(target_metric)
    
    # 备选逻辑：如果指定的指标不存在，尝试通用 fallback
    if score is None:
        log_message(f"{target_metric},指定的指标不存在")
        raise KeyError(f"Task '{task_name}' in {json_path} 缺失指标: {target_metric}")
                
    return score


def build_eval_root(task_cfg, model_cfg, is_sft):
    """构建单任务评测目录路径"""
    if not is_sft and "pt_eval_base" in task_cfg:
        base_path = task_cfg["pt_eval_base"]
    else:
        base_path = task_cfg["eval_base"]

    folder_name = model_cfg.get("sft_folder") if is_sft else model_cfg.get("pt_folder")
    if not folder_name:
        return None
    return os.path.join(base_path, folder_name, "shots_0")


def smooth_time_weighted_ema(x_values, y_values, alpha=TW_EMA_ALPHA):
    """按 x 间隔做 time-weighted EMA 平滑。"""
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)

    if len(x) <= 1 or len(y) <= 1:
        return y

    dx = np.diff(x)
    positive_dx = dx[dx > 0]
    base_dt = float(np.median(positive_dx)) if len(positive_dx) > 0 else 1.0
    if base_dt <= 0:
        base_dt = 1.0

    smoothed = np.empty_like(y)
    smoothed[0] = y[0]

    for i in range(1, len(y)):
        dt = max((x[i] - x[i - 1]) / base_dt, 1e-9)
        weight = 1.0 - np.exp(-alpha * dt)
        smoothed[i] = smoothed[i - 1] * (1.0 - weight) + y[i] * weight

    return smoothed


def maybe_smooth_curve(x_values, y_values):
    if not ENABLE_TIME_WEIGHTED_EMA:
        return np.asarray(y_values)
    return smooth_time_weighted_ema(x_values, y_values)

def _parse_folder_to_tokens_b(folder_name, pattern):
    """解析文件夹名得到 token(B)；若匹配到 checkpoint/step，则按“步数->token”换算。"""
    match = re.search(pattern, folder_name)
    if match:
        raw_val = float(match.group(1))
        lower_pattern = pattern.lower()
        if "checkpoint" in lower_pattern or "step" in lower_pattern:
            return raw_val * TOKENS_PER_STEP_GLOBAL
        return raw_val

    # 兜底：自动识别 checkpoint-<step>
    step_match = re.search(r'checkpoint-(\d+)', folder_name)
    if step_match:
        step_val = int(step_match.group(1))
        return step_val * TOKENS_PER_STEP_GLOBAL

    return None

def _read_single_checkpoint_task_score(args):
    full_path, folder, pattern, task_name, model_key, source_name = args

    token_val = _parse_folder_to_tokens_b(folder, pattern)
    if token_val is None:
        return None

    json_files = [f for f in os.listdir(full_path) if f.endswith(".json")]
    json_files.sort(key=lambda x: not x.startswith("results_"))

    for json_file in json_files:
        json_path = os.path.join(full_path, json_file)
        try:
            score = extract_task_score(json_path, task_name)
        except Exception as e:
            log_message(f"[Skip Dataset] model={model_key} task={task_name} source={source_name} checkpoint={folder} reason={e}")
            continue

        if score is not None:
            return token_val, score

    return None

def build_single_task_perf_data(eval_root, pattern, task_name, model_key="UnknownModel", source_name="Unknown", verbose=False):
    """去指定的 eval_root 下，翻找特定 task_name 的所有分数"""
    data_points = []
    if not os.path.exists(eval_root):
        return data_points

    checkpoint_args = []

    for folder in os.listdir(eval_root):
        full_path = os.path.join(eval_root, folder)
        if not os.path.isdir(full_path):
            continue

        checkpoint_args.append((full_path, folder, pattern, task_name, model_key, source_name))

    if not checkpoint_args:
        return data_points

    desc = f"Read {source_name}-{model_key}-{task_name}"
    with ThreadPoolExecutor(max_workers=READ_MAX_WORKERS) as exe:
        futures = [exe.submit(_read_single_checkpoint_task_score, args) for args in checkpoint_args]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, leave=False):
            res = fut.result()
            if res is not None:
                data_points.append(res)

    if verbose:
        found_tokens = {t for t, _ in data_points}
        for full_path, folder, pattern, _, _, _ in checkpoint_args:
            token_val = _parse_folder_to_tokens_b(folder, pattern)
            if token_val is not None and token_val not in found_tokens:
                log_message(f"\nCheckpoint: {folder} | Token: {token_val}B | ✗ Missing Task: {task_name}")

    data_points.sort(key=lambda x: x[0])
    return data_points

def collect_multi_task_perf(config, pattern, is_sft):
    """(保留自第一段代码) 用于计算所有任务的严格对齐平均分"""
    task_results = {}
    all_seen_tokens = set()
    included_tasks = [task_name for task_name in TASKS.keys() if task_name not in EXCLUDED_TASKS]
    
    for task_name in included_tasks:
        task_cfg = TASKS[task_name]
        eval_root = build_eval_root(task_cfg, config, is_sft)
        if not eval_root:
            continue
        data_points = build_single_task_perf_data(
            eval_root,
            pattern,
            task_name,
            model_key=config.get("model_key", "UnknownModel"),
            source_name="SFT" if is_sft else "PT"
        )
        
        if data_points:
            task_dict = {t: s for t, s in data_points}
            task_results[task_name] = task_dict
            all_seen_tokens.update(task_dict.keys())

    if not task_results:
        return np.array([]), np.array([])

    common_tokens = set.intersection(*(set(res.keys()) for res in task_results.values()))
    sorted_common = sorted(list(common_tokens))
    valid_tasks = list(task_results.keys())

    # 数据对齐报警机制
    dropped_tokens = all_seen_tokens - common_tokens
    if dropped_tokens:
        type_str = "SFT" if is_sft else "PT"
        log_message(f"\n[!] {config.get('sft_folder' if is_sft else 'pt_folder')} ({type_str}) 数据对齐报警:")
        for t in sorted(list(dropped_tokens)):
            missing_from = [name for name in valid_tasks if t not in task_results[name]]
            log_message(f"  - Token {t}B 被剔除: 缺失任务 {missing_from}")

    final_avg_scores = [np.mean([task_results[name][t] for name in valid_tasks]) for t in sorted_common]
    return np.array(sorted_common), np.array(final_avg_scores)


def collect_subset_task_perf(config, pattern, is_sft, task_subset):
    """计算指定任务子集的严格对齐平均分"""
    task_results = {}

    for task_name in task_subset:
        task_cfg = TASKS[task_name]

        eval_root = build_eval_root(task_cfg, config, is_sft)
        if not eval_root:
            continue
        data_points = build_single_task_perf_data(
            eval_root,
            pattern,
            task_name,
            model_key=config.get("model_key", "UnknownModel"),
            source_name="SFT" if is_sft else "PT"
        )

        if data_points:
            task_dict = {t: s for t, s in data_points}
            task_results[task_name] = task_dict

    if not task_results:
        return np.array([]), np.array([])

    common_tokens = set.intersection(*(set(res.keys()) for res in task_results.values()))
    sorted_common = sorted(list(common_tokens))
    valid_tasks = list(task_results.keys())

    final_avg_scores = [
        np.mean([task_results[name][t] for name in valid_tasks])
        for t in sorted_common
    ]

    return np.array(sorted_common), np.array(final_avg_scores)

def get_id_ood_tasks():
    all_tasks = list(TASKS.keys())

    if RUN_TYPE == "math":
        id_tasks = ["gsm8k", "minerva_math"]
    else:  # general
        id_tasks = ["ifeval"]


    ood_tasks = [t for t in all_tasks if t not in id_tasks and t not in EXCLUDED_TASKS]

    return id_tasks, ood_tasks

# =========================
# 4. ASI 计算
# =========================

def angle_func(a, b):
    dot = np.sum(a * b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    cos = np.clip(dot / (norm + 1e-9), -1.0, 1.0)
    return np.arccos(cos)

def compute_metrics_wrapper(file_paths):
    all_z_ang = []
    all_a_in = []
    all_a_mid = []
    all_a_out = []

    for file_path in file_paths:
        try:
            raw_tensor = torch.load(file_path, map_location='cpu', weights_only=False)
            tensor = raw_tensor.detach().to(torch.float32).numpy()
        except Exception:
            # print(f"Error loading {file_path}: {e}")
            continue

        if tensor.ndim == 2:
            tensor = np.expand_dims(tensor, axis=0)

        for h in tensor:
            L_total = len(h) - 1
            if L_total < 1:
                continue

            h0 = h[0]
            hL = h[-1]

            z_ang = angle_func(h0, hL)
            a_in = angle_func(h0, h[1])
            a_out = angle_func(h[-2], hL)

            if L_total > 2:
                mids = []
                for l in range(1, L_total - 1):
                    mids.append(angle_func(h[l], h[l + 1]))
                a_mid = np.mean(mids)
            else:
                a_mid = 0.0

            all_z_ang.append(z_ang)
            all_a_in.append(a_in)
            all_a_mid.append(a_mid)
            all_a_out.append(a_out)
    
    if not all_z_ang:
        return None

    return (
        np.mean(all_z_ang),
        np.mean(all_a_in),
        np.mean(all_a_mid),
        np.mean(all_a_out),
    )

def process_single_task(args):
    _, tokens_b, file_paths = args
    metrics = compute_metrics_wrapper(file_paths)
    if metrics is None:
        return None
    z_ang, a_in, a_mid, a_out = metrics
    
    # Calculate ASI_full (Including z_ang)
    asi_full = (z_ang + a_in + a_mid + a_out) / 4.0
    return tokens_b, asi_full

def get_asi_metrics(root_path, batch_size=128, plot_interval=100):
    if not root_path:
        return [], []
    
    cache_key = (root_path, batch_size, plot_interval)
    if cache_key in ASI_CACHE:
        return ASI_CACHE[cache_key]

    # Try locating Layer_Hidden_Train directly or via version folders
    target_dir = os.path.join(root_path, "Layer_Hidden_Train")
    if not os.path.exists(target_dir):
        # Look for v* folders
        subdirs = [os.path.join(root_path, d) for d in os.listdir(root_path) 
                   if os.path.isdir(os.path.join(root_path, d)) and d.startswith("v")]
        if subdirs:
            # Sort by modification time or name, assuming latest is best or handle multiple
            latest = sorted(subdirs)[-1]
            target_dir = os.path.join(latest, "Layer_Hidden_Train")
    
    if not os.path.exists(target_dir):
        # fallback attempting to find ANY Layer_Hidden_Train recursively depth=2
        found = False
        for r, ds, fs in os.walk(root_path):
            if "Layer_Hidden_Train" in ds:
                target_dir = os.path.join(r, "Layer_Hidden_Train")
                found = True
                break
        if not found:
            return [], []

    files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.pt')]
    if not files:
        return [], []

    step_map = {}
    for f in files:
        m = re.search(r'Step(\d+)_Rank\d+\.pt', os.path.basename(f))
        if m:
            s = int(m.group(1))
            if s not in step_map:
                step_map[s] = []
            step_map[s].append(f)
    
    if not step_map:
        return [], []

    tasks = []
    for s in sorted(step_map.keys()):
        tk = s * TOKENS_PER_STEP_GLOBAL
        tasks.append((s, tk, step_map[s]))
    
    final_results = []
    # Running in process pool
    with ProcessPoolExecutor(max_workers=8) as exe:
        futures = [exe.submit(process_single_task, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Read COE Metrics", leave=False):
            res = fut.result()
            if res:
                final_results.append(res)
    
    final_results.sort(key=lambda x: x[0])
    
    tokens = [x[0] for x in final_results]
    values = [x[1] for x in final_results] # ASI_full

    ASI_CACHE[cache_key] = (tokens, values)
    return tokens, values

# =========================
# 5. 绘图模块
# =========================

# 提取出一个全局变量，方便你随时调整缩放比例
# 例如：0.01 表示大于 stage1 的部分，每增加 10B 相当于前面 1000B (1TB) 的视觉宽度
COMPRESSION_RATIO = 0.05

def transform_x(x_seq, stage1_tokens, compression_ratio=COMPRESSION_RATIO):
    """
    将真实 Token 坐标映射到视觉坐标
    """
    if not stage1_tokens:
        return x_seq
    
    mapped_seq = []
    for x in x_seq:
        if x <= stage1_tokens:
            mapped_seq.append(x)
        else:
            # 超过部分除以压缩倍率，映射到图表上的物理位置
            mapped_seq.append(stage1_tokens + (x - stage1_tokens) / compression_ratio)
    return mapped_seq

def x_axis_formatter(x_prime, _pos, stage1_tokens, compression_ratio=COMPRESSION_RATIO):
    """
    将图表上的视觉坐标还原为真实的 Token 数量标签
    """
    if not stage1_tokens:
        return f"{x_prime:g}"
    
    if x_prime <= stage1_tokens:
        orig_x = x_prime
    else:
        # 视觉坐标乘以压缩倍率，完美还原回真实的 x 值
        orig_x = stage1_tokens + (x_prime - stage1_tokens) * compression_ratio
        
    return f"{orig_x:g}"


def draw_asi_plot_overall(model_key):
    """绘制全部任务平均得分"""
    config = MODELS[model_key]
    config["model_key"] = model_key
    root = config.get("root")
    stage1_tokens = config.get("stage1_tokens")

    try:
        sft_tokens, sft_avgs = collect_multi_task_perf(config, config["eval_pattern"], True)
    except Exception as e:
        log_message(f"[Skip Part] model={model_key} part=SFT reason={e}")
        sft_tokens, sft_avgs = np.array([]), np.array([])

    try:
        pt_tokens, pt_avgs = collect_multi_task_perf(config, config["pt_pattern"], False)
    except Exception as e:
        log_message(f"[Skip Part] model={model_key} part=PT reason={e}")
        pt_tokens, pt_avgs = np.array([]), np.array([])

    try:
        if root:
            asi_tokens, asi_values = get_asi_metrics(root)
        else:
            asi_tokens, asi_values = [], []
    except Exception as e:
        log_message(f"[Skip Part] model={model_key} part=COE reason={e}")
        asi_tokens, asi_values = [], []

    has_asi = len(asi_tokens) > 0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Tokens (B)")
    ax.grid(True, linestyle=":", alpha=0.6)

    if stage1_tokens:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, pos: x_axis_formatter(x, pos, stage1_tokens, COMPRESSION_RATIO)
        ))
        ax.axvline(x=stage1_tokens, color='black', linestyle=':', alpha=1, label="Stage 1 End")

    if has_asi:
        asi_tx = transform_x(asi_tokens, stage1_tokens)
        asi_values_sm = maybe_smooth_curve(asi_tokens, asi_values)
        ax.plot(asi_tx, asi_values_sm, marker='o', linewidth=1, markersize=1, label="ASI_full", color='blue')
        ax.set_ylabel("ASI_full", color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2 = ax.twinx()
    else:
        ax.set_ylabel("Average Score")
        ax2 = ax

    if len(sft_tokens) > 0:
        sft_tx = transform_x(sft_tokens, stage1_tokens)
        sft_avgs_sm = maybe_smooth_curve(sft_tokens, sft_avgs)
        ax2.plot(sft_tx, sft_avgs_sm, color='red', linestyle='--', marker='D', label="SFT Avg")

    if len(pt_tokens) > 0:
        pt_tx = transform_x(pt_tokens, stage1_tokens)
        pt_avgs_sm = maybe_smooth_curve(pt_tokens, pt_avgs)
        ax2.plot(pt_tx, pt_avgs_sm, color='green', linestyle='-.', marker='s', label="PT Avg")

    ax2.set_ylabel("Average Score (PT & SFT All Tasks)")

    lines = ax.get_lines()
    if has_asi:
        lines += ax2.get_lines()

    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)

    ax.set_title(f"{model_key}: ASI_full vs Overall SFT & PT Performance",
                 fontsize=14, fontweight='bold', pad=40)

    plt.tight_layout()

    model_save_dir = os.path.join(SAVE_DIR, model_key)
    os.makedirs(model_save_dir, exist_ok=True)
    save_path = os.path.join(model_save_dir, f"Overall_Average.png")

    plt.savefig(save_path, dpi=300)
    plt.close()

    log_message(f"[√] Saved Overall Average to: {save_path}")

def draw_asi_plot_group(model_key, group_key):

    config = MODELS[model_key]
    config["model_key"] = model_key
    root = config.get("root")
    group = TASK_GROUPS[group_key]
    group_tasks = group["tasks"]
    stage1_tokens = config.get("stage1_tokens")

    sft_data, pt_data = {}, {}

    for task in group_tasks:

        task_cfg = TASKS[task]

        sft_perf = []
        if config.get("sft_folder"):
            sft_base = task_cfg["eval_base"]
            sft_root = os.path.join(sft_base, config["sft_folder"], "shots_0")
            sft_perf = build_single_task_perf_data(
                sft_root,
                config["eval_pattern"],
                task,
                model_key=model_key,
                source_name="SFT"
            )

        if sft_perf:
            sft_data[task] = sft_perf

        pt_base = task_cfg.get("pt_eval_base", task_cfg["eval_base"])
        pt_root = os.path.join(pt_base, config["pt_folder"], "shots_0")

        pt_perf = build_single_task_perf_data(
            pt_root,
            config["pt_pattern"],
            task,
            model_key=model_key,
            source_name="PT"
        )

        if pt_perf:
            pt_data[task] = pt_perf

    try:
        if root:
            asi_tokens, asi_values = get_asi_metrics(root)
        else:
            asi_tokens, asi_values = [], []
    except Exception as e:
        log_message(f"[Skip Part] model={model_key} part=COE reason={e}")
        asi_tokens, asi_values = [], []

    has_asi = len(asi_tokens) > 0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Tokens (B)")
    ax.grid(True, linestyle=":", alpha=0.6)

    if stage1_tokens:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, pos: x_axis_formatter(x, pos, stage1_tokens, COMPRESSION_RATIO)
        ))
        ax.axvline(x=stage1_tokens, color='black', linestyle=':', alpha=1)

    if has_asi:
        asi_tx = transform_x(asi_tokens, stage1_tokens)
        asi_values_sm = maybe_smooth_curve(asi_tokens, asi_values)
        ax.plot(asi_tx, asi_values_sm, marker='o', linewidth=1, markersize=1, label="ASI_full", color='blue')
        ax.set_ylabel("ASI_full", color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2 = ax.twinx()
    else:
        ax2 = ax

    fixed_colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]

    task_colors = {task: fixed_colors[i % len(fixed_colors)] for i, task in enumerate(group_tasks)}

    for task in group_tasks:

        color = task_colors[task]

        if task in sft_data:

            tokens, scores = zip(*sft_data[task])
            tx = transform_x(tokens, stage1_tokens)
            scores_sm = maybe_smooth_curve(tokens, scores)

            ax2.plot(tx, scores_sm,
                     linestyle='--',
                     marker='D',
                     color=color,
                     linewidth=1,
                     label=f"SFT-{task}")

        if task in pt_data:

            tokens, scores = zip(*pt_data[task])
            tx = transform_x(tokens, stage1_tokens)
            scores_sm = maybe_smooth_curve(tokens, scores)

            ax2.plot(tx, scores_sm,
                     linestyle='-.',
                     marker='s',
                     color=color,
                     alpha=0.4,
                     linewidth=1,
                     label=f"PT-{task}")

    ax2.set_ylabel("Accuracy (PT & SFT)")

    lines = ax.get_lines()

    if has_asi:
        lines += ax2.get_lines()

    labels = [l.get_label() for l in lines]

    ax.legend(lines, labels,
              loc='lower center',
              bbox_to_anchor=(0.5, 1.02),
              ncol=3,
              frameon=False,
              fontsize=9)

    ax.set_title(f"{model_key}: ASI_full vs {group['title']}",
                 fontsize=14,
                 fontweight='bold',
                 pad=60)

    plt.tight_layout()

    model_save_dir = os.path.join(SAVE_DIR, model_key)

    os.makedirs(model_save_dir, exist_ok=True)

    save_path = os.path.join(model_save_dir, FILE_NAME_MAP[group_key])

    plt.savefig(save_path, dpi=300)

    plt.close()

    log_message(f"[√] Saved Group '{group_key}' to: {save_path}")

def draw_asi_plot_id_ood(model_key, subset_name, task_subset):

    config = MODELS[model_key]
    config["model_key"] = model_key
    root = config.get("root")
    stage1_tokens = config.get("stage1_tokens")

    try:
        sft_tokens, sft_avgs = collect_subset_task_perf(
            config, config["eval_pattern"], True, task_subset
        )
    except Exception as e:
        log_message(f"[Skip Part] model={model_key} part=SFT reason={e}")
        sft_tokens, sft_avgs = np.array([]), np.array([])

    try:
        pt_tokens, pt_avgs = collect_subset_task_perf(
            config, config["pt_pattern"], False, task_subset
        )
    except Exception as e:
        log_message(f"[Skip Part] model={model_key} part=PT reason={e}")
        pt_tokens, pt_avgs = np.array([]), np.array([])

    try:
        if root:
            asi_tokens, asi_values = get_asi_metrics(root)
        else:
            asi_tokens, asi_values = [], []
    except Exception as e:
        log_message(f"[Skip Part] model={model_key} part=COE reason={e}")
        asi_tokens, asi_values = [], []

    has_asi = len(asi_tokens) > 0

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_xlabel("Tokens (B)")
    ax.grid(True, linestyle=":", alpha=0.6)

    if stage1_tokens:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, pos: x_axis_formatter(x, pos, stage1_tokens, COMPRESSION_RATIO)
        ))

        ax.axvline(x=stage1_tokens, color='black', linestyle=':', alpha=1)

    if has_asi:
        asi_tx = transform_x(asi_tokens, stage1_tokens)
        asi_values_sm = maybe_smooth_curve(asi_tokens, asi_values)
        ax.plot(asi_tx, asi_values_sm, marker='o', linewidth=1, markersize=1, label="ASI_full", color='blue')
        ax.set_ylabel("ASI_full", color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2 = ax.twinx()
    else:
        ax2 = ax

    if len(sft_tokens) > 0:

        sft_tx = transform_x(sft_tokens, stage1_tokens)
        sft_avgs_sm = maybe_smooth_curve(sft_tokens, sft_avgs)

        ax2.plot(sft_tx,
                 sft_avgs_sm,
                 color='red',
                 linestyle='--',
                 marker='D',
                 label=f"SFT {subset_name} Avg")

    if len(pt_tokens) > 0:

        pt_tx = transform_x(pt_tokens, stage1_tokens)
        pt_avgs_sm = maybe_smooth_curve(pt_tokens, pt_avgs)

        ax2.plot(pt_tx,
                 pt_avgs_sm,
                 color='green',
                 linestyle='-.',
                 marker='s',
                 label=f"PT {subset_name} Avg")

    ax2.set_ylabel(f"{subset_name} Average Score")

    lines = ax.get_lines()

    if has_asi:
        lines += ax2.get_lines()

    labels = [l.get_label() for l in lines]

    ax.legend(lines, labels,
              loc='lower center',
              bbox_to_anchor=(0.5, 1.02),
              ncol=3,
              frameon=False)

    ax.set_title(f"{model_key}: ASI_full vs {subset_name} Performance",
                 fontsize=14,
                 fontweight='bold',
                 pad=40)

    plt.tight_layout()

    model_save_dir = os.path.join(SAVE_DIR, model_key)
    os.makedirs(model_save_dir, exist_ok=True)
    save_path = os.path.join(model_save_dir, f"{subset_name}_Average.png")

    plt.savefig(save_path, dpi=300)

    plt.close()

    log_message(f"[√] Saved {subset_name} Average to: {save_path}")

# =========================
# 6. 执行
# =========================

if __name__ == "__main__":
    for model in MODELS.keys():
        log_message(f"\n================ Processing {model} ================")
        # 1. 生成整体大盘平均分图表 (继承第一段代码功能)
        draw_asi_plot_overall(model)

        id_tasks, ood_tasks = get_id_ood_tasks()

        draw_asi_plot_id_ood(model, "ID", id_tasks)
        draw_asi_plot_id_ood(model, "OOD", ood_tasks)
        
        # 2. 生成各子领域明细图表 (继承第二段代码功能)
        for group_key in TASK_GROUPS.keys():
            draw_asi_plot_group(model, group_key)

    log_message("\n🎉 全部图表生成完成！")