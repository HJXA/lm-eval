import os
import re
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# =========================
# 1. 全局配置
# =========================

RANKS = list(range(4))

# 任务分组 (来自第二段代码)
TASK_GROUPS = {
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

filename_map = {"commonsense": "CommonSense.png", "science": "Science.png", "mmlu": "MMLU.png","Ifeval":"Ifeval.png","Math":"Math.png"}

TASKS = {
    "arc_challenge": {
        "eval_base": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/results/ai2_arc,hellaswag,openbookqa,piqa,commonsense_qa,mmlu_continuation",
        "metric": "acc,none"
    },
    "arc_easy": {
        "eval_base": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/results/ai2_arc,hellaswag,openbookqa,piqa,commonsense_qa,mmlu_continuation",
        "metric": "acc,none"
    },
    "commonsense_qa": {
        "eval_base": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/results/ai2_arc,hellaswag,openbookqa,piqa,commonsense_qa,mmlu_continuation",
        "metric": "acc,none"
    },
    "hellaswag": {
        "eval_base": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/results/ai2_arc,hellaswag,openbookqa,piqa,commonsense_qa,mmlu_continuation",
        "metric": "acc,none"
    },
    "openbookqa": {
        "eval_base": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/results/ai2_arc,hellaswag,openbookqa,piqa,commonsense_qa,mmlu_continuation",
        "metric": "acc,none"
    },
    "piqa": {
        "eval_base": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/results/ai2_arc,hellaswag,openbookqa,piqa,commonsense_qa,mmlu_continuation",
        "metric": "acc,none"
    },
    "mmlu_continuation": {
        "eval_base": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/results/ai2_arc,hellaswag,openbookqa,piqa,commonsense_qa,mmlu_continuation",
        "metric": "acc,none"
    },
    "ifeval": {
        "eval_base": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/results/ifeval",
        "pt_eval_base": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/results/ifeval",
        "metric": "inst_level_strict_acc,none"
    },
    "gsm8k": {
        "eval_base": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/results/gsm8k,minerva_math",
        "pt_eval_base": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/results/gsm8k,minerva_math",
        "metric": "exact_match,flexible-extract"
    },
    "minerva_math": {
        "eval_base": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/results/gsm8k,minerva_math",
        "pt_eval_base": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/results/gsm8k,minerva_math",
        "metric": "math_verify,none"
    }
}



# =========================
# 2. 模型统一配置
# =========================

F4_MINIMIND_CACHE = {} # 新增缓存优化

Type = "general" # "general" / "math"

SAVE_DIR = f"/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/My/Analysis/SFT_Analysis/Acc_Figures/{Type}"
os.makedirs(SAVE_DIR, exist_ok=True)

MODELS = {
    # ================= MiniMind =================
    # "MiniMind": {
    #     "root": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Train/LLaMA-Factory/coe_train_result/COE_PT_Main/My_PT/PT_MiniMind2_100M",
    #     "general_sft_folder": "pt_minimid_general_sft_cpt_models",
    #     "math_sft_folder": "pt_minimid_openmath_sft_cpt_models",
    #     "eval_pattern": r'tokens_([0-9.]+)B',
    #     "coe_pattern": r'(\d+)B',
    #     "special_coe": True,
    #     "pt_folder": "PT_MiniMind2_100M/little_sets",
    #     "pt_pattern": r'tokens_([0-9.]+)B',
    # },

    # # ================= Olmo 系列 =================
    # "Olmo": {
    #     "root": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Train/LLaMA-Factory/coe_train_result/olmo_pt_cpt_coe_train_results",
    #     "general_sft_folder": "olmo_general_sft_cpt_models",
    #     "math_sft_folder": "olmo_openmath_sft_cpt_models",
    #     "eval_pattern": r'tokens(\d+)B',
    #     "coe_pattern": r'(\d+)B',
    #     "pt_folder": "OLMo_checkpoints/little_sets",
    #     "pt_pattern": r"tokens([\d\.]+)B"
    # },

    # "Olmo2": {
    #     "root": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Train/LLaMA-Factory/coe_train_result/olmo2_pt_cpt_coe_train_results",
    #     "general_sft_folder": "olmo2_general_sft_cpt_models",
    #     "math_sft_folder": "olmo2_openmath_sft_cpt_models",
    #     "eval_pattern": r'(\d+)B',
    #     "coe_pattern": r'(\d+)B',
    #     "pt_folder": "OLMo2_checkpoints/little_sets",
    #     "pt_pattern": r"([\d\.]+)B",
    #     "stage1_tokens": 3985,
    # },
    # "Olmo3": {
    #     "root": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Train/LLaMA-Factory/coe_train_result/layer_hidden/olmo3_pt_cpt_coe_train_results",
    #     "general_sft_folder": "olmo3_general_sft_cpt_models",
    #     "math_sft_folder": "olmo3_openmath_sft_cpt_models",
    #     "eval_pattern": r'tokens(\d+)B',
    #     "coe_pattern": r'(\d+)B',
    #     "pt_folder": "OLMo3_checkpoints/little_sets",
    #     "pt_pattern": r"tokens([\d\.]+)B",
    #     "stage1_tokens": 5930,
    # },

    # # ================= Pythia 系列 =================
    # "Pythia-14M": {
    #     "root": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Train/LLaMA-Factory/coe_train_result/pythia_14m_deduped_pt_cpt_coe_train_results",
    #     "general_sft_folder": "pythia_14m_general_sft_cpt_models",
    #     "math_sft_folder": "pythia_14m_openmath_sft_cpt_models",
    #     "eval_pattern": r'_(\d+)Btokens',
    #     "coe_pattern": r'(\d+)B',
    #     "pt_folder": "pythia_14m/little_sets",
    #     "pt_pattern": r'_(\d+)Btokens'
    # },

    # "Pythia-160M": {
    #     "root": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Train/LLaMA-Factory/coe_train_result/pythia_160m_pt_cpt_coe_train_results",
    #     "general_sft_folder": "pythia_160m_general_sft_cpt_models",
    #     "math_sft_folder": "pythia_160m_openmath_sft_cpt_models",
    #     "eval_pattern": r'_(\d+)Btokens',
    #     "coe_pattern": r'(\d+)B',
    #     "pt_folder": "pythia_160m/little_sets",
    #     "pt_pattern": r'_(\d+)Btokens'
    # },

    # "Pythia-1B": {
    #     "root": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Train/LLaMA-Factory/coe_train_result/pythia_1B_pt_cpt_coe_train_results",
    #     "general_sft_folder": "pythia_1B_general_sft_cpt_models",
    #     "math_sft_folder": "pythia_1B_openmath_sft_cpt_models",
    #     "eval_pattern": r'_(\d+)Btokens',
    #     "coe_pattern": r'(\d+)B',
    #     "pt_folder": "pythia_1B/little_sets",
    #     "pt_pattern": r'_(\d+)Btokens'
    # },

    # "Pythia-2.8B": {
    #     "root": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Train/LLaMA-Factory/coe_train_result/pythia_2_8B_pt_cpt_coe_train_results",
    #     "general_sft_folder": "pythia_2_8B_general_sft_cpt_models",
    #     "math_sft_folder": "pythia_2_8B_openmath_sft_cpt_models",
    #     "eval_pattern": r'_(\d+)Btokens',
    #     "coe_pattern": r'(\d+)B',
    #     "pt_folder": "pythia_2.8B/little_sets",
    #     "pt_pattern": r'_(\d+)Btokens'
    # },
    "Llama-14M": {
        "root": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Train/LLaMA-Factory/coe_train_result/llama_14m_pt_cpt_coe_train_results",
        "general_sft_folder": "LLama_14M_general_sft",
        "math_sft_folder": "LLama_14M_openmath_sft",
        "eval_pattern": r'(\d+)B',
        "coe_pattern": r'(\d+)B',
        "pt_folder": "LLama_14M",
        "pt_pattern": r'(\d+)B'
    },
}

for model_name, model_cfg in MODELS.items():
    if Type == "math":
        model_cfg["sft_folder"] = model_cfg["math_sft_folder"]
    else:
        model_cfg["sft_folder"] = model_cfg["general_sft_folder"]

if Type == "general":
    TASKS.pop("gsm8k", None)
    TASKS.pop("minerva_math", None)
    TASK_GROUPS.pop("Math", None)
elif Type == "math":
    TASKS.pop("ifeval", None)
    TASK_GROUPS.pop("Ifeval", None)

# =========================
# 3. 数据提取统一函数
# =========================
def extract_task_score(json_path, task_name, is_sft):
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
        print(f"{target_metric},指定的指标不存在")
        raise 0
                
    return score

def build_single_task_perf_data(eval_root, pattern, task_name, is_sft, verbose=False):
    """去指定的 eval_root 下，翻找特定 task_name 的所有分数"""
    data_points = []
    if not os.path.exists(eval_root):
        return data_points

    for folder in os.listdir(eval_root):
        full_path = os.path.join(eval_root, folder)
        if not os.path.isdir(full_path):
            continue

        match = re.search(pattern, folder)
        if not match:
            continue

        token_val = float(match.group(1))

        json_files = [f for f in os.listdir(full_path) if f.endswith(".json")]
        json_files.sort(key=lambda x: not x.startswith("results_"))

        found = False
        for file in json_files:
            json_path = os.path.join(full_path, file)
            score = extract_task_score(json_path, task_name, is_sft)
            if score is not None:
                data_points.append((token_val, score))
                found = True
                break  

        if not found and verbose:
            print(f"\nCheckpoint: {folder} | Token: {token_val}B | ✗ Missing Task: {task_name}")

    data_points.sort(key=lambda x: x[0])
    return data_points

def collect_multi_task_perf(config, pattern, is_sft):
    """(保留自第一段代码) 用于计算所有任务的严格对齐平均分"""
    task_results = {}
    all_seen_tokens = set()
    
    for task_name, task_cfg in TASKS.items():
        # ✅ 仅 ifeval / gsm8k 特判 PT 的 eval_base
        if not is_sft and "pt_eval_base" in task_cfg:
            base_path = task_cfg["pt_eval_base"]
        else:
            base_path = task_cfg["eval_base"]

        eval_root = os.path.join(
            base_path,
            config["sft_folder"] if is_sft else config["pt_folder"],
            "shots_0"
        )

        data_points = build_single_task_perf_data(eval_root, pattern, task_name, is_sft)
        
        if data_points:
            task_dict = {t: s for t, s in data_points}
            task_results[task_name] = task_dict
            all_seen_tokens.update(task_dict.keys())

    if not task_results:
        return np.array([]), np.array([])

    common_tokens = set.intersection(*(set(res.keys()) for res in task_results.values()))
    sorted_common = sorted(list(common_tokens))

    # 数据对齐报警机制
    dropped_tokens = all_seen_tokens - common_tokens
    if dropped_tokens:
        type_str = "SFT" if is_sft else "PT"
        print(f"\n[!] {config.get('sft_folder' if is_sft else 'pt_folder')} ({type_str}) 数据对齐报警:")
        for t in sorted(list(dropped_tokens)):
            missing_from = [name for name in TASKS.keys() if name not in task_results or t not in task_results[name]]
            print(f"  - Token {t}B 被剔除: 缺失任务 {missing_from}")

    final_avg_scores = [np.mean([task_results[name][t] for name in TASKS.keys()]) for t in sorted_common]
    return np.array(sorted_common), np.array(final_avg_scores)


def collect_subset_task_perf(config, pattern, is_sft, task_subset):
    """计算指定任务子集的严格对齐平均分"""
    task_results = {}
    all_seen_tokens = set()

    for task_name in task_subset:
        task_cfg = TASKS[task_name]

        if not is_sft and "pt_eval_base" in task_cfg:
            base_path = task_cfg["pt_eval_base"]
        else:
            base_path = task_cfg["eval_base"]

        eval_root = os.path.join(
            base_path,
            config["sft_folder"] if is_sft else config["pt_folder"],
            "shots_0"
        )

        data_points = build_single_task_perf_data(eval_root, pattern, task_name, is_sft)

        if data_points:
            task_dict = {t: s for t, s in data_points}
            task_results[task_name] = task_dict
            all_seen_tokens.update(task_dict.keys())

    if not task_results:
        return np.array([]), np.array([])

    common_tokens = set.intersection(*(set(res.keys()) for res in task_results.values()))
    sorted_common = sorted(list(common_tokens))

    final_avg_scores = [
        np.mean([task_results[name][t] for name in task_subset])
        for t in sorted_common
    ]

    return np.array(sorted_common), np.array(final_avg_scores)

def get_id_ood_tasks():
    all_tasks = list(TASKS.keys())

    if Type == "math":
        id_tasks = ["gsm8k", "minerva_math"]
    else:  # general
        id_tasks = ["ifeval"]

    ood_tasks = [t for t in all_tasks if t not in id_tasks]

    return id_tasks, ood_tasks

# =========================
# 4. F4 计算
# =========================

def get_f4_minimind_step(root_path, batch_size=80, plot_interval=100):
    cache_key = (root_path, batch_size, plot_interval)

    if cache_key in F4_MINIMIND_CACHE:
        # 缓存命中，避免重复读取计算（适配第二段代码优化）
        return F4_MINIMIND_CACHE[cache_key]
    
    base_dir = os.path.join(root_path, "coe_select_results", "step_checkpoint")
    pat = re.compile(r"Step(\d+)_Rank\d+_coe\.pkl")

    if not os.path.exists(base_dir):
        return [], []

    files = os.listdir(base_dir)
    steps = [int(pat.match(f).group(1)) for f in files if pat.match(f)]

    if not steps:
        return [], []

    max_step = max(steps)
    num_bins = int(np.ceil(max_step / plot_interval))
    TOKENS_PER_STEP = 2048 * len(RANKS) * batch_size
    bins = [{"ang": []} for _ in range(num_bins)]

    for rank in RANKS:
        path = os.path.join(base_dir, f"Step{max_step}_Rank{rank}_coe.pkl")
        if not os.path.exists(path):
            continue

        with open(path, "rb") as f:
            data = pickle.load(f)

        ys = np.array([float(d[1]) for d in data])
        n = (len(ys)//batch_size)*batch_size
        ys = ys[:n].reshape(-1, batch_size).mean(axis=1)
        splits = np.array_split(ys, num_bins)

        for i, s in enumerate(splits):
            bins[i]["ang"].extend(s)

    token_labels, ang_means = [], []
    for i, b in enumerate(bins):
        step = min((i+1)*plot_interval, max_step)
        token = step * TOKENS_PER_STEP / 1e9
        token_labels.append(token)
        ang_means.append(np.mean(b["ang"]) if b["ang"] else np.nan)

    F4_MINIMIND_CACHE[cache_key] = (token_labels, ang_means)
    return token_labels, ang_means

def get_f4_metrics(root_path, folder_pattern):
    data_points = []
    if not os.path.exists(root_path):
        return [], []
    for folder in os.listdir(root_path):
        full = os.path.join(root_path, folder)
        if not os.path.isdir(full):
            continue

        match = re.search(folder_pattern, folder)
        if not match:
            continue

        token_val = float(match.group(1))
        target_dir = os.path.join(full, "coe_select_results")
        ang_values = []

        for r in RANKS:
            p = os.path.join(target_dir, f"Rank{r}_coe.pkl")
            if not os.path.exists(p):
                continue
            with open(p, "rb") as f:
                res = pickle.load(f)
                for item in res:
                    ang_values.append(float(item[1]))

        if ang_values:
            data_points.append((token_val, np.mean(ang_values)))

    data_points.sort(key=lambda x: x[0])
    return zip(*data_points) if data_points else ([], [])

# =========================
# 5. 绘图模块
# =========================


# =========================
# 5. 绘图模块 (修正版)
# =========================
import matplotlib.ticker as ticker
import os
import matplotlib.pyplot as plt

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

def x_axis_formatter(x_prime, pos, stage1_tokens, compression_ratio=COMPRESSION_RATIO):
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


def draw_f4_plot_overall(model_key):
    """绘制全部任务平均得分"""
    config = MODELS[model_key]
    root = config["root"]
    stage1_tokens = config.get("stage1_tokens")

    sft_tokens, sft_avgs = collect_multi_task_perf(config, config["eval_pattern"], True)
    pt_tokens, pt_avgs = collect_multi_task_perf(config, config["pt_pattern"], False)

    if config.get("special_coe", False):
        f4_tokens, f4_values = get_f4_minimind_step(root)
    else:
        f4_tokens, f4_values = get_f4_metrics(root, config["coe_pattern"])

    has_ang = len(f4_tokens) > 0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Tokens (B)")
    ax.grid(True, linestyle=":", alpha=0.6)

    if stage1_tokens:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, pos: x_axis_formatter(x, pos, stage1_tokens, COMPRESSION_RATIO)
        ))
        ax.axvline(x=stage1_tokens, color='black', linestyle=':', alpha=1, label="Stage 1 End")

    if has_ang:
        f4_tx = transform_x(f4_tokens, stage1_tokens)
        ax.plot(f4_tx, f4_values, marker='o', linewidth=1, markersize=3, label="Ang", color='blue')
        ax.set_ylabel("Ang", color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2 = ax.twinx()
    else:
        ax.set_ylabel("Average Score")
        ax2 = ax

    if len(sft_tokens) > 0:
        sft_tx = transform_x(sft_tokens, stage1_tokens)
        ax2.plot(sft_tx, sft_avgs, color='red', linestyle='--', marker='D', label="SFT Avg")

    if len(pt_tokens) > 0:
        pt_tx = transform_x(pt_tokens, stage1_tokens)
        ax2.plot(pt_tx, pt_avgs, color='green', linestyle='-.', marker='s', label="PT Avg")

    ax2.set_ylabel("Average Score (PT & SFT All Tasks)")

    lines = ax.get_lines()
    if has_ang:
        lines += ax2.get_lines()

    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)

    ax.set_title(f"{model_key}: Ang vs Overall SFT & PT Performance",
                 fontsize=14, fontweight='bold', pad=40)

    plt.tight_layout()

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"{model_key}_Overall_Average.png")

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[√] Saved Overall Average to: {save_path}")

def draw_f4_plot_group(model_key, group_key):

    config = MODELS[model_key]
    root = config["root"]
    group = TASK_GROUPS[group_key]
    tasks = group["tasks"]
    stage1_tokens = config.get("stage1_tokens")

    sft_data, pt_data = {}, {}

    for task in tasks:

        task_cfg = TASKS[task]

        sft_base = task_cfg["eval_base"]
        sft_root = os.path.join(sft_base, config["sft_folder"], "shots_0")
        sft_perf = build_single_task_perf_data(sft_root, config["eval_pattern"], task, True)

        if sft_perf:
            sft_data[task] = sft_perf

        pt_base = task_cfg.get("pt_eval_base", task_cfg["eval_base"])
        pt_root = os.path.join(pt_base, config["pt_folder"], "shots_0")

        pt_perf = build_single_task_perf_data(pt_root, config["pt_pattern"], task, False)

        if pt_perf:
            pt_data[task] = pt_perf

    if config.get("special_coe", False):
        f4_tokens, f4_values = get_f4_minimind_step(root)
    else:
        f4_tokens, f4_values = get_f4_metrics(root, config["coe_pattern"])

    has_ang = len(f4_tokens) > 0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Tokens (B)")
    ax.grid(True, linestyle=":", alpha=0.6)

    if stage1_tokens:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, pos: x_axis_formatter(x, pos, stage1_tokens, COMPRESSION_RATIO)
        ))
        ax.axvline(x=stage1_tokens, color='black', linestyle=':', alpha=1)

    if has_ang:
        f4_tx = transform_x(f4_tokens, stage1_tokens)
        ax.plot(f4_tx, f4_values, marker='o', linewidth=1, markersize=3, label="Ang", color='blue')
        ax.set_ylabel("Ang", color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2 = ax.twinx()
    else:
        ax2 = ax

    fixed_colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]

    task_colors = {task: fixed_colors[i % len(fixed_colors)] for i, task in enumerate(tasks)}

    for task in tasks:

        color = task_colors[task]

        if task in sft_data:

            tokens, scores = zip(*sft_data[task])
            tx = transform_x(tokens, stage1_tokens)

            ax2.plot(tx, scores,
                     linestyle='--',
                     marker='D',
                     color=color,
                     linewidth=1,
                     label=f"SFT-{task}")

        if task in pt_data:

            tokens, scores = zip(*pt_data[task])
            tx = transform_x(tokens, stage1_tokens)

            ax2.plot(tx, scores,
                     linestyle='-.',
                     marker='s',
                     color=color,
                     alpha=0.4,
                     linewidth=1,
                     label=f"PT-{task}")

    ax2.set_ylabel("Accuracy (PT & SFT)")

    lines = ax.get_lines()

    if has_ang:
        lines += ax2.get_lines()

    labels = [l.get_label() for l in lines]

    ax.legend(lines, labels,
              loc='lower center',
              bbox_to_anchor=(0.5, 1.02),
              ncol=3,
              frameon=False,
              fontsize=9)

    ax.set_title(f"{model_key}: Ang vs {group['title']}",
                 fontsize=14,
                 fontweight='bold',
                 pad=60)

    plt.tight_layout()

    model_save_dir = os.path.join(SAVE_DIR, model_key)

    os.makedirs(model_save_dir, exist_ok=True)

    save_path = os.path.join(model_save_dir, filename_map[group_key])

    plt.savefig(save_path, dpi=300)

    plt.close()

    print(f"[√] Saved Group '{group_key}' to: {save_path}")

def draw_f4_plot_id_ood(model_key, subset_name, task_subset):

    config = MODELS[model_key]
    root = config["root"]
    stage1_tokens = config.get("stage1_tokens")

    sft_tokens, sft_avgs = collect_subset_task_perf(
        config, config["eval_pattern"], True, task_subset
    )

    pt_tokens, pt_avgs = collect_subset_task_perf(
        config, config["pt_pattern"], False, task_subset
    )

    if config.get("special_coe", False):
        f4_tokens, f4_values = get_f4_minimind_step(root)
    else:
        f4_tokens, f4_values = get_f4_metrics(root, config["coe_pattern"])

    has_ang = len(f4_tokens) > 0

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_xlabel("Tokens (B)")
    ax.grid(True, linestyle=":", alpha=0.6)

    if stage1_tokens:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, pos: x_axis_formatter(x, pos, stage1_tokens, COMPRESSION_RATIO)
        ))

        ax.axvline(x=stage1_tokens, color='black', linestyle=':', alpha=1)

    if has_ang:

        f4_tx = transform_x(f4_tokens, stage1_tokens)

        ax.plot(f4_tx, f4_values,
                marker='o',
                linewidth=1,
                markersize=3,
                label="Ang",
                color='blue')

        ax.set_ylabel("Ang", color='blue')
        ax.tick_params(axis='y', labelcolor='blue')

        ax2 = ax.twinx()

    else:

        ax2 = ax

    if len(sft_tokens) > 0:

        sft_tx = transform_x(sft_tokens, stage1_tokens)

        ax2.plot(sft_tx,
                 sft_avgs,
                 color='red',
                 linestyle='--',
                 marker='D',
                 label=f"SFT {subset_name} Avg")

    if len(pt_tokens) > 0:

        pt_tx = transform_x(pt_tokens, stage1_tokens)

        ax2.plot(pt_tx,
                 pt_avgs,
                 color='green',
                 linestyle='-.',
                 marker='s',
                 label=f"PT {subset_name} Avg")

    ax2.set_ylabel(f"{subset_name} Average Score")

    lines = ax.get_lines()

    if has_ang:
        lines += ax2.get_lines()

    labels = [l.get_label() for l in lines]

    ax.legend(lines, labels,
              loc='lower center',
              bbox_to_anchor=(0.5, 1.02),
              ncol=3,
              frameon=False)

    ax.set_title(f"{model_key}: Ang vs {subset_name} Performance",
                 fontsize=14,
                 fontweight='bold',
                 pad=40)

    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, f"{model_key}_{subset_name}_Average.png")

    plt.savefig(save_path, dpi=300)

    plt.close()

    print(f"[√] Saved {subset_name} Average to: {save_path}")

# =========================
# 6. 执行
# =========================

if __name__ == "__main__":
    for model in MODELS.keys():
        print(f"\n================ Processing {model} ================")
        # 1. 生成整体大盘平均分图表 (继承第一段代码功能)
        draw_f4_plot_overall(model)

        id_tasks, ood_tasks = get_id_ood_tasks()

        draw_f4_plot_id_ood(model, "ID", id_tasks)
        draw_f4_plot_id_ood(model, "OOD", ood_tasks)
        
        # 2. 生成各子领域明细图表 (继承第二段代码功能)
        for group_key in TASK_GROUPS.keys():
            draw_f4_plot_group(model, group_key)

    print("\n🎉 全部图表生成完成！")