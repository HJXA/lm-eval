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
        "tasks": ["gsm8k"]
    }
}

filename_map = {"commonsense": "CommonSense.png", "science": "Science.png", "mmlu": "MMLU.png","Ifeval":"Ifeval.png","Math":"Math.png"}

TASKS = {
    "arc_challenge": {
        "eval_base": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/results/ai2_arc,hellaswag,openbookqa,piqa,mmlu,commonsense_qa",
        "metric": "acc,none"
    },
    "arc_easy": {
        "eval_base": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/results/ai2_arc,hellaswag,openbookqa,piqa,mmlu,commonsense_qa",
        "metric": "acc,none"
    },
    "commonsense_qa": {
        "eval_base": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/results/ai2_arc,hellaswag,openbookqa,piqa,mmlu,commonsense_qa",
        "metric": "acc,none"
    },
    "hellaswag": {
        "eval_base": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/results/ai2_arc,hellaswag,openbookqa,piqa,mmlu,commonsense_qa",
        "metric": "acc,none"
    },
    "openbookqa": {
        "eval_base": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/results/ai2_arc,hellaswag,openbookqa,piqa,mmlu,commonsense_qa",
        "metric": "acc,none"
    },
    "piqa": {
        "eval_base": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/results/ai2_arc,hellaswag,openbookqa,piqa,mmlu,commonsense_qa",
        "metric": "acc,none"
    },
    "mmlu_continuation": {
        "eval_base": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/results/mmlu_continuation",
        "metric": "acc,none"
    },
    "ifeval": {
        "eval_base": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/results/ifeval",
        "metric": "inst_level_strict_acc,none"
    },
    "gsm8k": {
        "eval_base": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/results/gsm8k",
        "metric": "exact_match,flexible-extract"

    }
}

SAVE_DIR = "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Eval/lm-evaluation-harness/My/Analysis/SFT_Analysis/Acc_Figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# 2. 模型统一配置
# =========================

F4_MINIMIND_CACHE = {} # 新增缓存优化

MODELS = {
    # ================= MiniMind =================
    "MiniMind": {
        "root": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Train/LLaMA-Factory/coe_train_result/COE_PT_Main/My_PT/PT_MiniMind2_100M",
        "eval_folder": "pt_minimid_general_sft_cpt_models",
        "eval_pattern": r'tokens_([0-9.]+)B',
        "coe_pattern": r'(\d+)B',
        "special_coe": True,
        "pt_folder": "PT_MiniMind2_100M/little_sets",
        "pt_pattern": r'tokens_([0-9.]+)B',
    },

    # ================= Olmo 系列 =================
    "Olmo": {
        "root": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Train/LLaMA-Factory/coe_train_result/olmo_pt_cpt_coe_train_results",
        "eval_folder": "olmo_general_sft_cpt_models",
        "eval_pattern": r'tokens(\d+)B',
        "coe_pattern": r'(\d+)B',
        "pt_folder": "OLMo_checkpoints/little_sets",
        "pt_pattern": r"tokens([\d\.]+)B"
    },

    "Olmo2": {
        "root": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Train/LLaMA-Factory/coe_train_result/olmo2_pt_cpt_coe_train_results",
        "eval_folder": "olmo2_general_sft_cpt_models",
        "eval_pattern": r'tokens(\d+)B',
        "coe_pattern": r'(\d+)B',
        "pt_folder": "OLMo2_checkpoints/little_sets",
        "pt_pattern": r"tokens([\d\.]+)B"
    },

    # ================= Pythia 系列 =================
    "Pythia-14M": {
        "root": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Train/LLaMA-Factory/coe_train_result/pythia_14m_deduped_pt_cpt_coe_train_results",
        "eval_folder": "pythia_14m_general_sft_cpt_models",
        "eval_pattern": r'_(\d+)Btokens',
        "coe_pattern": r'(\d+)B',
        "pt_folder": "pythia_14m/little_sets",
        "pt_pattern": r'_(\d+)Btokens'
    },

    "Pythia-160M": {
        "root": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Train/LLaMA-Factory/coe_train_result/pythia_160m_pt_cpt_coe_train_results",
        "eval_folder": "pythia_160m_general_sft_cpt_models",
        "eval_pattern": r'_(\d+)Btokens',
        "coe_pattern": r'(\d+)B',
        "pt_folder": "pythia_160m/little_sets",
        "pt_pattern": r'_(\d+)Btokens'
    },

    "Pythia-1B": {
        "root": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Train/LLaMA-Factory/coe_train_result/pythia_1B_pt_cpt_coe_train_results",
        "eval_folder": "pythia_1B_general_sft_cpt_models",
        "eval_pattern": r'_(\d+)Btokens',
        "coe_pattern": r'(\d+)B',
        "pt_folder": "pythia_1B/little_sets",
        "pt_pattern": r'_(\d+)Btokens'
    },

    "Pythia-2.8B": {
        "root": "/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/Train/LLaMA-Factory/coe_train_result/pythia_2_8B_pt_cpt_coe_train_results",
        "eval_folder": "pythia_2_8B_general_sft_cpt_models",
        "eval_pattern": r'_(\d+)Btokens',
        "coe_pattern": r'(\d+)B',
        "pt_folder": "pythia_2.8B/little_sets",
        "pt_pattern": r'_(\d+)Btokens'
    },
}

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
        eval_root = os.path.join(
            task_cfg["eval_base"], 
            config["eval_folder"] if is_sft else config["pt_folder"], 
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
        print(f"\n[!] {config.get('eval_folder' if is_sft else 'pt_folder')} ({type_str}) 数据对齐报警:")
        for t in sorted(list(dropped_tokens)):
            missing_from = [name for name in TASKS.keys() if name not in task_results or t not in task_results[name]]
            print(f"  - Token {t}B 被剔除: 缺失任务 {missing_from}")

    final_avg_scores = [np.mean([task_results[name][t] for name in TASKS.keys()]) for t in sorted_common]
    return np.array(sorted_common), np.array(final_avg_scores)

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

def draw_f4_plot_overall(model_key):
    """功能1：绘制全部任务大盘的平均得分 (基于第一段代码)"""
    config = MODELS[model_key]
    root = config["root"]
    
    sft_tokens, sft_avgs = collect_multi_task_perf(config, config["eval_pattern"], True)
    pt_tokens, pt_avgs = collect_multi_task_perf(config, config["pt_pattern"], False)

    if config.get("special_coe", False):
        f4_tokens, f4_values = get_f4_minimind_step(root)
    else:
        f4_tokens, f4_values = get_f4_metrics(root, config["coe_pattern"])

    if not f4_tokens:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(f4_tokens, f4_values, marker='o', linewidth=1, markersize=3, label="Ang", color='blue')
    ax.set_xlabel("Tokens (B)")
    ax.set_ylabel("Ang", color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax.twinx()
    
    if len(sft_tokens) > 0:
        ax2.plot(sft_tokens, sft_avgs, color='red', linestyle='--', marker='D', label="SFT Avg")
        for x, y in zip(sft_tokens, sft_avgs):
            ax2.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=5, color='red')

    if len(pt_tokens) > 0:
        ax2.plot(pt_tokens, pt_avgs, color='green', linestyle='-.', marker='s', label="PT Avg")
        for x, y in zip(pt_tokens, pt_avgs):
            ax2.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0, -14), ha='center', fontsize=5, color='green')

    ax2.set_ylabel("Average Score (PT & SFT All Tasks)", color='black')

    lines = ax.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=10)
    ax.set_title(f"{model_key}: Ang vs Overall SFT & PT Performance", fontsize=14, fontweight='bold', pad=40)

    plt.tight_layout()
    model_save_dir = os.path.join(SAVE_DIR)
    os.makedirs(model_save_dir, exist_ok=True)
    save_path = os.path.join(model_save_dir, f"{model_key}_Overall_Average.png")
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[√] Saved Overall Average to: {save_path}")

def draw_f4_plot_group(model_key, group_key):
    """功能2：绘制特定任务组的明细得分对比 (基于第二段代码)"""
    config = MODELS[model_key]
    root = config["root"]
    group = TASK_GROUPS[group_key]
    tasks = group["tasks"]

    sft_data, pt_data = {}, {}

    for task in tasks:
        task_base = TASKS[task]["eval_base"]
        # 获取 SFT
        sft_root = os.path.join(task_base, config["eval_folder"], "shots_0")
        sft_perf = build_single_task_perf_data(sft_root, config["eval_pattern"], task, True)
        if sft_perf: sft_data[task] = sft_perf
        
        # 获取 PT
        pt_root = os.path.join(task_base, config["pt_folder"], "shots_0")
        pt_perf = build_single_task_perf_data(pt_root, config["pt_pattern"], task, False)
        if pt_perf: pt_data[task] = pt_perf

    if config.get("special_coe", False):
        f4_tokens, f4_values = get_f4_minimind_step(root)
    else:
        f4_tokens, f4_values = get_f4_metrics(root, config["coe_pattern"])

    if not f4_tokens:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(f4_tokens, f4_values, marker='o', linewidth=1, markersize=3, label="Ang", color='blue')
    ax.set_xlabel("Tokens (B)")
    ax.set_ylabel("Ang", color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax.twinx()
    fixed_colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2"]
    task_colors = {task: fixed_colors[i % len(fixed_colors)] for i, task in enumerate(tasks)}

    for task in tasks:
        color = task_colors[task]

        if task in sft_data:
            tokens, scores = zip(*sft_data[task])
            ax2.plot(tokens, scores, linestyle='--', marker='D', color=color, markersize=3, linewidth=1, label=f"SFT-{task}")
            for x, y in zip(tokens, scores):
                ax2.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=5, color=color)

        if task in pt_data:
            tokens, scores = zip(*pt_data[task])
            ax2.plot(tokens, scores, linestyle='-.', marker='s', color=color, alpha=0.4, markersize=3, linewidth=1, label=f"PT-{task}")
            for x, y in zip(tokens, scores):
                ax2.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0, -14), ha='center', fontsize=5, color=color, alpha=0.4)

    ax2.set_ylabel("Accuracy (PT & SFT)", color='black')

    lines = ax.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=9)
    ax.set_title(f"{model_key}: Ang vs {group['title']}", fontsize=14, fontweight='bold', pad=60)

    plt.tight_layout()

    model_save_dir = os.path.join(SAVE_DIR, model_key)
    os.makedirs(model_save_dir, exist_ok=True)
    save_path = os.path.join(model_save_dir, filename_map[group_key])

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[√] Saved Group '{group_key}' to: {save_path}")

# =========================
# 6. 执行
# =========================

if __name__ == "__main__":
    for model in MODELS.keys():
        print(f"\n================ Processing {model} ================")
        # 1. 生成整体大盘平均分图表 (继承第一段代码功能)
        draw_f4_plot_overall(model)
        
        # 2. 生成各子领域明细图表 (继承第二段代码功能)
        for group_key in TASK_GROUPS.keys():
            draw_f4_plot_group(model, group_key)

    print("\n🎉 全部图表生成完成！")