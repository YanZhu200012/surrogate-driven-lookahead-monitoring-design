"""
evaluate_unified.py
对 run_100.py 生成的多样本结果进行统一评估（性能 + 空间 + 成本）。

输出文件
--------
性能相关：
  fig1_boxplots.png                 终态指标箱线图（K_R² / C_R² / Uncertainty）
  fig2_learning_curves.png          学习曲线（mean ± 1σ 阴影）
  fig2_learning_curves_phases.png   带三阶段分隔线的学习曲线
  fig3_scatter_kr2_cr2.png          K_R² vs C_R² 散点图
  summary.txt                       4 方法均值±标准差 + 相对降幅
  phase_analysis.txt                early/mid/late 三阶段斜率分析
  statistical_tests.txt             AL vs (Greedy / LHS / Random) Wilcoxon + Cohen's d

空间分布：
  spatial_stats.csv                 每样本每方法 6 项空间指标原始数据
  spatial_summary.txt               4 方法空间指标汇总 + 三组成对统计检验

成本-性能权衡：
  cost_performance.txt              well-saving analysis、ROI 估算、应用场景边界

直接运行：
  python evaluate_unified.py

"""

import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import wilcoxon

# ==========================================
# 全局绘图样式（与 evaluate_all.py 保持一致）
# ==========================================
JOURNAL_RC = {
    'font.family':        'Arial',
    'font.size':          12,
    'axes.titlesize':     13,
    'axes.labelsize':     12,
    'xtick.labelsize':    10,
    'ytick.labelsize':    11,
    'legend.fontsize':    11,
    'figure.titlesize':   14,
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
}
SAVE_KW = dict(dpi=300, bbox_inches='tight')

METHOD_COLORS = {
    'active':  '#d62728',   # 红色 — 主动选井
    'greedy':  '#1f77b4',   # 蓝色 — 贪心 MaxVar
    'lhs':  '#2ca02c',   # 绿色 — LHS 空间填充（10 次 trial 取均值）
    'random':  '#7f7f7f',   # 灰色 — 随机
}
METHOD_LABELS = {
    'active':  'Look-ahead',
    'greedy':  'Greedy',
    'lhs':  'LHS',
    'random':  'Random',
}
# 顺序决定箱线图、学习曲线图例的展示顺序
METHODS = ('active', 'greedy', 'lhs', 'random')

# 多 trial 方法集合：这些方法的 npz 使用 'all_trial_histories' 结构，
# 加载/统计时需取多次 trial 的均值。集中定义避免多处硬编码遗漏。
MULTI_TRIAL_METHODS = {'random', 'lhs'}

# ─── 空间统计相关常量（对应 evaluate_spatial_stats 模块） ───────────────────
GRID_SIZE      = 51
N_INIT_WELLS   = 3
BOUNDARY_WIDTH = 8
TIME_WEIGHTS   = [0.1, 0.15, 0.2, 0.25, 0.3]
N_TIME_STEPS   = 5

SPATIAL_METRIC_KEYS = ['ann_distance', 'clark_evans', 'coverage_rate',
                       'boundary_rate', 'plume_inside', 'centroid_dist']

SPATIAL_METRIC_LABELS = {
    'ann_distance':  'ANN Distance (grid)',
    'clark_evans':   'Clark-Evans R',
    'coverage_rate': 'Coverage Rate',
    'boundary_rate': 'Boundary Rate',
    'plume_inside':  'Plume Inside Rate',
    'centroid_dist': 'Centroid Offset',
}


def apply_journal_style():
    plt.rcParams.update(JOURNAL_RC)


# ==========================================
# 工具函数
# ==========================================
def _as_dict(entry):
    """将 numpy void / 0-d object array / 普通 dict 统一转为 dict。"""
    if hasattr(entry, 'item'):
        entry = entry.item()
    if isinstance(entry, dict):
        return entry
    # numpy void (structured array element)
    return {k: entry[k] for k in entry.dtype.names}


# ==========================================
# 数据加载
# ==========================================
def load_sample_final_metrics(output_root, sample_idx, methods=METHODS):
    """
    加载单个样本的最终指标（TARGET_WELLS 口井时的状态）。

    单 trial 方法（active / greedy）  : 读取 'history'，取最后一步
    多 trial 方法（random / lhs） : 读取 'all_trial_histories'，对每次 trial
                                       取最后一步，再求 10 次 trial 的均值

    返回：
        dict: {method: {'K_r2', 'K_rmse', 'C_r2', 'C_rmse', 'C_var',
                         'C_mean_std', 'n_wells', 'initial_mean_std'}}
              若某方法数据不存在或读取失败则该键对应 None。
    """
    result = {}
    for method in methods:
        path = os.path.join(output_root, f'sample_{sample_idx:03d}',
                            f'{method}.npz')
        if not os.path.exists(path):
            result[method] = None
            continue
        try:
            data = np.load(path, allow_pickle=True)
            if method in MULTI_TRIAL_METHODS:
                # random / lhs — 取多次试验的均值
                all_hist = data['all_trial_histories']
                finals = []
                for trial_hist in all_hist:
                    e = _as_dict(trial_hist[-1])
                    finals.append({k: float(e[k])
                                   for k in ('K_r2', 'K_rmse', 'C_r2',
                                             'C_rmse', 'C_var', 'C_mean_std')})
                avg = {k: float(np.mean([f[k] for f in finals]))
                       for k in finals[0]}
                avg['n_wells'] = int(_as_dict(all_hist[0][-1])['n_wells'])
                avg['initial_mean_std'] = float(data['initial_mean_std'])
                result[method] = avg
            else:
                # active / greedy — 单次确定性 history
                hist  = data['history']
                entry = _as_dict(hist[-1])
                result[method] = {
                    'K_r2':             float(entry['K_r2']),
                    'K_rmse':           float(entry['K_rmse']),
                    'C_r2':             float(entry['C_r2']),
                    'C_rmse':           float(entry['C_rmse']),
                    'C_var':            float(entry['C_var']),
                    'C_mean_std':       float(entry['C_mean_std']),
                    'n_wells':          int(entry['n_wells']),
                    'initial_mean_std': float(data['initial_mean_std']),
                }
        except Exception as exc:
            print(f"  [Warning] 样本 {sample_idx} 方法 {method} 读取失败: {exc}")
            result[method] = None
    return result


def load_all_samples(output_root, n_samples=20, start=0, methods=METHODS):
    """
    从 output_root 中加载 n_samples 个有效样本的最终指标。

    以 start 为起点向后扫描，跳过不存在的样本，直到收集到 n_samples 个
    或 sample_index 超出 start + n_samples * 3（搜索上限）为止。

    返回：
        collected : {method: {'K_r2', 'K_rmse', 'C_r2', 'C_rmse',
                               'C_var', 'C_mean_std'} 各为 np.ndarray}
        loaded_idx: List[int]  已成功加载的样本编号
    """
    keys = ('K_r2', 'K_rmse', 'C_r2', 'C_rmse', 'C_var', 'C_mean_std')
    per_method = {m: {k: [] for k in keys} for m in methods}
    loaded_idx = []

    search_limit = start + n_samples * 3   # 最多搜索这么多个目录
    for si in range(start, search_limit):
        if len(loaded_idx) >= n_samples:
            break
        sample_res = load_sample_final_metrics(output_root, si, methods)
        if all(v is None for v in sample_res.values()):
            continue
        loaded_idx.append(si)
        for m in methods:
            m_data = sample_res[m]
            for k in keys:
                per_method[m][k].append(
                    m_data[k] if (m_data is not None) else np.nan)

    for m in methods:
        for k in keys:
            per_method[m][k] = np.array(per_method[m][k], dtype=np.float32)

    print(f"[Info] 已加载 {len(loaded_idx)} 个样本 | "
          f"索引范围: {loaded_idx[0] if loaded_idx else '—'} ~ "
          f"{loaded_idx[-1] if loaded_idx else '—'}")
    return per_method, loaded_idx


def load_learning_curves(output_root, sample_indices, methods=METHODS):
    """
    加载逐步学习曲线（每口井添加后的指标变化）。

    单 trial 方法（active / greedy）  : 直接遍历 'history'
    多 trial 方法（random / lhs） : 遍历 'all_trial_histories' 的每次 trial，
                                       把所有 trial 的逐步指标都汇入对应 n_wells 桶中
                                       （后续 mean/std 在桶内计算，反映该井数下
                                        所有 trial × 所有 sample 的总体分布）

    返回：
        curves: {method: {'n_wells', 'K_r2_mean', 'K_r2_std',
                           'C_r2_mean', 'C_r2_std',
                           'C_mean_std_mean', 'C_mean_std_std'}}
                若某方法无数据则对应值为 None。
    """
    raw = {m: defaultdict(lambda: {'C_r2': [], 'C_mean_std': [], 'K_r2': []})
           for m in methods}

    for si in sample_indices:
        for method in methods:
            path = os.path.join(output_root, f'sample_{si:03d}',
                                f'{method}.npz')
            if not os.path.exists(path):
                continue
            try:
                data = np.load(path, allow_pickle=True)
                if method in MULTI_TRIAL_METHODS:
                    # random / lhs
                    for trial_hist in data['all_trial_histories']:
                        for entry in trial_hist:
                            e  = _as_dict(entry)
                            nw = int(e['n_wells'])
                            raw[method][nw]['C_r2'].append(float(e['C_r2']))
                            raw[method][nw]['C_mean_std'].append(
                                float(e['C_mean_std']))
                            raw[method][nw]['K_r2'].append(float(e['K_r2']))
                else:
                    # active / greedy
                    for entry in data['history']:
                        e  = _as_dict(entry)
                        nw = int(e['n_wells'])
                        raw[method][nw]['C_r2'].append(float(e['C_r2']))
                        raw[method][nw]['C_mean_std'].append(float(e['C_mean_std']))
                        raw[method][nw]['K_r2'].append(float(e['K_r2']))
            except Exception as exc:
                print(f"  [Warning] 学习曲线 sample={si} method={method}: {exc}")

    curves = {}
    for m in methods:
        nw_list = sorted(raw[m].keys())
        if not nw_list:
            curves[m] = None
            continue
        curves[m] = {
            'n_wells':          nw_list,
            'K_r2_mean':        np.array([np.mean(raw[m][nw]['K_r2'])
                                          for nw in nw_list]),
            'K_r2_std':         np.array([np.std(raw[m][nw]['K_r2'])
                                          for nw in nw_list]),
            'C_r2_mean':        np.array([np.mean(raw[m][nw]['C_r2'])
                                          for nw in nw_list]),
            'C_r2_std':         np.array([np.std(raw[m][nw]['C_r2'])
                                          for nw in nw_list]),
            'C_mean_std_mean':  np.array([np.mean(raw[m][nw]['C_mean_std'])
                                          for nw in nw_list]),
            'C_mean_std_std':   np.array([np.std(raw[m][nw]['C_mean_std'])
                                          for nw in nw_list]),
        }
    return curves


# ==========================================
# 图 1：终态指标箱线图（主图）
# ==========================================
def plot_boxplots(collected, out_dir, methods=METHODS):
    """
    三方法 × 三指标（K_R2 / C_R2 / C_mean_std）并排箱线图。

    K_R2、C_R2、C_mean_std 各占一个子图，方法颜色区分。
    去除子图小标题和X轴标签，在第一个子图中显示全局图例。
    """
    apply_journal_style()

    metric_cfg = [
        ('K_r2', 'K_R²'),
        ('C_r2', 'C_R²'),
        ('C_mean_std', 'Uncertainty'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    valid_n = np.sum(~np.isnan(collected[methods[0]]['K_r2'])) \
        if methods else 0

    for ax, (metric_key, ylabel) in zip(axes, metric_cfg):
        data_list, colors, labels = [], [], []
        for m in methods:
            vals = collected[m][metric_key]
            valid = vals[~np.isnan(vals)]
            if len(valid) == 0:
                continue
            data_list.append(valid)
            colors.append(METHOD_COLORS[m])
            labels.append(METHOD_LABELS[m])

        if not data_list:
            ax.axis('off')
            continue

        bp = ax.boxplot(
            data_list, patch_artist=True, widths=0.5,
            medianprops=dict(color='black', linewidth=1.8),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            flierprops=dict(marker='o', markersize=3,
                            markerfacecolor='gray', alpha=0.6))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # 隐藏 X 轴的刻度和标签
        ax.set_xticks([])

        ax.set_ylabel(ylabel, fontsize=JOURNAL_RC['axes.labelsize'])
        ax.set_axisbelow(True)

    # 在第一个子图 (axes[0]) 中添加图例
    if 'bp' in locals() and labels:
        axes[0].legend(bp['boxes'], labels,
                       fontsize=JOURNAL_RC.get('legend.fontsize', 11),
                       loc='best', framealpha=0.9)

    fig.suptitle(
        f'Final-State Comparison (n = {valid_n} samples)',
        fontsize=JOURNAL_RC['figure.titlesize'], y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig1_boxplots.png')
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    print(f"  [Saved] {path}")


# ==========================================
# 图 2：学习曲线
# ==========================================
from matplotlib.ticker import MaxNLocator


def plot_learning_curves(curves, out_dir, methods=METHODS):
    """
    K_R2 / C_R2 / C_mean_std 随井数增加的变化（均值 ± 1σ 阴影）。
    去除小标题，强制 X 轴为整数刻度，且只保留第一个图的图例。
    """
    apply_journal_style()

    subplot_cfg = [
        ('K_r2_mean', 'K_r2_std', 'K_R²'),
        ('C_r2_mean', 'C_r2_std', 'C_R²'),
        ('C_mean_std_mean', 'C_mean_std_std', 'Uncertainty'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for i, (ax, (mean_key, std_key, ylabel)) in enumerate(zip(axes, subplot_cfg)):
        plotted = False
        for m in methods:
            if curves.get(m) is None:
                continue
            nw = np.array(curves[m]['n_wells'])
            mean = curves[m][mean_key]
            std = curves[m][std_key]
            color = METHOD_COLORS[m]
            ax.plot(nw, mean, color=color, linewidth=1.8,
                    label=METHOD_LABELS[m])
            ax.fill_between(nw, mean - std, mean + std,
                            color=color, alpha=0.18)
            plotted = True

        ax.set_xlabel('Number of Wells',
                      fontsize=JOURNAL_RC['axes.labelsize'])
        ax.set_ylabel(ylabel, fontsize=JOURNAL_RC['axes.labelsize'])

        # 强制 X 轴只使用整数刻度
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # 只在第一个子图 (i == 0) 显示图例
        if plotted and i == 0:
            ax.legend(fontsize=JOURNAL_RC['legend.fontsize'], loc='best')

        ax.set_axisbelow(True)

    fig.suptitle('Learning Curves: Performance vs. Number of Wells',
                 fontsize=JOURNAL_RC['figure.titlesize'], y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig2_learning_curves.png')
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    print(f"  [Saved] {path}")


# ==========================================
# 图 3：K_R2 vs C_R2 散点图
# ==========================================
def plot_scatter(collected, out_dir, methods=METHODS):
    """
    每个样本一个点（K_R2, C_R2），四方法用颜色和标记区分。
    辅助对角线（y = x）提示两指标的一致性。
    """
    apply_journal_style()

    markers = {'active': 'o', 'greedy': 's', 'lhs': 'D', 'random': '^'}

    fig, ax = plt.subplots(figsize=(6, 5.5))

    all_vals = []
    for m in methods:
        k_r2 = collected[m]['K_r2']
        c_r2 = collected[m]['C_r2']
        mask = ~(np.isnan(k_r2) | np.isnan(c_r2))
        ax.scatter(k_r2[mask], c_r2[mask],
                   c=METHOD_COLORS[m], marker=markers.get(m, 'o'),
                   label=METHOD_LABELS[m],
                   alpha=0.75, s=45, edgecolors='none')
        all_vals.extend(k_r2[mask].tolist())
        all_vals.extend(c_r2[mask].tolist())

    if all_vals:
        lo, hi = min(all_vals) - 0.02, max(all_vals) + 0.02
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8, alpha=0.4,
                label='y = x')

    ax.set_xlabel('K_R²', fontsize=JOURNAL_RC['axes.labelsize'])
    ax.set_ylabel('C_R²', fontsize=JOURNAL_RC['axes.labelsize'])
    ax.set_title('K_R² vs C_R² (per sample)',
                 fontsize=JOURNAL_RC['axes.titlesize'])
    ax.legend(fontsize=JOURNAL_RC['legend.fontsize'])
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join(out_dir, 'fig3_scatter_kr2_cr2.png')
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    print(f"  [Saved] {path}")


# ==========================================
# 统计显著性检验
# ==========================================
def run_statistical_tests(collected, methods=METHODS):
    """
    对布井策略的最终性能进行统计显著性检验。

    检验设计
    ─────────────────────────────────────────────────────────────────────────
    Wilcoxon 符号秩检验（单侧，paired）
        适用理由：每个样本在相同随机场上同时运行多种策略，数据天然配对。
        Wilcoxon 检验对分布形态无假设（非参数），比配对 t 检验更稳健。
        H₀：两方法性能相同（中位数差 = 0）
        单侧方向：Active 优于基线（K_R² / C_R² 更高；C_mean_std 更低）

    Cohen's d（配对差值版本）
        d = mean(diff) / std(diff, ddof=1)
        |d| < 0.2 可忽略  |d| 0.2–0.5 小  |d| 0.5–0.8 中  |d| > 0.8 大
    ─────────────────────────────────────────────────────────────────────────

    返回：
        dict: {
            'AL_vs_Greedy': {metric: {'n', 'wilcoxon_stat', 'p_one', 'Cohens_d'}},
            'AL_vs_LHS':    {metric: {...}},
            'AL_vs_Random': {metric: {...}},
        }
    """
    if 'active' not in methods:
        return {}

    metrics = {
        'K_r2':       'higher',   # Active 应更高
        'C_r2':       'higher',
        'C_mean_std': 'lower',    # Active 应更低（不确定性更小）
    }

    # 基线列表：(方法名, 显示标签)
    baselines = [
        ('greedy', 'AL_vs_Greedy'),
        ('lhs', 'AL_vs_LHS'),
        ('random', 'AL_vs_Random'),
    ]

    comparisons = {}
    for baseline, label in baselines:
        if baseline not in methods:
            continue
        comparisons[label] = {}

        for metric, direction in metrics.items():
            a = collected['active'][metric]
            b = collected[baseline][metric]

            # 去除任意一方为 NaN 的配对
            mask = ~(np.isnan(a) | np.isnan(b))
            a_c, b_c = a[mask], b[mask]
            n = int(mask.sum())

            if n < 4:
                comparisons[label][metric] = dict(
                    n=n, wilcoxon_stat=np.nan, p_one=np.nan, Cohens_d=np.nan)
                continue

            diff = a_c - b_c

            # Wilcoxon 符号秩检验（双侧），再转单侧
            try:
                stat, p_two = stats.wilcoxon(a_c, b_c, alternative='two-sided')
                mean_diff = float(np.mean(diff))
                correct_direction = (mean_diff > 0) if direction == 'higher' \
                                    else (mean_diff < 0)
                p_one = float(p_two / 2) if correct_direction \
                        else float(1 - p_two / 2)
            except Exception:
                stat, p_one = np.nan, np.nan

            # Cohen's d（配对差值版本）
            d_diff = diff if direction == 'higher' else -diff
            Cohens_d = float(np.mean(d_diff) / np.std(d_diff, ddof=1)) \
                       if np.std(d_diff, ddof=1) > 0 else np.nan

            comparisons[label][metric] = dict(
                n             = n,
                wilcoxon_stat = float(stat) if not np.isnan(stat) else np.nan,
                p_one         = p_one,
                Cohens_d      = Cohens_d,
            )

    return comparisons


# ==========================================
# 阶段性分析（动态学习曲线三阶段划分）
# ==========================================
PHASE_BOUNDS = {
    'Early (3–8)':   (3,  8),
    'Mid   (9–14)':  (9,  14),
    'Late  (15–20)': (15, 20),
}


def compute_phase_stats(curves, methods=METHODS):
    """
    对学习曲线按三阶段划分，计算各方法在每个阶段的性能均值及斜率。

    斜率（slope）= 阶段末尾均值 − 阶段起始均值，反映该阶段的改善速率。
    对 C_mean_std（越低越好），斜率取相反数使"正值 = 改善"。
    """
    metric_cfg = [
        ('K_r2_mean',       'K_r2',       'higher'),
        ('C_r2_mean',       'C_r2',       'higher'),
        ('C_mean_std_mean', 'C_mean_std', 'lower'),
    ]

    phase_stats = {}

    for m in methods:
        if curves.get(m) is None:
            phase_stats[m] = None
            continue

        nw_arr = np.array(curves[m]['n_wells'])
        phase_stats[m] = {}

        for phase_label, (lo, hi) in PHASE_BOUNDS.items():
            mask = (nw_arr >= lo) & (nw_arr <= hi)
            phase_nw = nw_arr[mask]
            phase_stats[m][phase_label] = {}

            for mean_key, short_key, direction in metric_cfg:
                vals = curves[m][mean_key][mask]
                n_pts = int(mask.sum())

                if n_pts == 0:
                    phase_stats[m][phase_label][short_key] = {
                        'mean': np.nan, 'slope': np.nan, 'n_points': 0}
                    continue

                mean_val = float(np.mean(vals))

                if n_pts >= 2:
                    raw_slope = float(vals[-1] - vals[0])
                    slope = raw_slope if direction == 'higher' else -raw_slope
                else:
                    slope = np.nan

                phase_stats[m][phase_label][short_key] = {
                    'mean':     mean_val,
                    'slope':    slope,
                    'n_points': n_pts,
                }

    return phase_stats


def write_phase_analysis(phase_stats, curves, out_dir, methods=METHODS):
    """
    将三阶段分析结果格式化输出到终端并保存为 phase_analysis.txt。
    """
    metric_labels = {
        'K_r2':       'K_R²',
        'C_r2':       'C_R²',
        'C_mean_std': 'C mean std',
    }
    metrics = list(metric_labels.keys())
    phases  = list(PHASE_BOUNDS.keys())

    lines = []
    lines.append("=" * 90)
    lines.append("  Phase Analysis: Learning Curve Decomposition (3 Stages)")
    lines.append("  Early (3–8 wells) | Mid (9–14 wells) | Late (15–20 wells)")
    lines.append("  Slope = end_value − start_value within phase")
    lines.append("  (For C_mean_std: slope is negated so positive = improvement)")
    lines.append("=" * 90)

    # ── 表 1：阶段均值 ──────────────────────────────────────────────────────
    lines.append("")
    lines.append("Table 1. Mean metric value in each phase")
    lines.append("-" * 90)

    col_method = 18
    col_phase  = 22
    col_metric = 14

    header = (f"  {'Method':<{col_method}} {'Phase':<{col_phase}} "
              + "".join(f"{metric_labels[mk]:>{col_metric}}" for mk in metrics))
    lines.append(header)
    lines.append("  " + "-" * 84)

    for m in methods:
        if phase_stats.get(m) is None:
            continue
        first_method = True
        for ph in phases:
            meth_col = METHOD_LABELS.get(m, m) if first_method else ""
            first_method = False
            row = f"  {meth_col:<{col_method}} {ph:<{col_phase}}"
            for mk in metrics:
                val = phase_stats[m][ph][mk]['mean']
                row += f"{val:>{col_metric}.4f}" if not np.isnan(val) else f"{'N/A':>{col_metric}}"
            lines.append(row)
        lines.append("  " + "·" * 84)

    # ── 表 2：阶段斜率（改善速率）──────────────────────────────────────────
    lines.append("")
    lines.append("Table 2. Slope (improvement rate) in each phase  [positive = improvement]")
    lines.append("-" * 90)
    lines.append(header)
    lines.append("  " + "-" * 84)

    for m in methods:
        if phase_stats.get(m) is None:
            continue
        first_method = True
        for ph in phases:
            meth_col = METHOD_LABELS.get(m, m) if first_method else ""
            first_method = False
            row = f"  {meth_col:<{col_method}} {ph:<{col_phase}}"
            for mk in metrics:
                val = phase_stats[m][ph][mk]['slope']
                row += f"{val:>{col_metric}.4f}" if not np.isnan(val) else f"{'N/A':>{col_metric}}"
            lines.append(row)
        lines.append("  " + "·" * 84)

    # ── 量化描述 ──────────────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 90)
    lines.append("  Quantitative Summary for Manuscript (Section 4.3.2)")
    lines.append("=" * 90)

    if 'active' in methods and phase_stats.get('active') is not None:
        lines.append("")
        lines.append("  [C mean std — uncertainty reduction across phases]")
        for ph in phases:
            row_parts = []
            for m in methods:
                if phase_stats.get(m) is None:
                    continue
                mean_v  = phase_stats[m][ph]['C_mean_std']['mean']
                slope_v = phase_stats[m][ph]['C_mean_std']['slope']
                row_parts.append(
                    f"{METHOD_LABELS.get(m,m)}: mean={mean_v:.4f}, "
                    f"slope={slope_v:+.4f}")
            lines.append(f"  {ph}: " + " | ".join(row_parts))

        lines.append("")
        lines.append("  [C_R² — prediction accuracy across phases]")
        for ph in phases:
            row_parts = []
            for m in methods:
                if phase_stats.get(m) is None:
                    continue
                mean_v  = phase_stats[m][ph]['C_r2']['mean']
                slope_v = phase_stats[m][ph]['C_r2']['slope']
                row_parts.append(
                    f"{METHOD_LABELS.get(m,m)}: mean={mean_v:.4f}, "
                    f"slope={slope_v:+.4f}")
            lines.append(f"  {ph}: " + " | ".join(row_parts))

        # 关键结论：AL 在早期阶段的斜率优势（对所有非 active 方法比较）
        lines.append("")
        lines.append("  [Key finding: Early-phase slope comparison (C mean std)]")
        al_early = phase_stats['active']['Early (3–8)']['C_mean_std']['slope']
        for m in methods:
            if m == 'active' or phase_stats.get(m) is None:
                continue
            b_early = phase_stats[m]['Early (3–8)']['C_mean_std']['slope']
            if not (np.isnan(al_early) or np.isnan(b_early) or b_early == 0):
                ratio = al_early / b_early if b_early != 0 else np.nan
                lines.append(
                    f"  AL vs {METHOD_LABELS.get(m,m):<16}: "
                    f"AL slope={al_early:+.4f}, baseline slope={b_early:+.4f}"
                    + (f", ratio={ratio:.2f}x" if not np.isnan(ratio) else ""))

    lines.append("=" * 90)

    text = "\n".join(lines)
    print("\n" + text)

    path = os.path.join(out_dir, 'phase_analysis.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text + "\n")
    print(f"\n  [Saved] {path}")
    return text


def plot_learning_curves_with_phases(curves, out_dir, methods=METHODS):
    """
    在原学习曲线基础上叠加三阶段分隔线和阶段标注。
    """
    apply_journal_style()

    subplot_cfg = [
        ('K_r2_mean', 'K_r2_std', 'K_R²'),
        ('C_r2_mean', 'C_r2_std', 'C_R²'),
        ('C_mean_std_mean', 'C_mean_std_std', 'Uncertainty'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    phase_dividers = [8.5, 14.5]
    phase_labels   = ['Early\n(3–8)', 'Mid\n(9–14)', 'Late\n(15–20)']
    phase_centers  = [5.5, 11.5, 17.5]

    for i, (ax, (mean_key, std_key, ylabel)) in enumerate(zip(axes, subplot_cfg)):
        plotted = False
        y_all = []

        for m in methods:
            if curves.get(m) is None:
                continue
            nw = np.array(curves[m]['n_wells'])
            mean = curves[m][mean_key]
            std = curves[m][std_key]
            color = METHOD_COLORS[m]
            ax.plot(nw, mean, color=color, linewidth=1.8,
                    label=METHOD_LABELS[m])
            ax.fill_between(nw, mean - std, mean + std,
                            color=color, alpha=0.18)
            y_all.extend((mean - std).tolist())
            y_all.extend((mean + std).tolist())
            plotted = True

        if plotted and y_all:
            y_min, y_max = min(y_all), max(y_all)
            y_range = y_max - y_min

            ax.set_ylim(bottom=y_min - y_range * 0.05, top=y_max + y_range * 0.18)
            y_label_pos = y_max + y_range * 0.06

            for xd in phase_dividers:
                ax.axvline(x=xd, color='gray', linewidth=0.8,
                           linestyle='--', alpha=0.55, zorder=0)

            for xc, pl in zip(phase_centers, phase_labels):
                ax.text(xc, y_label_pos, pl,
                        ha='center', va='bottom',
                        fontsize=9, color='dimgray',
                        transform=ax.transData)

        ax.set_xlabel('Number of Wells',
                      fontsize=JOURNAL_RC['axes.labelsize'])
        ax.set_ylabel(ylabel, fontsize=JOURNAL_RC['axes.labelsize'])

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if plotted and i == 0:
            ax.legend(fontsize=JOURNAL_RC['legend.fontsize'], loc='best')

        ax.set_axisbelow(True)

    fig.suptitle(
        'Learning Curves with Phase Decomposition\n'
        'Early (3–8 wells) | Mid (9–14 wells) | Late (15–20 wells)',
        fontsize=JOURNAL_RC['figure.titlesize'], y=1.02)
    fig.tight_layout()

    path = os.path.join(out_dir, 'fig2_learning_curves_phases.png')
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    print(f"  [Saved] {path}")


def _fmt_p(p):
    """期刊标准 p 值格式化。"""
    if np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _sig(p):
    """显著性星号。"""
    if np.isnan(p):   return "n.s."
    if p < 0.001:     return "***"
    if p < 0.01:      return "**"
    if p < 0.05:      return "*"
    return "n.s."


def write_statistical_tests(test_results, out_dir):
    """
    将统计检验结果格式化输出到终端并保存为 statistical_tests.txt。
    """
    metric_labels = {
        'K_r2':       'K_R²      (↑ better)',
        'C_r2':       'C_R²      (↑ better)',
        'C_mean_std': 'C mean std (↓ better)',
    }

    lines = []
    lines.append("=" * 82)
    lines.append("  Statistical Tests: Active Learning vs. Baselines")
    lines.append("  Wilcoxon signed-rank test (one-sided, paired) + Cohen's d")
    lines.append("=" * 82)
    lines.append("  Rationale: Each sample uses the same random field for all methods")
    lines.append("  (natural paired structure). Wilcoxon is non-parametric and robust")
    lines.append("  for small n. One-sided p tests the direction Active > baseline.")
    lines.append("-" * 82)
    lines.append(f"  {'Comparison':<22} {'Metric':<26} {'n':>4} "
                 f"{'W stat':>10} {'p (one-sided)':>15} {'Cohens_d':>10} {'Sig.':>6}")
    lines.append("  " + "-" * 78)

    for comp_label, metrics in test_results.items():
        display = comp_label.replace('AL_vs_', 'AL vs ')
        first = True
        for metric, res in metrics.items():
            comp_col = display if first else ""
            first = False
            n   = res['n']
            w   = res['wilcoxon_stat']
            p   = res['p_one']
            d   = res['Cohens_d']
            sig = _sig(p)
            w_str = f"{w:.3f}" if not np.isnan(w) else "N/A"
            d_str = f"{d:.3f}" if not np.isnan(d) else "N/A"
            lines.append(
                f"  {comp_col:<22} {metric_labels[metric]:<26} {n:>4} "
                f"{w_str:>10} {_fmt_p(p):>15} {d_str:>10} {sig:>6}")
        lines.append("  " + "-" * 78)

    lines.append("")
    lines.append("  Cohen's d interpretation (paired difference):")
    lines.append("    |d| < 0.2  negligible  |  0.2–0.5  small  |  "
                 "0.5–0.8  medium  |  > 0.8  large")
    lines.append("  * p < 0.05   ** p < 0.01   *** p < 0.001  (one-sided Wilcoxon)")
    lines.append("=" * 82)

    text = "\n".join(lines)
    print("\n" + text)

    path = os.path.join(out_dir, 'statistical_tests.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text + "\n")
    print(f"\n  [Saved] {path}")
    return text


# ==========================================
# 汇总文本
# ==========================================
def write_summary(collected, out_dir, methods=METHODS):
    col_w = (18, 22, 22, 22)
    header = (f"{'Method':<{col_w[0]}} "
              f"{'K_R2 mean±std':>{col_w[1]}} "
              f"{'C_R2 mean±std':>{col_w[2]}} "
              f"{'C_mean_std mean±std':>{col_w[3]}}")
    sep = "=" * (sum(col_w) + 3)
    lines = [
        "Active Well Selection: Multi-Sample Evaluation Summary",
        sep,
        header,
        "-" * (sum(col_w) + 3),
    ]

    for m in methods:
        k_r2  = collected[m]['K_r2']
        c_r2  = collected[m]['C_r2']
        cmstd = collected[m]['C_mean_std']
        mask  = ~np.isnan(k_r2)
        n     = int(np.sum(mask))

        def fmt(arr):
            v = arr[mask]
            return f"{np.mean(v):.4f}±{np.std(v):.4f} (n={n})"

        lines.append(
            f"{METHOD_LABELS[m]:<{col_w[0]}} "
            f"{fmt(k_r2):>{col_w[1]}} "
            f"{fmt(c_r2):>{col_w[2]}} "
            f"{fmt(cmstd):>{col_w[3]}}")

    lines.append(sep)

    # 方法间差值（active vs others）
    if 'active' in methods:
        lines.append("")
        lines.append("Improvement of Active over baselines (final C_mean_std):")
        act_mask = ~np.isnan(collected['active']['C_mean_std'])
        act_mean = np.mean(collected['active']['C_mean_std'][act_mask])
        for m in methods:
            if m == 'active':
                continue
            b_mask = ~np.isnan(collected[m]['C_mean_std'])
            b_mean = np.mean(collected[m]['C_mean_std'][b_mask])
            if b_mean > 0:
                imp = (1 - act_mean / b_mean) * 100
                lines.append(f"  Active vs {METHOD_LABELS[m]:<18}: "
                              f"{imp:+.1f}% reduction in C_mean_std")

    text = "\n".join(lines)
    print("\n" + text)
    path = os.path.join(out_dir, 'summary.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text + "\n")
    print(f"\n  [Saved] {path}")


# ============================================================================
# 空间分布统计模块（已扩展为 4 方法 + 适配多 trial）
# ============================================================================
def compute_spatial_stats(wells, n_init=N_INIT_WELLS, grid_size=GRID_SIZE,
                          plume_mask=None, boundary_width=BOUNDARY_WIDTH):
    """
    计算单组井位的 6 项空间分布统计指标。
    仅统计新增井（排除前 n_init 口初始井）。
    """
    wells_arr = np.array(wells)
    added     = wells_arr[n_init:]
    n_added   = len(added)
    G         = grid_size

    stats_d = dict(n_added=n_added, ann_distance=np.nan, clark_evans=np.nan,
                   coverage_rate=np.nan, boundary_rate=np.nan,
                   plume_inside=np.nan, centroid_dist=np.nan)

    if n_added == 0:
        return stats_d

    # ① ANN
    if n_added >= 2:
        dists = []
        for i in range(n_added):
            diff = added - added[i]
            d = np.sqrt((diff ** 2).sum(axis=1))
            d[i] = np.inf
            dists.append(d.min())
        ann = float(np.mean(dists))
        stats_d['ann_distance'] = ann

        # ② Clark-Evans R
        density  = n_added / float(G * G)
        ann_exp  = 0.5 / np.sqrt(density)
        stats_d['clark_evans'] = ann / ann_exp

    # ③ Coverage rate
    r       = boundary_width / 2.0
    covered = np.zeros((G, G), dtype=bool)
    rr, cc  = np.meshgrid(np.arange(G), np.arange(G), indexing='ij')
    for w in added:
        d2 = (rr - w[0]) ** 2 + (cc - w[1]) ** 2
        covered |= (d2 <= r ** 2)
    stats_d['coverage_rate'] = float(covered.sum()) / (G * G)

    # ④ Boundary rate
    in_b = (
        (added[:, 0] <= boundary_width) |
        (added[:, 0] >= G - 1 - boundary_width) |
        (added[:, 1] <= boundary_width) |
        (added[:, 1] >= G - 1 - boundary_width)
    )
    stats_d['boundary_rate'] = float(in_b.sum()) / n_added

    # ⑤ Plume inside
    if plume_mask is not None:
        pm = np.array(plume_mask, dtype=bool)
        inside = np.array([pm[int(w[0]), int(w[1])] for w in added])
        stats_d['plume_inside'] = float(inside.sum()) / n_added

    # ⑥ Centroid offset (归一化)
    centroid    = added.mean(axis=0)
    grid_center = np.array([(G - 1) / 2, (G - 1) / 2])
    raw_dist    = float(np.linalg.norm(centroid - grid_center))
    max_dist    = np.sqrt(2) * (G - 1) / 2.0
    stats_d['centroid_dist'] = raw_dist / max_dist

    return stats_d


def _get_plume_mask(sample_dir):
    """
    从 active/greedy/lhs 任一可用 .npz 的 C_post_std 字段生成 plume_mask。
    注意：lhs.npz 与 random.npz 一样是多 trial 结构，不直接含 C_post_std，
          因此 plume_mask 只能从 active.npz 或 greedy.npz 提取。
    """
    for fname in ['active.npz', 'greedy.npz']:
        path = os.path.join(sample_dir, fname)
        if not os.path.exists(path):
            continue
        try:
            data = np.load(path, allow_pickle=True)
            if 'C_post_std' in data:
                c_std = np.array(data['C_post_std'])
                if c_std.ndim == 3:
                    weights = np.array(TIME_WEIGHTS)
                    var_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
                    for t in range(min(N_TIME_STEPS, c_std.shape[0])):
                        var_map += weights[t] * (c_std[t] ** 2)
                    threshold = np.percentile(var_map, 75)
                    return var_map >= threshold
        except Exception:
            pass
    return None


def _extract_wells(data):
    """从单 history npz 数据中提取 final_wells。"""
    if 'final_wells' in data:
        fw = data['final_wells']
        return fw.tolist() if hasattr(fw, 'tolist') else fw
    if 'history' in data:
        h = data['history']
        if hasattr(h, 'tolist'):
            h = h.tolist()
        if isinstance(h, list) and len(h) > 0:
            last = _as_dict(h[-1])
            if 'wells' in last:
                return last['wells']
    return None


def _load_multi_trial_spatial(npz_path, plume_mask):
    """
    从多 trial npz (random / lhs) 中加载所有 trial 的最终井位，
    分别计算空间指标后取均值。
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        all_hist = None
        for key in ['all_trial_histories', 'all_histories']:
            if key in data:
                all_hist = data[key]
                if hasattr(all_hist, 'tolist'):
                    all_hist = all_hist.tolist()
                break

        if all_hist is None or len(all_hist) == 0:
            return None

        trial_list = []
        for trial_hist in all_hist:
            if not isinstance(trial_hist, list) or len(trial_hist) == 0:
                continue
            last = _as_dict(trial_hist[-1])
            if 'wells' not in last:
                continue
            ts = compute_spatial_stats(
                last['wells'], N_INIT_WELLS, GRID_SIZE, plume_mask)
            trial_list.append(ts)

        if not trial_list:
            return None

        avg = {'n_added': trial_list[0]['n_added']}
        for k in SPATIAL_METRIC_KEYS:
            vals = [ts[k] for ts in trial_list if not np.isnan(ts[k])]
            avg[k] = float(np.mean(vals)) if vals else np.nan
        return avg
    except Exception as e:
        print(f"  [Warning] 多 trial 空间统计加载失败 {npz_path}: {e}")
        return None


def load_spatial_sample(input_dir, sample_idx, methods=METHODS):
    """加载单样本的多方法空间统计。返回 {method: stats_dict 或 None}。"""
    sample_dir = os.path.join(input_dir, f'sample_{sample_idx:03d}')
    if not os.path.isdir(sample_dir):
        return {m: None for m in methods}

    plume_mask = _get_plume_mask(sample_dir)
    result = {}

    for method in methods:
        path = os.path.join(sample_dir, f'{method}.npz')
        if not os.path.exists(path):
            result[method] = None
            continue

        if method in MULTI_TRIAL_METHODS:
            # random / lhs — 取多次 trial 均值
            result[method] = _load_multi_trial_spatial(path, plume_mask)
        else:
            # active / greedy — 单次 history
            try:
                data  = np.load(path, allow_pickle=True)
                wells = _extract_wells(data)
                if wells is not None:
                    result[method] = compute_spatial_stats(
                        wells, N_INIT_WELLS, GRID_SIZE, plume_mask)
                else:
                    result[method] = None
            except Exception as e:
                print(f"  [Warning] sample {sample_idx} {method}: {e}")
                result[method] = None

    return result


def load_all_spatial(input_dir, n_samples, methods=METHODS):
    """对所有样本聚合空间统计。返回 (collected, loaded_idx)。"""
    collected = {m: {k: [] for k in SPATIAL_METRIC_KEYS} for m in methods}
    loaded_idx = []

    for si in range(n_samples):
        sample_res = load_spatial_sample(input_dir, si, methods=methods)
        if all(v is None for v in sample_res.values()):
            continue
        loaded_idx.append(si)
        for m in methods:
            s = sample_res[m]
            for k in SPATIAL_METRIC_KEYS:
                collected[m][k].append(s[k] if s is not None else np.nan)

    for m in methods:
        for k in SPATIAL_METRIC_KEYS:
            collected[m][k] = np.array(collected[m][k], dtype=np.float64)

    print(f"[Info] 空间统计：已加载 {len(loaded_idx)} 个样本 | "
          f"索引范围: {loaded_idx[0] if loaded_idx else '—'} ~ "
          f"{loaded_idx[-1] if loaded_idx else '—'}")
    return collected, loaded_idx


def _cohens_d_paired(x, y):
    diff = x - y
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-15))


def run_spatial_tests(spatial_collected, methods=METHODS):
    """对每项空间指标做 Active vs 各基线的 Wilcoxon + Cohen's d。"""
    if 'active' not in methods:
        return {}

    # 基线列表：(方法名, 显示标签)
    baselines = [
        ('greedy', 'AL_vs_Greedy'),
        ('lhs', 'AL_vs_LHS'),
        ('random', 'AL_vs_Random'),
    ]

    results = {}
    for m_base, comp in baselines:
        if m_base not in methods:
            continue
        results[comp] = {}
        for k in SPATIAL_METRIC_KEYS:
            a = spatial_collected['active'][k]
            b = spatial_collected[m_base][k]
            mask = ~(np.isnan(a) | np.isnan(b))
            n = int(mask.sum())
            if n < 5:
                results[comp][k] = dict(n=n, W=np.nan, p=np.nan, d=np.nan)
                continue
            a_v, b_v = a[mask], b[mask]
            diff    = a_v - b_v
            nonzero = diff[diff != 0]
            if len(nonzero) < 5:
                results[comp][k] = dict(n=n, W=np.nan, p=np.nan, d=np.nan)
                continue
            try:
                w, p = wilcoxon(nonzero, alternative='two-sided')
                d = _cohens_d_paired(a_v, b_v)
                results[comp][k] = dict(n=n, W=float(w), p=float(p), d=d)
            except Exception:
                results[comp][k] = dict(n=n, W=np.nan, p=np.nan, d=np.nan)
    return results


def save_spatial_csv(spatial_collected, loaded_idx, out_dir, methods=METHODS):
    path = os.path.join(out_dir, 'spatial_stats.csv')
    header = ','.join(['sample', 'method'] + SPATIAL_METRIC_KEYS)
    with open(path, 'w') as f:
        f.write(header + '\n')
        for i, si in enumerate(loaded_idx):
            for m in methods:
                vals = [str(si), m]
                for k in SPATIAL_METRIC_KEYS:
                    v = spatial_collected[m][k][i]
                    vals.append(f"{v:.6f}" if not np.isnan(v) else "")
                f.write(','.join(vals) + '\n')
    print(f"  [Saved] {path}")


def write_spatial_summary(spatial_collected, spatial_tests, out_dir,
                          methods=METHODS):
    lines = []
    lines.append("=" * 110)
    lines.append("  Spatial Distribution Statistics: Multi-Sample Aggregate")
    lines.append("  (added wells only; initial 3 LHS wells excluded)")
    lines.append("  Note: For multi-trial methods (Random, LHS), values are")
    lines.append("        averaged across the 10 trials per sample.")
    lines.append("=" * 110)

    # 描述统计表
    lines.append("")
    col_metric = 24
    col_method = 22
    header = f"  {'Metric':<{col_metric}}"
    for m in methods:
        header += f"{METHOD_LABELS[m]:>{col_method}}"
    lines.append(header)
    lines.append("  " + "-" * (col_metric + col_method * len(methods)))

    for k in SPATIAL_METRIC_KEYS:
        row = f"  {SPATIAL_METRIC_LABELS[k]:<{col_metric}}"
        for m in methods:
            vals  = spatial_collected[m][k]
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                cell = f"{np.mean(valid):.4f} ± {np.std(valid):.4f}"
            else:
                cell = "N/A"
            row += f"{cell:>{col_method}}"
        lines.append(row)

    # 统计检验表
    if spatial_tests:
        lines.append("")
        lines.append("=" * 110)
        lines.append("  Statistical Tests (Wilcoxon signed-rank, two-sided)")
        lines.append("=" * 110)
        lines.append(f"  {'Metric':<26} {'Comparison':<18} "
                     f"{'n':>4} {'W':>10} {'p':>14} {'Cohen d':>10} {'Sig':>6}")
        lines.append("  " + "-" * 90)

        comp_order = ['AL_vs_Greedy', 'AL_vs_LHS', 'AL_vs_Random']
        for k in SPATIAL_METRIC_KEYS:
            first = True
            for comp in comp_order:
                if comp not in spatial_tests:
                    continue
                r = spatial_tests[comp][k]
                mk_col   = SPATIAL_METRIC_LABELS[k] if first else ""
                first    = False
                comp_lbl = comp.replace('AL_vs_', 'vs ')
                w_str = f"{r['W']:.0f}" if not np.isnan(r['W']) else "N/A"
                d_str = f"{r['d']:.3f}" if not np.isnan(r['d']) else "N/A"
                lines.append(
                    f"  {mk_col:<26} {comp_lbl:<18} {r['n']:>4} "
                    f"{w_str:>10} {_fmt_p(r['p']):>14} {d_str:>10} "
                    f"{_sig(r['p']):>6}")
            lines.append("  " + "-" * 90)

    # 图例
    lines.append("")
    lines.append("  Metric interpretation:")
    lines.append("    ANN Distance  : avg nearest-neighbor dist (grid units, higher=dispersed)")
    lines.append("    Clark-Evans R : R<1 clustered, R=1 random, R>1 dispersed")
    lines.append("    Coverage Rate : fraction of grid covered (higher=wider)")
    lines.append("    Boundary Rate : fraction of wells in edge zone")
    lines.append("    Plume Inside  : fraction in high-variance zone (>75th pct)")
    lines.append("    Centroid Offs : normalized dist to grid center (0=center, 1=corner)")
    lines.append("")
    lines.append("  Cohen's d: |d|<0.2 negligible | 0.2-0.5 small | "
                 "0.5-0.8 medium | >0.8 large")
    lines.append("  * p<0.05  ** p<0.01  *** p<0.001")
    lines.append("=" * 110)

    text = "\n".join(lines)
    print("\n" + text)
    path = os.path.join(out_dir, 'spatial_summary.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text + "\n")
    print(f"\n  [Saved] {path}")


# ============================================================================
# 成本-性能权衡分析
# ============================================================================
def _equiv_well_count(curve, target_value, direction='lower'):
    """
    在学习曲线上反查"达到 target_value 时所需井数"。
    用 log-log 线性插值（学习曲线常呈 power-law 衰减形状）。
    """
    if curve is None:
        return np.nan
    nw = np.array(curve['n_wells'], dtype=float)
    if direction == 'lower':
        vals = np.array(curve['C_mean_std_mean'])
    else:
        vals = np.array(curve['C_r2_mean'])
    valid = (nw > 0) & (vals > 0)
    nw, vals = nw[valid], vals[valid]
    if len(nw) < 2:
        return np.nan

    logx = np.log(nw)
    logy = np.log(vals)
    b, a = np.polyfit(logx, logy, 1)

    if target_value <= 0:
        return np.nan
    log_target = np.log(target_value)
    log_n      = (log_target - a) / b
    return float(np.exp(log_n))


def write_cost_performance_report(collected, curves, out_dir,
                                  progress_path=None,
                                  cost_per_well_low=5000,
                                  cost_per_well_high=50000,
                                  gpu_cost_per_hour=2.0,
                                  methods=METHODS):
    """
    生成成本-性能权衡报告 cost_performance.txt。

    参数：
      cost_per_well_low/high : 单口监测井成本区间（USD）
      gpu_cost_per_hour      : GPU 算力单价（USD/小时）
    """
    import json

    lines = []
    lines.append("=" * 88)
    lines.append("  Cost-Performance Trade-off Analysis")
    lines.append("=" * 88)

    # ── 1) 计算时间 ─────────────────────────────────────────────────────────
    times = {m: [] for m in methods}
    if progress_path and os.path.exists(progress_path):
        try:
            with open(progress_path) as f:
                progress = json.load(f)
            for sk, info in progress.get('completed_samples', {}).items():
                for m in methods:
                    if m in info and info[m] is not None and 'time_min' in info[m]:
                        times[m].append(float(info[m]['time_min']))
        except Exception as e:
            lines.append(f"  [Warning] 读取 progress.json 失败: {e}")

    lines.append("")
    lines.append("  [1] Per-sample computation time (read from progress.json)")
    lines.append(f"  {'Method':<22} {'mean (min)':>14} {'std':>10}  "
                 f"{'samples':>9}")
    lines.append("  " + "-" * 60)
    time_means = {}
    for m in methods:
        if times[m]:
            t_arr = np.array(times[m])
            time_means[m] = float(t_arr.mean())
            lines.append(f"  {METHOD_LABELS[m]:<22} {t_arr.mean():>14.2f} "
                         f"{t_arr.std():>10.2f}  {len(t_arr):>9d}")
        else:
            time_means[m] = np.nan
            lines.append(f"  {METHOD_LABELS[m]:<22} {'N/A':>14} {'N/A':>10}  "
                         f"{0:>9d}")

    # ── 2) 终态性能对比 ──────────────────────────────────────────────────────
    lines.append("")
    lines.append("  [2] Final-state performance (mean across samples)")
    lines.append(f"  {'Method':<22} {'C_mean_std':>14} {'C_R²':>12} {'K_R²':>12}")
    lines.append("  " + "-" * 60)
    final_means = {}
    for m in methods:
        cmstd = collected[m]['C_mean_std']
        cr2   = collected[m]['C_r2']
        kr2   = collected[m]['K_r2']
        mask  = ~np.isnan(cmstd)
        if mask.any():
            final_means[m] = {
                'C_mean_std': float(np.mean(cmstd[mask])),
                'C_r2':       float(np.mean(cr2[mask])),
                'K_r2':       float(np.mean(kr2[mask])),
            }
            lines.append(f"  {METHOD_LABELS[m]:<22} "
                         f"{final_means[m]['C_mean_std']:>14.4f} "
                         f"{final_means[m]['C_r2']:>12.4f} "
                         f"{final_means[m]['K_r2']:>12.4f}")

    # ── 3) 等性能井数节省 ────────────────────────────────────────────────────
    lines.append("")
    lines.append("  [3] Well-saving analysis: how many wells does Active need to")
    lines.append("      reach each baseline's terminal C_mean_std?")
    lines.append("      (log-log interpolation on 100-sample averaged learning curve)")
    lines.append(f"  {'vs Baseline':<22} {'Baseline target':>17} "
                 f"{'Active n_wells':>16} {'Wells saved':>14}")
    lines.append("  " + "-" * 76)

    wells_saved = {}
    if 'active' in methods and curves.get('active') is not None:
        target_n = curves['active']['n_wells'][-1]
        for base in methods:
            if base == 'active' or base not in final_means:
                continue
            target_val = final_means[base]['C_mean_std']
            n_at_target = _equiv_well_count(curves['active'], target_val,
                                            direction='lower')
            saved = target_n - n_at_target if not np.isnan(n_at_target) else np.nan
            wells_saved[base] = saved
            n_str     = f"{n_at_target:.1f}" if not np.isnan(n_at_target) else "N/A"
            saved_str = (f"{saved:.1f}  ({saved/target_n*100:.1f}%)"
                         if not np.isnan(saved) else "N/A")
            lines.append(f"  {METHOD_LABELS[base]:<22} {target_val:>17.4f} "
                         f"{n_str:>16} {saved_str:>14}")

    # ── 4) ROI 估算 ──────────────────────────────────────────────────────────
    if wells_saved:
        lines.append("")
        lines.append("  [4] Return-on-investment estimate")
        lines.append(f"      Per-well infrastructure cost:  "
                     f"${cost_per_well_low:,} ~ ${cost_per_well_high:,}")
        active_min = time_means.get('active', np.nan)
        if not np.isnan(active_min):
            extra_min = active_min - max(
                [time_means.get(m, 0) for m in methods if m != 'active' and
                 not np.isnan(time_means.get(m, np.nan))],
                default=0)
            extra_hours = max(extra_min, 0) / 60.0
            extra_gpu_usd = extra_hours * gpu_cost_per_hour
            lines.append(f"      Active extra GPU time:         "
                         f"{extra_min:.1f} min "
                         f"(~${extra_gpu_usd:.2f} at ${gpu_cost_per_hour}/h)")
        else:
            extra_gpu_usd = np.nan

        lines.append("")
        lines.append(f"  {'vs Baseline':<22} {'Wells saved':>13} "
                     f"{'$ saved (low / high)':>32} {'ROI':>14}")
        lines.append("  " + "-" * 88)
        for base, saved in wells_saved.items():
            if np.isnan(saved) or saved <= 0:
                continue
            saved_low  = saved * cost_per_well_low
            saved_high = saved * cost_per_well_high
            if not np.isnan(extra_gpu_usd) and extra_gpu_usd > 0:
                roi_low  = saved_low  / extra_gpu_usd
                roi_high = saved_high / extra_gpu_usd
                roi_str  = f"{roi_low:.0f}× ~ {roi_high:.0f}×"
            else:
                roi_str = "N/A"
            lines.append(f"  {METHOD_LABELS[base]:<22} {saved:>13.1f} "
                         f"{f'${saved_low:>9,.0f} / ${saved_high:>10,.0f}':>32} "
                         f"{roi_str:>14}")

    # ── 5) 解读 ──────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("  Interpretation:")
    lines.append("    - Active sampling carries higher one-time computational cost,")
    lines.append("      but achieves equivalent C uncertainty with fewer monitoring wells.")
    lines.append("    - In any field-deployment scenario where per-well cost dominates")
    lines.append("      (typical wells $5K–$50K vs. GPU compute <$10), Active is")
    lines.append("      economically rational despite its computational overhead.")
    lines.append("    - For routine large-scale surveys with abundant well budget,")
    lines.append("      LHS offers a cost-effective alternative achieving most of")
    lines.append("      Active's uncertainty reduction at <1% of the computational cost.")
    lines.append("=" * 88)

    text = "\n".join(lines)
    print("\n" + text)
    path = os.path.join(out_dir, 'cost_performance.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text + "\n")
    print(f"\n  [Saved] {path}")


# ==========================================
# 主函数
# ==========================================
def main(input_dir='./test_100_results',
         output_dir=None,
         n_samples=100,
         start=0,
         methods=('active', 'greedy', 'lhs', 'random'),
         skip_curves=False,
         skip_spatial=False,
         skip_cost_analysis=False,
         cost_per_well_low=5000,
         cost_per_well_high=50000,
         gpu_cost_per_hour=2.0):
    """
    统一评估主入口（性能 + 空间 + 成本）。

    参数
    ----
    input_dir            : run_100.py 的输出目录
    output_dir           : 评估图像/文本输出目录，默认 <input_dir>/evaluation/
    n_samples            : 要加载的样本数量
    start                : 从第几个样本开始扫描
    methods              : 要比较的方法子集
                           合法选项 : 'active', 'greedy', 'lhs', 'random'
    skip_curves          : 跳过学习曲线 + 阶段分析
    skip_spatial         : 跳过空间分布统计
    skip_cost_analysis   : 跳过成本-性能权衡分析
    cost_per_well_low/high : 单口监测井成本区间，用于 ROI 估算
    gpu_cost_per_hour    : GPU 算力单价
    """
    methods = list(methods)
    valid   = {'active', 'greedy', 'lhs', 'random'}
    invalid = [m for m in methods if m not in valid]
    if invalid:
        raise ValueError(f"未知方法 {invalid}，可选 {sorted(valid)}")

    out_dir = output_dir or os.path.join(input_dir, 'evaluation')
    os.makedirs(out_dir, exist_ok=True)

    apply_journal_style()

    print("=" * 70)
    print(f"  evaluate_unified.py  |  input={input_dir}  n={n_samples}")
    print(f"  methods={methods}")
    print("=" * 70)

    # ============================================================
    # Section 1: 性能评估
    # ============================================================
    collected, loaded_idx = load_all_samples(
        input_dir, n_samples=n_samples, start=start, methods=methods)

    if not loaded_idx:
        print("[Error] 未找到任何样本数据，请先运行 run_100.py。")
        return

    print("\n[绘图] fig1: 终态指标箱线图（K_R² / C_R² / Uncertainty）...")
    plot_boxplots(collected, out_dir, methods=methods)

    curves = None
    if not skip_curves:
        print("[绘图] fig2: 学习曲线...")
        curves = load_learning_curves(input_dir, loaded_idx, methods=methods)
        plot_learning_curves(curves, out_dir, methods=methods)

        print("[绘图] fig2_phases: 带三阶段分隔线的学习曲线...")
        plot_learning_curves_with_phases(curves, out_dir, methods=methods)

        print("\n[阶段分析] 三阶段性能统计...")
        phase_stats = compute_phase_stats(curves, methods=methods)
        write_phase_analysis(phase_stats, curves, out_dir, methods=methods)

    print("[绘图] fig3: K_R² vs C_R² 散点图...")
    plot_scatter(collected, out_dir, methods=methods)

    write_summary(collected, out_dir, methods=methods)

    print("\n[统计检验] Wilcoxon 符号秩检验 + Cohen's d ...")
    test_results = run_statistical_tests(collected, methods=methods)
    write_statistical_tests(test_results, out_dir)

    # ============================================================
    # Section 2: 空间分布统计
    # ============================================================
    if not skip_spatial:
        print("\n[空间统计] 计算 6 项空间分布指标 ...")
        spatial_collected, spatial_idx = load_all_spatial(
            input_dir, n_samples=n_samples, methods=methods)
        if spatial_idx:
            save_spatial_csv(spatial_collected, spatial_idx, out_dir,
                             methods=methods)
            spatial_tests = run_spatial_tests(spatial_collected, methods=methods)
            write_spatial_summary(spatial_collected, spatial_tests, out_dir,
                                  methods=methods)
        else:
            print("  [Warning] 无空间统计数据，跳过。")

    # ============================================================
    # Section 3: 成本-性能权衡
    # ============================================================
    if not skip_cost_analysis and curves is not None:
        print("\n[成本分析] well-saving + ROI ...")
        progress_path = os.path.join(input_dir, 'progress.json')
        write_cost_performance_report(
            collected, curves, out_dir,
            progress_path=progress_path,
            cost_per_well_low=cost_per_well_low,
            cost_per_well_high=cost_per_well_high,
            gpu_cost_per_hour=gpu_cost_per_hour,
            methods=methods,
        )

    print(f"\n[Done] 所有图像与报告已保存至 {out_dir}/")


# ==========================================
# 代码启动配置区（直接运行本文件）
# ==========================================
if __name__ == "__main__":
    INPUT_DIR    = './test_100_results'
    OUTPUT_DIR   = None
    N_SAMPLES    = 100
    METHODS      = ['active', 'greedy', 'lhs', 'random']

    SKIP_CURVES        = False
    SKIP_SPATIAL       = False
    SKIP_COST_ANALYSIS = False

    COST_PER_WELL_LOW  = 5000
    COST_PER_WELL_HIGH = 50000
    GPU_COST_PER_HOUR  = 2.0

    main(input_dir=INPUT_DIR,
         output_dir=OUTPUT_DIR,
         n_samples=N_SAMPLES,
         methods=METHODS,
         skip_curves=SKIP_CURVES,
         skip_spatial=SKIP_SPATIAL,
         skip_cost_analysis=SKIP_COST_ANALYSIS,
         cost_per_well_low=COST_PER_WELL_LOW,
         cost_per_well_high=COST_PER_WELL_HIGH,
         gpu_cost_per_hour=GPU_COST_PER_HOUR)
