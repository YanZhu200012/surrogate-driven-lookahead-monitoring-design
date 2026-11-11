"""
evaluate_es_mda.py
ES-MDA 独立验证实验评估脚本

从 run_es_mda.py 保存的结果中加载数据，
比较先验、后验-3井、后验-20井三个阶段的 **K（水力传导系数）** 预测质量，
从而单独验证 ES-MDA 同化算法本身的有效性（与选井策略解耦）。

评估指标（均基于 K）：
  K R²       —— K 均值场与真实 K 场的决定系数
  K RMSE     —— K 均值场与真实 K 场的均方根误差
  K mean std —— K 集合在空间上的平均标准差（不确定性度量）

输出：./es_mda_results/evaluation/
  fig1_metrics_boxplot.png      —— K_R2 / K_RMSE / K_mean_std 箱线图
  fig2_variance_reduction.png   —— K 不确定性缩减柱状图
  fig3_field_{low,medium,high}.png —— 代表性样本场对比（先验 / 后验-3井 / 后验-20井）
  summary.txt                   —— 均值±标准差汇总表
  statistical_tests.txt         —— 单样本 / 配对 t 检验与 Cohen's d
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score
from scipy import stats

# ==========================================
# 全局绘图样式
# ==========================================
JOURNAL_RC = {
    'font.family':       'Arial',
    'font.size':         12,
    'axes.titlesize':    13,
    'axes.labelsize':    12,
    'xtick.labelsize':   10,
    'ytick.labelsize':   11,
    'legend.fontsize':   11,
    'figure.titlesize':  14,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
}
SAVE_KW = dict(dpi=300, bbox_inches='tight')
CBAR_KW = dict(fraction=0.046, pad=0.04)

COLORS = {
    'prior':   '#7f7f7f',   # 灰色
    'post3':   '#1f77b4',   # 蓝色
    'post20':  '#d62728',   # 红色
}
LABELS = {
    'prior':  'Prior',
    'post3':  'Posterior (3 wells)',
    'post20': 'Posterior (20 wells)',
}


# ==========================================
# 辅助函数
# ==========================================
def apply_journal_style():
    plt.rcParams.update(JOURNAL_RC)


def add_colorbar(fig, im, ax):
    """统一 colorbar，无单位标签"""
    cbar = fig.colorbar(im, ax=ax, **CBAR_KW)
    cbar.ax.tick_params(labelsize=JOURNAL_RC['xtick.labelsize'])
    return cbar


def _pick_samples_by_metric(results, metric='post20_K_mean_std'):
    """
    按指标从小到大均匀挑选 low / medium / high 三个代表性样本。
    """
    vals       = results[metric]
    loaded     = np.array(results['loaded_idx'])
    idx_sorted = np.argsort(vals)
    n          = len(idx_sorted)
    if n < 3:
        raise ValueError(
            f"至少需要 3 个样本才能选取 low/medium/high，当前仅有 {n} 个。")
    picks = {
        'low':    int(loaded[idx_sorted[2]]),
        'medium': int(loaded[idx_sorted[n // 2]]),
        'high':   int(loaded[idx_sorted[-2]]),
    }
    print(f"  [Field] 按 {metric} 选中样本 -> "
          f"Low (#{picks['low']}, val={vals[idx_sorted[2]]:.4f})  "
          f"Medium (#{picks['medium']}, val={vals[idx_sorted[n // 2]]:.4f})  "
          f"High (#{picks['high']}, val={vals[idx_sorted[-2]]:.4f})")
    return picks


# ==========================================
# 数据加载
# ==========================================
def load_results(output_root, n_samples=100):
    """加载所有样本的结果，返回按条件组织的字典。

    K 不确定性（K_mean_std）由 K_std 场在空间上取均值得到；
    先验 K R² / RMSE 由 prior_K_mean 场与 K_true 计算得到。
    """
    scalar_keys = [
        'post3_K_r2',  'post3_K_rmse',
        'post20_K_r2', 'post20_K_rmse',
    ]
    derived_keys = [
        'prior_K_r2',   'prior_K_rmse',
        'prior_K_mean_std',
        'post3_K_mean_std',
        'post20_K_mean_std',
    ]
    all_keys = scalar_keys + derived_keys
    results = {k: [] for k in all_keys}
    results['loaded_idx'] = []

    for si in range(n_samples):
        path = os.path.join(output_root, f'sample_{si:03d}.npz')
        if not os.path.exists(path):
            continue
        data = np.load(path, allow_pickle=True)

        # 直接保存的 K 指标
        for k in scalar_keys:
            results[k].append(float(data[k]) if k in data else np.nan)

        # 先验 K R² / RMSE（从场数据计算）
        if 'prior_K_mean' in data and 'K_true' in data:
            k_mean = data['prior_K_mean'].flatten()
            k_true = data['K_true'].flatten()
            results['prior_K_r2'].append(
                float(r2_score(k_true, k_mean)))
            results['prior_K_rmse'].append(
                float(np.sqrt(np.mean((k_mean - k_true) ** 2))))
        else:
            results['prior_K_r2'].append(np.nan)
            results['prior_K_rmse'].append(np.nan)

        # K mean std：K 集合标准差场的空间均值
        for prefix, key in [('prior',  'prior_K_mean_std'),
                             ('post3',  'post3_K_mean_std'),
                             ('post20', 'post20_K_mean_std')]:
            std_key = f'{prefix}_K_std'
            results[key].append(
                float(np.mean(data[std_key])) if std_key in data else np.nan)

        results['loaded_idx'].append(si)

    for k in all_keys:
        results[k] = np.array(results[k], dtype=np.float32)

    n = len(results['loaded_idx'])
    print(f"[Info] 已加载 {n} 个样本结果（共请求 {n_samples} 个）")
    return results


def load_sample_fields(output_root, sample_idx):
    """加载单个样本的场数据，用于可视化。"""
    path = os.path.join(output_root, f'sample_{sample_idx:03d}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")
    return np.load(path, allow_pickle=True)


# ==========================================
# 图 1：K_R2 / K_RMSE / K_mean_std 箱线图
# ==========================================
def plot_metrics_boxplot(results, out_dir):
    plt.rcParams.update(JOURNAL_RC)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    metrics = [
        ('K R²', [results['prior_K_r2'],
                  results['post3_K_r2'],
                  results['post20_K_r2']]),
        ('K RMSE', [results['prior_K_rmse'],
                    results['post3_K_rmse'],
                    results['post20_K_rmse']]),
        ('K mean std', [results['prior_K_mean_std'],
                        results['post3_K_mean_std'],
                        results['post20_K_mean_std']]),
    ]
    conditions = ['prior', 'post3', 'post20']

    for i, (ax, (ylabel, data_list)) in enumerate(zip(axes, metrics)):
        # 过滤掉含 NaN 的条件
        plot_data = []
        plot_labels = []
        plot_colors = []
        for c, d in zip(conditions, data_list):
            valid = d[~np.isnan(d)]
            if len(valid) == 0:
                continue
            plot_data.append(valid)
            plot_labels.append(LABELS[c])
            plot_colors.append(COLORS[c])

        if not plot_data:
            ax.axis('off')
            continue

        bp = ax.boxplot(plot_data, patch_artist=True, widths=0.5,
                        medianprops=dict(color='black', linewidth=1.5))
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # 隐藏 X 轴刻度，以图例区分阶段
        ax.set_xticks([])
        ax.set_ylabel(ylabel)

        # 图例只在第一个子图中显示
        if i == 0 and plot_labels:
            ax.legend(bp['boxes'], plot_labels,
                      fontsize=JOURNAL_RC.get('legend.fontsize', 11),
                      loc='lower right', framealpha=0.9)

    fig.suptitle('ES-MDA Effectiveness: Prior vs. Posterior\n'
                 '(3-well & 20-well, LHS random placement)',
                 fontsize=JOURNAL_RC['figure.titlesize'], y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig1_metrics_boxplot.png')
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    print(f"  [Saved] {path}")

# ==========================================
# 图 2：K 不确定性缩减柱状图
# ==========================================
def plot_variance_reduction(results, out_dir):
    plt.rcParams.update(JOURNAL_RC)
    fig, ax = plt.subplots(figsize=(6, 4.5))

    conditions = ['prior', 'post3', 'post20']
    means = [
        np.nanmean(results['prior_K_mean_std']),
        np.nanmean(results['post3_K_mean_std']),
        np.nanmean(results['post20_K_mean_std']),
    ]
    stds = [
        np.nanstd(results['prior_K_mean_std']),
        np.nanstd(results['post3_K_mean_std']),
        np.nanstd(results['post20_K_mean_std']),
    ]
    colors = [COLORS[c] for c in conditions]
    labels = [LABELS[c] for c in conditions]

    x = np.arange(len(conditions))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.8, width=0.5,
                  error_kw=dict(elinewidth=1.2, ecolor='black'))

    # 在柱顶标注均值
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.0005,
                f'{mean:.4f}', ha='center', va='bottom',
                fontsize=JOURNAL_RC['xtick.labelsize'])

    # 标注缩减百分比（相对于先验）
    prior_mean = means[0]
    for i, (mean, label) in enumerate(zip(means[1:], ['3w', '20w']), start=1):
        if prior_mean > 0:
            reduction = (1 - mean / prior_mean) * 100
            ax.annotate(f'↓{reduction:.1f}%',
                        xy=(i, mean / 2), ha='center', va='center',
                        fontsize=JOURNAL_RC['legend.fontsize'],
                        color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('K mean std (uncertainty)')
    ax.set_title('K Uncertainty Reduction by ES-MDA\n'
                 '(LHS random well placement)')
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join(out_dir, 'fig2_variance_reduction.png')
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    print(f"  [Saved] {path}")


# ==========================================
# 图 3：代表性样本场对比
# ==========================================
def plot_field_comparison(output_root, out_dir, results=None, sample_indices=None):
    """
    代表性样本场对比图。

    样本选取策略（优先级由高到低）：
      1. sample_indices 手动指定→ 转为 {'sample_N': N} 字典
      2. results 提供时按 post20_K_mean_std 排序，取 low/medium/high 三组
      3. 默认样本 0/1/2

    布局：4 行 × 4 列（GridSpec），每个分组输出一张图
      行标题（set_ylabel）：Ground Truth / Prior / Posterior 3-well / Posterior 20-well  [仅第 0 列]
      列标题（set_title） ：K mean / K std / C mean (t=5) / C std (t=5)                 [仅第 0 行]
      无评估指标、无单位、无 abc 标签
    输出文件：fig3_field_low.png / fig3_field_medium.png / fig3_field_high.png
    """
    # ── 选样 ─────────────────────────────────────────────
    if sample_indices is not None:
        groups = {f'sample_{si}': si for si in sample_indices}
    elif results is not None:
        groups = _pick_samples_by_metric(results)
    else:
        groups = {'sample_0': 0, 'sample_1': 1, 'sample_2': 2}

    apply_journal_style()

    n_rows, n_cols = 4, 4
    col_titles = ['K mean', 'K std', 'C mean (t=5)', 'C std (t=5)']   # C 取第 5 个时步（索引 4）
    row_labels  = ['Ground Truth', 'Prior',
                   'Posterior 3-well', 'Posterior 20-well']

    for group_name, si in groups.items():
        try:
            data = load_sample_fields(output_root, si)
        except FileNotFoundError as e:
            print(f"  [Warning] {e}，跳过样本 {si}")
            continue

        K_true = data['K_true']
        C_true = data['C_true']   # (5, 51, 51)

        fig = plt.figure(figsize=(n_cols * 3.5, n_rows * 3.0))
        gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                                hspace=0.12, wspace=0.30)

        # ── 行 0：Ground Truth ────────────────────────
        gt_arrays = [
            (K_true,                    'viridis'),
            (None,                      'viridis'),   # K std — N/A
            (C_true[4],                 'turbo'),
            (None,                      'turbo'),     # C std — N/A
        ]
        for j, (arr, cmap) in enumerate(gt_arrays):
            ax = fig.add_subplot(gs[0, j])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(col_titles[j],
                         fontsize=JOURNAL_RC['axes.titlesize'], pad=8)
            if j == 0:
                ax.set_ylabel(row_labels[0],
                              fontsize=JOURNAL_RC['axes.labelsize'],
                              labelpad=10)
            if arr is None:
                ax.axis('off')
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        ha='center', va='center',
                        fontsize=JOURNAL_RC['axes.labelsize'], color='gray')
            else:
                im = ax.imshow(arr, cmap=cmap, origin='lower', aspect='equal')
                add_colorbar(fig, im, ax)

        # ── 行 1–3：Prior / Post-3well / Post-20well ──
        phase_rows = [
            (1, row_labels[1],
             'prior_K_mean',  'prior_K_std',  'prior_C_mean',  'prior_C_std',  None),
            (2, row_labels[2],
             'post3_K_mean',  'post3_K_std',  'post3_C_mean',  'post3_C_std',  'wells_3'),
            (3, row_labels[3],
             'post20_K_mean', 'post20_K_std', 'post20_C_mean', 'post20_C_std', 'wells_20'),
        ]
        for row_idx, row_label, km, ks, cm, cs, wk in phase_rows:
            arrays = [
                (data[km],      'viridis'),
                (data[ks],      'viridis'),
                (data[cm][4],   'turbo'),
                (data[cs][4],   'turbo'),
            ]
            ax_list = []
            for j, (arr, cmap) in enumerate(arrays):
                ax = fig.add_subplot(gs[row_idx, j])
                im = ax.imshow(arr, cmap=cmap, origin='lower', aspect='equal')
                ax.set_xticks([]); ax.set_yticks([])
                add_colorbar(fig, im, ax)
                if j == 0:
                    ax.set_ylabel(row_label,
                                  fontsize=JOURNAL_RC['axes.labelsize'],
                                  labelpad=10)
                ax_list.append(ax)
            # 标注井位（仅 K mean 列，第 0 列）
            if wk and wk in data:
                wells_arr = data[wk]   # (n, 2)
                ax_list[0].scatter(
                    wells_arr[:, 1], wells_arr[:, 0],
                    c='red', s=20, marker='x', linewidths=1.0, zorder=5)

        path = os.path.join(out_dir, f'fig3_field_{group_name}.png')
        plt.savefig(path, **SAVE_KW)
        plt.close(fig)
        print(f"  [Saved] {path}")


# ==========================================
# 统计显著性检验
# ==========================================
def run_statistical_tests(results):
    """
    针对 ES-MDA 同化有效性进行统计检验，返回结构化结果字典。

    检验设计说明
    ─────────────────────────────────────────────────────────────────
    ① 单样本 t 检验（one-sample t-test, H₀: μ = 0）
       对象：prior_K_r2、post3_K_r2、post20_K_r2
       目的：正式检验各阶段 R² 是否显著不同于零
             （先验 R²<0 可证明集合初始高度离散；
              3 井后验 R²>0 可证明极少观测已有统计意义的信息提取）

    ② 配对 t 检验（paired t-test）
       对象：阶段间的 K_R² 差值（3w-prior, 20w-prior, 20w-3w）、
             K_mean_std 差值（不确定性缩减量）与 K_RMSE 差值
       目的：检验增加观测井带来的改善是否在统计上显著
             每个样本同时经历所有阶段，自然构成配对结构

    ③ Cohen's d 效应量
       计算配对差值的效应量，量化"改善幅度在实际意义上有多大"
       惯用标准：|d| < 0.2 小效应，0.2–0.8 中效应，> 0.8 大效应
    ─────────────────────────────────────────────────────────────────
    """

    def _valid_pair(a, b):
        """去除任意一方为 NaN 的配对，返回两个等长的干净数组。"""
        mask = ~(np.isnan(a) | np.isnan(b))
        return a[mask], b[mask]

    def _valid(a):
        return a[~np.isnan(a)]

    def _cohens_d_paired(diff):
        """配对差值的 Cohen's d = mean(diff) / std(diff, ddof=1)"""
        if len(diff) < 2:
            return np.nan
        return np.mean(diff) / np.std(diff, ddof=1)

    out = {}

    # ── ① 单样本 t 检验：各阶段 K_R² vs 0 ──────────────────────
    for stage, key in [('prior',  'prior_K_r2'),
                       ('post3',  'post3_K_r2'),
                       ('post20', 'post20_K_r2')]:
        arr = _valid(results[key])
        if len(arr) < 2:
            out[f'onesample_{stage}'] = dict(n=len(arr), mean=np.nan, std=np.nan,
                                              t=np.nan, p=np.nan, h0='μ=0')
            continue
        t_stat, p_two = stats.ttest_1samp(arr, popmean=0.0)
        # 双侧转单侧：prior 检验 μ<0，其余检验 μ>0
        p_one = p_two / 2 if ((stage == 'prior' and t_stat < 0) or
                               (stage != 'prior' and t_stat > 0)) else 1 - p_two / 2
        out[f'onesample_{stage}'] = dict(
            n    = len(arr),
            mean = float(np.mean(arr)),
            std  = float(np.std(arr, ddof=1)),
            t    = float(t_stat),
            p_two= float(p_two),
            p_one= float(p_one),
            h0   = 'μ=0'
        )

    # ── ② 配对 t 检验：K_R² 改善量 ─────────────────────────────
    pairs_r2 = [
        ('3w_vs_prior',  'post3_K_r2',  'prior_K_r2',  '3-well > Prior'),
        ('20w_vs_prior', 'post20_K_r2', 'prior_K_r2',  '20-well > Prior'),
        ('20w_vs_3w',    'post20_K_r2', 'post3_K_r2',  '20-well > 3-well'),
    ]
    for label, key_a, key_b, description in pairs_r2:
        a, b = _valid_pair(results[key_a], results[key_b])
        diff = a - b
        if len(diff) < 2:
            out[f'paired_r2_{label}'] = dict(n=len(diff), mean_diff=np.nan,
                                              t=np.nan, p=np.nan, d=np.nan)
            continue
        t_stat, p_two = stats.ttest_rel(a, b)
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2  # 单侧：改善方向
        out[f'paired_r2_{label}'] = dict(
            n         = len(diff),
            mean_diff = float(np.mean(diff)),
            std_diff  = float(np.std(diff, ddof=1)),
            t         = float(t_stat),
            p_two     = float(p_two),
            p_one     = float(p_one),
            d         = float(_cohens_d_paired(diff)),
            description = description
        )

    # ── ② 配对 t 检验：K_mean_std 不确定性缩减量 ────────────────
    # 注意方向：不确定性下降为"改善"，所以 diff = prior - post（正值为好）
    pairs_std = [
        ('3w_vs_prior',  'prior_K_mean_std', 'post3_K_mean_std',  'Uncertainty↓ 3-well'),
        ('20w_vs_prior', 'prior_K_mean_std', 'post20_K_mean_std', 'Uncertainty↓ 20-well'),
        ('20w_vs_3w',    'post3_K_mean_std', 'post20_K_mean_std', 'Uncertainty↓ 20w vs 3w'),
    ]
    for label, key_a, key_b, description in pairs_std:
        a, b = _valid_pair(results[key_a], results[key_b])
        diff = a - b   # 正值 = 不确定性降低
        if len(diff) < 2:
            out[f'paired_std_{label}'] = dict(n=len(diff), mean_diff=np.nan,
                                               t=np.nan, p=np.nan, d=np.nan)
            continue
        t_stat, p_two = stats.ttest_rel(a, b)
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        out[f'paired_std_{label}'] = dict(
            n         = len(diff),
            mean_diff = float(np.mean(diff)),
            std_diff  = float(np.std(diff, ddof=1)),
            t         = float(t_stat),
            p_two     = float(p_two),
            p_one     = float(p_one),
            d         = float(_cohens_d_paired(diff)),
            description = description
        )

    # ── ② 配对 t 检验：K_RMSE 改善量 ──────────────────────────
    # 注意方向：RMSE 下降为"改善"，所以 diff = prior - post（正值为好）
    pairs_rmse = [
        ('3w_vs_prior',  'prior_K_rmse', 'post3_K_rmse',  'RMSE↓ 3-well'),
        ('20w_vs_prior', 'prior_K_rmse', 'post20_K_rmse', 'RMSE↓ 20-well'),
        ('20w_vs_3w',    'post3_K_rmse', 'post20_K_rmse', 'RMSE↓ 20w vs 3w'),
    ]
    for label, key_a, key_b, description in pairs_rmse:
        a, b = _valid_pair(results[key_a], results[key_b])
        diff = a - b   # 正值 = RMSE 降低
        if len(diff) < 2:
            out[f'paired_rmse_{label}'] = dict(n=len(diff), mean_diff=np.nan,
                                                t=np.nan, p=np.nan, d=np.nan)
            continue
        t_stat, p_two = stats.ttest_rel(a, b)
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        out[f'paired_rmse_{label}'] = dict(
            n         = len(diff),
            mean_diff = float(np.mean(diff)),
            std_diff  = float(np.std(diff, ddof=1)),
            t         = float(t_stat),
            p_two     = float(p_two),
            p_one     = float(p_one),
            d         = float(_cohens_d_paired(diff)),
            description = description
        )

    return out


def _format_p(p):
    """将 p 值格式化为期刊常用写法。"""
    if np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _sig_label(p):
    """显著性星号标注。"""
    if np.isnan(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def write_statistical_tests(test_results, out_dir, n_samples):
    """将统计检验结果格式化输出到终端并保存为 statistical_tests.txt。"""

    lines = []
    lines.append("=" * 80)
    lines.append("  Statistical Significance Tests for ES-MDA Effectiveness")
    lines.append(f"  (based on {n_samples} independent test samples)")
    lines.append("=" * 80)

    # ── ① 单样本 t 检验 ──────────────────────────────────────────
    lines.append("")
    lines.append("─" * 80)
    lines.append("① One-Sample t-Test: K_R² vs. zero  (H₀: μ = 0)")
    lines.append("   Rationale: Formally test whether each stage's R² is")
    lines.append("   significantly different from zero (i.e., better than")
    lines.append("   predicting the global mean).  Prior R² < 0 confirms")
    lines.append("   the ensemble is initially dispersed; Posterior R² > 0")
    lines.append("   confirms that assimilation extracts meaningful information.")
    lines.append("─" * 80)
    lines.append(f"  {'Stage':<22} {'mean R²':>10} {'std':>8} {'t':>8} "
                 f"{'p (one-sided)':>16} {'Sig.':>6}")
    lines.append("  " + "-" * 72)
    for stage, key in [('Prior',           'onesample_prior'),
                       ('Posterior 3-well', 'onesample_post3'),
                       ('Posterior 20-well','onesample_post20')]:
        r = test_results.get(key, {})
        mean = r.get('mean', np.nan)
        std  = r.get('std',  np.nan)
        t    = r.get('t',    np.nan)
        p    = r.get('p_one', np.nan)
        sig  = _sig_label(p)
        lines.append(f"  {stage:<22} {mean:>10.4f} {std:>8.4f} {t:>8.3f} "
                     f"{_format_p(p):>16} {sig:>6}")
    lines.append("  * p < 0.05,  ** p < 0.01,  *** p < 0.001  (one-sided)")

    # ── ② 配对 t 检验：K_R² ──────────────────────────────────────
    lines.append("")
    lines.append("─" * 80)
    lines.append("② Paired t-Test: K_R² improvement between stages")
    lines.append("   Rationale: Each sample undergoes all three stages,")
    lines.append("   forming a natural paired structure.  Tests whether")
    lines.append("   adding wells produces a statistically significant R² gain.")
    lines.append("─" * 80)
    lines.append(f"  {'Comparison':<28} {'Δmean R²':>10} {'t':>8} " 
                 f"{'p (one-sided)':>16} {'Cohens_d':>10} {'Sig.':>6}")
    lines.append("  " + "-" * 80)
    for label, key in [('3-well  vs Prior',  'paired_r2_3w_vs_prior'),
                       ('20-well vs Prior',  'paired_r2_20w_vs_prior'),
                       ('20-well vs 3-well', 'paired_r2_20w_vs_3w')]:
        r   = test_results.get(key, {})
        dm  = r.get('mean_diff', np.nan)
        t   = r.get('t',         np.nan)
        p   = r.get('p_one',     np.nan)
        d   = r.get('d',         np.nan)
        sig = _sig_label(p)
        d_str = f"{d:.3f}" if not np.isnan(d) else "N/A"
        lines.append(f"  {label:<28} {dm:>10.4f} {t:>8.3f} "
                     f"{_format_p(p):>16} {d_str:>10} {sig:>6}")

    # ── ③ 配对 t 检验：K_mean_std ────────────────────────────────
    lines.append("")
    lines.append("─" * 80)
    lines.append("③ Paired t-Test: K uncertainty (K_mean_std) reduction")
    lines.append("   Δ = earlier_stage - later_stage  (positive = reduction)")
    lines.append("─" * 80)
    lines.append(f"  {'Comparison':<28} {'Δmean std':>10} {'t':>8} "
                 f"{'p (one-sided)':>16} {'Cohens_d':>10} {'Sig.':>6}")
    lines.append("  " + "-" * 80)
    for label, key in [('3-well  vs Prior',  'paired_std_3w_vs_prior'),
                       ('20-well vs Prior',  'paired_std_20w_vs_prior'),
                       ('20-well vs 3-well', 'paired_std_20w_vs_3w')]:
        r   = test_results.get(key, {})
        dm  = r.get('mean_diff', np.nan)
        t   = r.get('t',         np.nan)
        p   = r.get('p_one',     np.nan)
        d   = r.get('d',         np.nan)
        sig = _sig_label(p)
        d_str = f"{d:.3f}" if not np.isnan(d) else "N/A"
        lines.append(f"  {label:<28} {dm:>10.4f} {t:>8.3f} "
                     f"{_format_p(p):>16} {d_str:>10} {sig:>6}")

    # ── ④ 配对 t 检验：K_RMSE ────────────────────────────────────
    lines.append("")
    lines.append("─" * 80)
    lines.append("④ Paired t-Test: K_RMSE reduction between stages")
    lines.append("   Δ = earlier_stage - later_stage  (positive = reduction)")
    lines.append("─" * 80)
    lines.append(f"  {'Comparison':<28} {'Δmean RMSE':>10} {'t':>8} "
                 f"{'p (one-sided)':>16} {'Cohens_d':>10} {'Sig.':>6}")
    lines.append("  " + "-" * 80)
    for label, key in [('3-well  vs Prior',  'paired_rmse_3w_vs_prior'),
                       ('20-well vs Prior',  'paired_rmse_20w_vs_prior'),
                       ('20-well vs 3-well', 'paired_rmse_20w_vs_3w')]:
        r   = test_results.get(key, {})
        dm  = r.get('mean_diff', np.nan)
        t   = r.get('t',         np.nan)
        p   = r.get('p_one',     np.nan)
        d   = r.get('d',         np.nan)
        sig = _sig_label(p)
        d_str = f"{d:.3f}" if not np.isnan(d) else "N/A"
        lines.append(f"  {label:<28} {dm:>10.4f} {t:>8.3f} "
                     f"{_format_p(p):>16} {d_str:>10} {sig:>6}")

    lines.append("")
    lines.append("  Effect size interpretation (Cohen's d for paired differences):")
    lines.append("    |d| < 0.2: negligible,  0.2–0.5: small,  0.5–0.8: medium,  > 0.8: large")
    lines.append("=" * 80)

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
def write_summary(results, out_dir):
    n = len(results['loaded_idx'])
    lines = [
        "ES-MDA Effectiveness Summary (K metrics)",
        f"Number of test samples: {n}",
        "=" * 65,
        f"{'Condition':<22} {'K_mean_std (mean±std)':>26} {'K_R2 (mean±std)':>22} {'K_RMSE (mean±std)':>22}",
        "-" * 96,
    ]

    def fmt_arr(arr):
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return "—"
        return f"{np.mean(valid):.4f} ± {np.std(valid):.4f}"

    def row(label, mstd_arr, r2_arr=None, rmse_arr=None):
        mstd_str = fmt_arr(mstd_arr)
        r2_str   = fmt_arr(r2_arr) if r2_arr is not None else "—"
        rmse_str = fmt_arr(rmse_arr) if rmse_arr is not None else "—"
        return f"{label:<22} {mstd_str:>26} {r2_str:>22} {rmse_str:>22}"

    lines.append(row('Prior',
                     results['prior_K_mean_std'],  results['prior_K_r2'],
                     results['prior_K_rmse']))
    lines.append(row('Posterior (3 wells)',
                     results['post3_K_mean_std'],  results['post3_K_r2'],
                     results['post3_K_rmse']))
    lines.append(row('Posterior (20 wells)',
                     results['post20_K_mean_std'], results['post20_K_r2'],
                     results['post20_K_rmse']))
    lines.append("-" * 96)

    # 不确定性缩减率（K mean std）
    prior_mean  = np.nanmean(results['prior_K_mean_std'])
    post3_mean  = np.nanmean(results['post3_K_mean_std'])
    post20_mean = np.nanmean(results['post20_K_mean_std'])
    if prior_mean > 0:
        lines.append(f"K uncertainty reduction (3w vs prior):  "
                     f"{(1 - post3_mean  / prior_mean)*100:.1f}%")
        lines.append(f"K uncertainty reduction (20w vs prior): "
                     f"{(1 - post20_mean / prior_mean)*100:.1f}%")
    if post3_mean > 0:
        lines.append(f"K uncertainty reduction (20w vs 3w):    "
                     f"{(1 - post20_mean / post3_mean)*100:.1f}%")

    # RMSE 缩减率
    prior_rmse  = np.nanmean(results['prior_K_rmse'])
    post3_rmse  = np.nanmean(results['post3_K_rmse'])
    post20_rmse = np.nanmean(results['post20_K_rmse'])
    if prior_rmse > 0:
        lines.append(f"K RMSE reduction (3w vs prior):         "
                     f"{(1 - post3_rmse  / prior_rmse)*100:.1f}%")
        lines.append(f"K RMSE reduction (20w vs prior):        "
                     f"{(1 - post20_rmse / prior_rmse)*100:.1f}%")
    if post3_rmse > 0:
        lines.append(f"K RMSE reduction (20w vs 3w):           "
                     f"{(1 - post20_rmse / post3_rmse)*100:.1f}%")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)
    path = os.path.join(out_dir, 'summary.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(summary_text + "\n")
    print(f"\n  [Saved] {path}")


# ==========================================
# 主函数
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description='Evaluate ES-MDA results (prior vs. posterior comparison)')
    parser.add_argument('--input',   default='./es_mda_results',
                        help='run_es_mda.py 的输出目录')
    parser.add_argument('--output',  default=None,
                        help='图像输出目录（默认 <input>/evaluation/）')
    parser.add_argument('--n',       type=int, default=100,
                        help='要加载的样本总数')
    parser.add_argument('--vis-samples', nargs='+', type=int,
                        default=None,
                        help='用于场可视化的样本编号（默认 None，此时按 post20_K_mean_std 自动选 low/medium/high）')
    parser.add_argument('--no-fields', action='store_true',
                        help='跳过场对比图（节省时间）')
    args = parser.parse_args()

    out_dir = args.output or os.path.join(args.input, 'evaluation')
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update(JOURNAL_RC)

    print("=" * 70)
    print(f"  ES-MDA Evaluation  |  input={args.input}  output={out_dir}")
    print("=" * 70)

    results = load_results(args.input, n_samples=args.n)
    if len(results['loaded_idx']) == 0:
        print("[Error] 未找到任何样本结果，请先运行 run_es_mda.py。")
        return

    write_summary(results, out_dir)

    print("\n[统计检验] 显著性检验...")
    test_results = run_statistical_tests(results)
    write_statistical_tests(test_results, out_dir, len(results['loaded_idx']))

    print("\n[绘图] fig1: 指标箱线图...")
    plot_metrics_boxplot(results, out_dir)

    print("[绘图] fig2: 不确定性缩减柱状图...")
    plot_variance_reduction(results, out_dir)

    if not args.no_fields:
        print("[绘图] fig3: 代表性样本场对比...")
        plot_field_comparison(args.input, out_dir,
                              results=results,
                              sample_indices=args.vis_samples)

    print(f"\n[Done] 所有图像已保存至 {out_dir}/")


if __name__ == "__main__":
    main()