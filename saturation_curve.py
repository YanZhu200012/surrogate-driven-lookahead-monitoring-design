"""
saturation_curve.py
ES-MDA 饱和曲线脚本（LHS 随机布置，C 不确定性）

实验目的
--------
验证 20 口井作为饱和参考的必要性：随着观测井数量增加，C 集合不确定性
（C_mean_std）呈典型指数衰减，在 20 口井附近出现明显拐点，之后边际增益
迅速趋近于 0，从而为主动选井实验提供性能上限基准。

由于选井策略本身基于 C 不确定性最大化，饱和曲线指标与选井指标保持一致。

指标
----
C_mean_std = sqrt(mean(weighted_var(C_post, axis=ensemble)))
  后验浓度场集合在时空上的加权平均标准差，与 Look-ahead 选井标准一致。

图像布局（双 Y 轴）
------------------
X 轴  ：观测井数量（Number of Wells, 3 → 40）
左 Y  ：C_mean_std（实线 + ±1σ 误差带）
右 Y  ：ΔC_mean_std / ΔWells（边际增益，柱状图）

输出
----
./saturation_results/saturation_data.json   —— 各井数的 C_mean_std 原始数据
./saturation_results/saturation_curve.png   —— 饱和曲线图
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import shared_utils as su


# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
class SatConfig(su.BaseConfig):
    OUTPUT_ROOT = './saturation_results'


# 默认井数序列：3-15 密集采样，20 附近确保有点，之后逐步稀疏
DEFAULT_WELL_COUNTS = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18, 20, 22, 25, 28, 30, 35, 40]
# 默认井数序列：3-15 密集采样，20 附近确保有点，之后逐步稀疏
# DEFAULT_WELL_COUNTS = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 20, 25, 30, 35, 40]

# ──────────────────────────────────────────────
# 绘图样式（与项目其他脚本保持一致）
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# 核心计算
# ──────────────────────────────────────────────
def compute_c_mean_std(n_wells, K_true, H_true, k_train_data,
                       vae, fno_kh, fno_kc, config, prior_seed, lhs_seed):
    """
    对单个测试样本，以 n_wells 口 LHS 随机井运行 ES-MDA，
    返回后验 C 集合的时间加权空间平均标准差（C_mean_std）。
    与 Look-ahead 选井使用的不确定性指标完全一致。

    参数
    ----
    prior_seed : 控制先验集合的随机性（每个测试样本不同）
    lhs_seed   : 控制 LHS 井位的随机性（每种 n_wells 保持一致）
    """
    z_prior = su.generate_prior(vae, k_train_data, config.N_ENSEMBLE,
                                config, seed=prior_seed)
    wells   = su.init_wells_lhs(n_wells, config.GRID_SIZE, lhs_seed)
    h_obs   = su.get_observations(H_true, wells)
    K_post, _ = su.run_esmda(vae, fno_kh, z_prior, wells, h_obs, config)
    C_post    = su.predict_C_batch(K_post, fno_kc, config)
    _, _, _, _, mean_std = su.compute_weighted_C_variance(C_post, config)
    return mean_std                                # scalar


# ──────────────────────────────────────────────
# 主计算循环（支持断点续跑）
# ──────────────────────────────────────────────
def run_saturation(config, well_counts, n_samples,
                   vae, fno_kh, fno_kc, k_train_data, K_all, H_all, C_all,
                   save_path, resume=True):
    """
    对每种井数分别跑 n_samples 个测试场，收集 C_mean_std，
    保存到 JSON 文件中（支持中断后继续）。
    """
    json_path = os.path.join(save_path, 'saturation_data.json')

    if resume and os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        print(f"[Resume] 已加载已有结果 ({json_path})")
    else:
        data = {}

    total_start = time.time()

    for n_wells in well_counts:
        key = str(n_wells)

        if key in data and len(data[key]) >= n_samples:
            print(f"[Skip]  {n_wells:>3} 口井：已有 {len(data[key])} 个样本，跳过。")
            continue

        if key not in data:
            data[key] = []

        start_si = len(data[key])
        lhs_seed = config.SEED * 10000 + n_wells   # 每种井数有固定井位（避免与先验 seed 碰撞）

        print(f"\n[Wells={n_wells:>3}]  "
              f"计算样本 {start_si}~{n_samples - 1} "
              f"（共 {n_samples - start_si} 个）...")
        t0 = time.time()

        for si in range(start_si, n_samples):
            K_true, H_true, _ = su.get_test_sample(K_all, H_all, C_all, si)
            prior_seed = config.SEED * 100000 + si    # 每个样本有独立先验

            val = compute_c_mean_std(
                n_wells, K_true, H_true, k_train_data,
                vae, fno_kh, fno_kc, config, prior_seed, lhs_seed)
            data[key].append(val)

        elapsed = time.time() - t0
        vals = np.array(data[key])
        print(f"  C_mean_std = {vals.mean():.4f} ± {vals.std():.4f}  "
              f"[{elapsed:.1f}s]")

        # 每轮结束保存一次，防止中断丢失
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

    total_min = (time.time() - total_start) / 60
    print(f"\n[Done] 总耗时 {total_min:.1f} min  →  {json_path}")
    return data


# ──────────────────────────────────────────────
# 绘图
# ──────────────────────────────────────────────
def plot_saturation_curve(data, well_counts, save_path):
    """
    双 Y 轴饱和曲线图。
    左轴：C_mean_std 均值曲线 + ±1σ 误差带
    右轴：ΔC_mean_std / ΔWells 边际增益柱状图
    """
    plt.rcParams.update(JOURNAL_RC)

    # ── 整理有效数据点 ──────────────────────────
    xs, mus, stds = [], [], []
    for n in well_counts:
        key = str(n)
        if key not in data or len(data[key]) == 0:
            continue
        vals = np.array(data[key])
        xs.append(n)
        mus.append(float(vals.mean()))
        stds.append(float(vals.std()))

    if len(xs) < 2:
        print("[Warning] 有效数据点不足，无法绘图。")
        return

    xs   = np.array(xs,   dtype=float)
    mus  = np.array(mus,  dtype=float)
    stds = np.array(stds, dtype=float)

    # ── 边际增益 ───────────────────────────────
    # marginals[i] = (mus[i] - mus[i+1]) / (xs[i+1] - xs[i])
    # 正值表示多加一口井带来的不确定性下降量
    delta_mus    = -(np.diff(mus))          # 下降量（期望为正）
    delta_wells  = np.diff(xs)
    marginals    = delta_mus / delta_wells  # ΔC_mean_std / ΔWells
    marg_x       = (xs[:-1] + xs[1:]) / 2  # 柱子中心

    # ── 画布 ──────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()

    color_main = '#1f77b4'   # 蓝色：主曲线
    color_bar  = '#d62728'   # 红色：边际增益

    # 左轴：误差带 + 实线
    ax1.fill_between(xs, mus - stds, mus + stds,
                     color=color_main, alpha=0.18, zorder=1,
                     label='±1 std band')
    ax1.plot(xs, mus, color=color_main, linewidth=2.0, zorder=3,
             marker='o', markersize=4.5, label='C mean std (mean)')

    # 作为上限参考（20 口井）
    elbow = 20
    if elbow in xs:
        ax1.axvline(x=elbow, color='gray', linestyle='--',
                    linewidth=0.9, alpha=0.8, zorder=2)

    # 右轴：边际增益柱状图（宽度自适应步长）
    bar_width = np.minimum(delta_wells * 0.55, 1.6)
    ax2.bar(marg_x, marginals, width=bar_width,
            color=color_bar, alpha=0.40, zorder=0,
            label='ΔC mean std / ΔWells')
    # 零参考线
    ax2.axhline(0, color=color_bar, linewidth=0.6, alpha=0.5)
    ax2.set_ylim(bottom=0)

    # ── 轴标签 ────────────────────────────────
    ax1.set_xlabel('Number of Wells',
                   fontsize=JOURNAL_RC['axes.labelsize'])
    ax1.set_ylabel('C mean std  (C uncertainty)',
                   color=color_main,
                   fontsize=JOURNAL_RC['axes.labelsize'])
    ax1.tick_params(axis='y', labelcolor=color_main)

    ax2.set_ylabel('ΔC mean std / ΔWells  (marginal gain)',
                   color=color_bar,
                   fontsize=JOURNAL_RC['axes.labelsize'])
    ax2.tick_params(axis='y', labelcolor=color_bar)

    # x 轴刻度：仅显示整数
    ax1.set_xticks(xs.astype(int))
    ax1.set_xticklabels([str(int(x)) for x in xs],
                        rotation=45, ha='right',
                        fontsize=JOURNAL_RC['xtick.labelsize'])
    ax1.set_xlim(xs[0] - 1, xs[-1] + 1)

    # 网格仅左轴
    ax1.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)

    # ── 图例合并 ──────────────────────────────
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               loc='upper right',
               fontsize=JOURNAL_RC['legend.fontsize'])

    fig.suptitle(
        'ES-MDA Saturation Curve: C Uncertainty vs. Number of Wells\n'
        '(LHS random well placement)',
        fontsize=JOURNAL_RC['figure.titlesize'])
    fig.tight_layout()

    out_path = os.path.join(save_path, 'saturation_curve.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {out_path}")


# ──────────────────────────────────────────────
# 打印汇总表
# ──────────────────────────────────────────────
def print_summary(data, well_counts):
    print("\n" + "=" * 58)
    print(f"{'Wells':>7}  {'C_mean_std mean':>16}  {'std':>8}  {'N':>5}")
    print("-" * 58)
    for n in well_counts:
        key = str(n)
        if key not in data or len(data[key]) == 0:
            continue
        vals = np.array(data[key])
        print(f"  {n:>5}    {vals.mean():>14.4f}   {vals.std():>8.4f}  "
              f"{len(vals):>5}")
    print("=" * 58)


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────
def main():
    # =========================================================================
    # 用户自定义参数区 (修改这里的参数即可控制程序运行)
    # =========================================================================
    OUTPUT_DIR   = './saturation_results' # 输出目录
    N_SAMPLES    = 30                   # 用于平均的测试样本数
    WELL_COUNTS  = DEFAULT_WELL_COUNTS    # 要测试的井数列表 (如: [3, 5, 10, 20])
    RESUME       = True                   # 是否断点续跑 (True/False)
    PLOT_ONLY    = False                  # 是否仅从已有 JSON 重新绘图，不重算 (True/False)
    # =========================================================================

    well_counts = sorted(set(WELL_COUNTS))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  ES-MDA Saturation Curve  (C uncertainty)")
    print(f"  wells      = {well_counts}")
    print(f"  n_samples  = {N_SAMPLES}")
    print(f"  output     = {OUTPUT_DIR}")
    print("=" * 70)

    config = SatConfig()
    config.OUTPUT_ROOT = OUTPUT_DIR

    json_path = os.path.join(OUTPUT_DIR, 'saturation_data.json')

    if PLOT_ONLY:
        if not os.path.exists(json_path):
            print(f"[Error] 找不到 {json_path}，请先将 PLOT_ONLY 设为 False 运行计算。")
            return
        with open(json_path) as f:
            data = json.load(f)
        print(f"[Info] 已加载 {json_path}")
    else:
        vae, fno_kh, fno_kc = su.load_all_models(config)
        K_all, H_all, C_all, k_train_data = su.load_test_data(config)

        n_avail = K_all.shape[0]
        if N_SAMPLES > n_avail:
            print(f"[Warning] n_samples={N_SAMPLES} > 可用样本数 {n_avail}，"
                  f"已截断为 {n_avail}。")
            N_SAMPLES = n_avail

        data = run_saturation(
            config, well_counts, N_SAMPLES,
            vae, fno_kh, fno_kc, k_train_data, K_all, H_all, C_all,
            OUTPUT_DIR, resume=RESUME)

    print_summary(data, well_counts)

    print("\n[绘图] 生成饱和曲线图...")
    plot_saturation_curve(data, well_counts, OUTPUT_DIR)

    print(f"\n[Done] 所有输出已保存至 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()