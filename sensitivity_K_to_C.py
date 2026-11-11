"""
sensitivity_K_to_C.py
K→C 敏感性分析（正文 Table 5 敏感性百分位、Fig. 12 第 4 行与补充材料）

做四件事
--------
1) 单样本图（默认 Sample 14，与 Fig.12 一致）
   背景：S(x) = sigma_K(x) * sqrt( sum_{t,y} w_t * (dC_t(y)/dK(x))^2 )
         C = FNO-2(K_true)，w_t = TIME_WEIGHTS，y 遍历全部格点；
         即 K(x) 对"整个浓度场"的加权雅可比范数 × K 的先验标准差。
         不先对 C 取域平均，避免正负响应抵消、羽流位移被掩盖；
         完整雅可比用 Hutchinson 随机探针无偏估计（--probes 控制探针数）。
         局限：各格点 K 视为独立，未计入 K 的空间相关性，属一阶敏感性指示，非严格方差分解。
   叠加：四种策略的 20 口井（白叉=初始 3 口井，蓝点=新增井）+ C_true(t=5) 的 0.5 等值线（羽流前缘）
   标题附带该样本新增井所在位置的 S 百分位均值。

2) 100 样本统计
   对每个样本、每种策略，计算新增井（排除前 3 口初始井）所在位置的 S 在全网格中的百分位
   （均匀布井的期望约 50%），LHS / random 先对样本内 10 个布局取均值；
   再做 Active vs 各基线的双侧配对 Wilcoxon + Cohen's d。

3) [a] 运移区域富集比（100 样本，同上配对检验）
   三类区域由真值场划定（诊断用）：
     high_K       : K_true >= 75% 分位（优势流路径）
     low_K        : K_true <= 25% 分位（低渗屏障）
     plume_fringe : 0.1 <= C_true(t=5)/max <= 0.5（羽流边界带）
   富集比 = 新增井落入该区域的比例 / 该区域面积占比（均匀布井期望 = 1）。

4) [b] 集合方差贡献图（单样本）
   在四种策略共享的初始 3 井后验集合上（look-ahead 开始选井时可用的信息，不用真值），
     VC(x) = sum_t w_t sum_y Cov(K(x), C_t(y))^2 / Var(K(x))  /  sum_t w_t sum_y Var(C_t(y))
   即 K(x) 单独线性解释的加权浓度方差占比（0~1），直接连接 K 非均质性与 C 方差。
   3 井后验由 run_100.get_prior / prepare_initial 原样复现，
   并与 active.npz 中保存的 post3_C_variance_map 核对。

说明
----
- S 是基于 FNO-2 代理模型的敏感性，仅作诊断（用真值 K 求梯度，设计阶段并不可得）；
  VC 基于集合协方差，属线性一阶近似。
- sigma_K(x) 取训练集 6000 个 K 场的逐格点标准差（近似先验；--no-sigma-k 可关闭）。
- 默认对 S 与 VC 做 sigma=1 的高斯平滑，抑制逐像素噪声（--sigma 0 关闭）。

- 多 trial 方法统计时始终使用全部 10 个布局（代表性样本的 final_wells 只用于出图）。

用法
----
  python sensitivity_K_to_C.py                      # 全部
  python sensitivity_K_to_C.py --sample 14 --skip-stats
  python sensitivity_K_to_C.py --skip-figure --skip-ensemble

输出（./sensitivity_results/）
  sensitivity_maps.npy                     (N,51,51) 全部测试样本的 S(x)
  fig_sensitivity_sample_XXX.png           单样本 S 图 + 四方法井位
  well_sensitivity_percentile.csv          每样本每方法的井位百分位
  figS_sensitivity_percentile_boxplot.png  百分位箱线图
  well_zone_enrichment.csv                 [a] 每样本每方法每区域的富集比
  figS_zone_enrichment_boxplot.png         [a] 富集比箱线图
  well_sensitivity_summary.txt             百分位 + 富集比汇总与检验
  variance_contribution_sample_XXX.npy     [b] VC(x)
  figS_variance_contribution_sample_XXX.png[b] K_true / S / VC 三联图
  variance_contribution_sample_XXX.txt     [b] 各方法新增井的 VC 百分位、S–VC 相关
"""

import os
import argparse
import numpy as np
import torch
import scipy.io as sio
from scipy import stats
from scipy.ndimage import gaussian_filter

import shared_utils as su
import run_100 as rl


class SensConfig(rl.TestConfig):
    OUTPUT_ROOT = './sensitivity_results'


METHODS = ('active', 'greedy', 'lhs', 'random')
# 与 run_100.METHOD_FILES 一致
RESULT_FILES = {'active': 'active.npz', 'greedy': 'greedy.npz',
                'lhs': 'lhs.npz', 'random': 'random.npz'}
LABELS = {'active': 'Look-ahead', 'greedy': 'Greedy',
          'lhs': 'LHS', 'random': 'Random'}
COLORS = {'active': '#d62728', 'greedy': '#1f77b4',
          'lhs': '#2ca02c', 'random': '#7f7f7f'}
N_INIT = 3

# [a] 区域划分阈值
K_LOW_PCT, K_HIGH_PCT = 25, 75
PLUME_BAND = (0.1, 0.5)          # C_true(t=5) / max 的羽流边界带
ZONES = ('high_K', 'low_K', 'plume_fringe')
ZONE_LABELS = {'high_K': 'High-K (≥P75)',
               'low_K': 'Low-K (≤P25)',
               'plume_fringe': 'Plume fringe'}


# ==========================================
# 1. 敏感性计算
# ==========================================
def load_fno_kc(config):
    kc = np.load(config.KC_NORM_PATH)
    model = su.FNO2d_KC(modes1=12, modes2=12, width=64,
                        in_channels=1, out_channels=5,
                        c_mean=float(kc['c_mean']),
                        c_std=float(kc['c_std'])).to(config.DEVICE)
    model.load_state_dict(torch.load(config.FNO_KC_PATH,
                                     map_location=config.DEVICE))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_prior_sigma_k(config):
    """训练集 K 场的逐格点标准差，近似先验 sigma_K(x)。"""
    mat = sio.loadmat(config.K_DATA_PATH)
    key = [k for k in mat.keys() if not k.startswith('__')][0]
    return mat[key].astype(np.float32).std(axis=0)             # (G,G)


def compute_sensitivity(model, K_all, config, n_probe=64, chunk=32,
                        sigma_k=None, sigma=1.0, seed=0):
    """
    S(x) = sigma_K(x) * sqrt( sum_{t,y} w_t (dC_t(y)/dK(x))^2 )

    Hutchinson 估计：取 v ~ N(0, I)（形状同 C），则
        E[ (J^T (sqrt(w) * v))_x^2 ] = sum_{t,y} w_t J_{(t,y),x}^2
    把同一个 K 复制成一批（eval 模式下批内样本互相独立），
    每个副本对应一个探针，一次反传即得 chunk 个探针的 J^T(sqrt(w) v)。
    """
    dev = config.DEVICE
    w_sqrt = torch.tensor(config.TIME_WEIGHTS, dtype=torch.float32,
                          device=dev).sqrt().view(1, -1, 1, 1)
    gen = torch.Generator(device='cpu').manual_seed(seed)

    maps = []
    for n in range(len(K_all)):
        k0 = torch.tensor(K_all[n][None, None], dtype=torch.float32, device=dev)
        acc, cnt = 0.0, 0
        while cnt < n_probe:
            b = min(chunk, n_probe - cnt)
            k = k0.repeat(b, 1, 1, 1).requires_grad_(True)
            c = model(k, return_normalized=False)               # (b,5,G,G)
            v = torch.randn(c.shape, generator=gen).to(dev)
            loss = (w_sqrt * v * c).sum()
            g, = torch.autograd.grad(loss, k)                   # (b,1,G,G)
            acc = acc + (g ** 2).sum(dim=0).squeeze(0)
            cnt += b
        maps.append(torch.sqrt(acc / n_probe).cpu().numpy())
    S = np.stack(maps)                                          # (N,G,G)

    if sigma > 0:
        S = np.stack([gaussian_filter(m, sigma=sigma) for m in S])
    if sigma_k is not None:
        S = S * sigma_k[None]
    return S


# ==========================================
# 2. 井位读取
# ==========================================
def _as_dict(entry):
    if hasattr(entry, 'item'):
        entry = entry.item()
    if isinstance(entry, dict):
        return entry
    return {k: entry[k] for k in entry.dtype.names}


def result_path(root, sample_idx, method):
    return os.path.join(root, f'sample_{sample_idx:03d}', RESULT_FILES[method])


def load_layouts(path):
    """返回该文件中所有最终布局：单布局方法长度为 1，多 trial 方法为 10。

    多 trial 文件优先读 all_trial_histories（代表性样本额外保存的 final_wells
    只是其中一个展示 trial，不能用于统计）。
    """
    with np.load(path, allow_pickle=True) as data:
        if 'all_trial_histories' in data:
            return [np.asarray(_as_dict(th[-1])['wells'], dtype=int)
                    for th in data['all_trial_histories']]
        if 'final_wells' in data:
            return [np.asarray(data['final_wells'], dtype=int)]
        return [np.asarray(_as_dict(data['history'][-1])['wells'], dtype=int)]


def display_trial(sample_idx, n_layouts):
    """与 figure_plot.select_random_trial / run_100 代表性 trial 一致。"""
    if n_layouts <= 1:
        return 0
    return int(np.random.default_rng(seed=sample_idx).integers(0, n_layouts))


def well_percentiles(S, wells, n_init=N_INIT):
    """新增井所在位置的 S 在整幅网格中的百分位（0~100）。"""
    added = np.array(wells)[n_init:].astype(int)
    vals = S[added[:, 0], added[:, 1]]
    flat = S.ravel()
    return np.array([(flat < v).mean() * 100.0 for v in vals])


# ==========================================
# [a] 运移区域与富集比
# ==========================================
def zone_masks(K_true, C_true):
    k_lo, k_hi = np.percentile(K_true, [K_LOW_PCT, K_HIGH_PCT])
    c_last = C_true[-1]
    c_rel = c_last / c_last.max() if c_last.max() > 0 else c_last
    return {
        'high_K':       K_true >= k_hi,
        'low_K':        K_true <= k_lo,
        'plume_fringe': (c_rel >= PLUME_BAND[0]) & (c_rel <= PLUME_BAND[1]),
    }


def zone_enrichment(wells, mask, n_init=N_INIT):
    """新增井落入区域的比例 / 区域面积占比；均匀布井期望 = 1。"""
    area = float(mask.mean())
    if area == 0:
        return np.nan
    added = np.array(wells)[n_init:].astype(int)
    return float(mask[added[:, 0], added[:, 1]].mean()) / area


# ==========================================
# 3. 单样本图
# ==========================================
def plot_sample_figure(S, C_true, layouts_by_method, sample_idx, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})

    G = S.shape[0]
    S_show = S / S.max()
    methods = [m for m in METHODS if m in layouts_by_method]

    fig, axes = plt.subplots(1, len(methods), figsize=(4.6 * len(methods), 4.6),
                             squeeze=False)
    im = None
    for ax, m in zip(axes[0], methods):
        wells = np.array(layouts_by_method[m]['wells'])
        pct = well_percentiles(S, wells).mean()

        im = ax.imshow(S_show, origin='lower', cmap='YlOrRd', vmin=0, vmax=1,
                       aspect='equal', interpolation='bilinear')
        ax.contour(C_true[4], levels=[0.5], colors='k', linewidths=1.0,
                   linestyles='--')
        init, added = wells[:N_INIT], wells[N_INIT:]
        ax.scatter(init[:, 1], init[:, 0], c='white', marker='x', s=110,
                   linewidths=2, edgecolors='white', zorder=10,
                   label='Initial wells')
        ax.scatter(added[:, 1], added[:, 0], c='blue', marker='o', s=45,
                   edgecolors='white', linewidths=0.8, zorder=10,
                   label='Added wells')
        ax.set_xlim(-0.5, G - 0.5)
        ax.set_ylim(-0.5, G - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        tag = layouts_by_method[m].get('tag', '')
        ax.set_title(f"{LABELS[m]}{tag}\nmean sensitivity percentile = {pct:.3f}%",
                     fontsize=12)

    axes[0, 0].legend(loc='lower right', fontsize=10, framealpha=0.8)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label(r'Normalized $\sigma_K\,\|\partial C/\partial K\|_w$', fontsize=11)
    fig.suptitle(f'Sample {sample_idx} - K→C sensitivity with final 20-well layouts '
                 '(dashed: C=0.5 contour at t=5)', fontsize=13, y=1.02)

    path = os.path.join(out_dir, f'fig_sensitivity_sample_{sample_idx:03d}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {path}")


def pick_layouts_for_figure(sample_idx, results_dir):
    """多 trial 方法取展示 trial（LHS 与 Random 同一编号，与 Fig.12 一致）。"""
    out = {}
    for m in METHODS:
        p = result_path(results_dir, sample_idx, m)
        if not os.path.exists(p):
            continue
        layouts = load_layouts(p)
        idx = display_trial(sample_idx, len(layouts))
        tag = f' (Trial {idx})' if len(layouts) > 1 else ''
        out[m] = dict(wells=layouts[idx], tag=tag)
    return out


# ==========================================
# 4. 100 样本统计（百分位 + [a] 富集比）
# ==========================================
def _boxplot(ax, arr):
    data, labels, cols = [], [], []
    for m in METHODS:
        v = arr[m][~np.isnan(arr[m])]
        if len(v):
            data.append(v)
            labels.append(LABELS[m])
            cols.append(COLORS[m])
    if not data:
        return False
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=1.6),
                    flierprops=dict(marker='o', markersize=3,
                                    markerfacecolor='gray', alpha=0.6))
    for patch, c in zip(bp['boxes'], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    return True


def plot_percentile_boxplot(arr, out_dir):
    """补充材料图：四种方法新增井的敏感性百分位箱线图（100 样本，50% 为均匀布井期望）。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    if not _boxplot(ax, arr):
        plt.close(fig)
        return
    ax.axhline(50, color='gray', linestyle='--', linewidth=0.9)
    ax.set_ylabel('Sensitivity percentile of added wells (%)')
    ax.text(0.98, 0.02, 'dashed line: uniform expectation (50%)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color='gray')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figS_sensitivity_percentile_boxplot.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {path}")


def plot_zone_boxplot(zone_arr, out_dir):
    """[a] 补充材料图：三类运移区域的新增井富集比（1 = 均匀布井期望）。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})

    fig, axes = plt.subplots(1, len(ZONES), figsize=(5.0 * len(ZONES), 4.4))
    for ax, z in zip(axes, ZONES):
        if _boxplot(ax, zone_arr[z]):
            ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.9)
            ax.set_title(ZONE_LABELS[z], fontsize=12)
    axes[0].set_ylabel('Enrichment ratio of added wells')
    fig.text(0.99, 0.01, 'dashed line: uniform expectation (1.0)',
             ha='right', va='bottom', fontsize=9, color='gray')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figS_zone_enrichment_boxplot.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {path}")


def _paired_lines(arr, fmt_diff='{:>+12.3f}'):
    """Look-ahead vs 各基线：双侧配对 Wilcoxon + Cohen's d（d = mean(diff)/sd(diff)）。"""
    lines = [f"  {'comparison':<24}{'n':>5}{'mean diff':>12}{'W':>10}{'p':>12}{'d':>8}"]
    for m in ('greedy', 'lhs', 'random'):
        a, b = arr['active'], arr[m]
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 5:
            continue
        diff = a[mask] - b[mask]
        w, p = stats.wilcoxon(a[mask], b[mask], alternative='two-sided')
        d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-15)
        p_str = '< 0.001' if p < 0.001 else f"{p:.3f}"
        lines.append(f"  {'Look-ahead vs ' + LABELS[m]:<24}{int(mask.sum()):>5}"
                     + fmt_diff.format(np.mean(diff))
                     + f"{w:>10.0f}{p_str:>12}{d:>8.3f}")
    return lines


def run_percentile_stats(S_all, K_all, C_all, results_dir, n_samples, out_dir):
    per_method = {m: [] for m in METHODS}
    per_zone = {z: {m: [] for m in METHODS} for z in ZONES}
    rows, zone_rows = [], []
    for si in range(n_samples):
        masks = zone_masks(K_all[si], C_all[si])
        got = {}
        got_z = {z: {} for z in ZONES}
        for m in METHODS:
            p = result_path(results_dir, si, m)
            if not os.path.exists(p):
                got[m] = np.nan
                for z in ZONES:
                    got_z[z][m] = np.nan
                continue
            try:
                layouts = load_layouts(p)
                got[m] = float(np.mean([well_percentiles(S_all[si], w).mean()
                                        for w in layouts]))
                for z in ZONES:
                    got_z[z][m] = float(np.mean([zone_enrichment(w, masks[z])
                                                 for w in layouts]))
            except Exception as e:
                print(f"  [Warning] sample {si} {m}: {e}")
                got[m] = np.nan
                for z in ZONES:
                    got_z[z][m] = np.nan
        for m in METHODS:
            per_method[m].append(got[m])
            for z in ZONES:
                per_zone[z][m].append(got_z[z][m])
        rows.append((si, got))
        zone_rows.append((si, got_z, {z: float(masks[z].mean()) for z in ZONES}))

    arr = {m: np.array(v, dtype=float) for m, v in per_method.items()}
    zone_arr = {z: {m: np.array(v, dtype=float) for m, v in per_zone[z].items()}
                for z in ZONES}
    plot_percentile_boxplot(arr, out_dir)
    plot_zone_boxplot(zone_arr, out_dir)

    with open(os.path.join(out_dir, 'well_sensitivity_percentile.csv'), 'w') as f:
        f.write('sample,' + ','.join(METHODS) + '\n')
        for si, got in rows:
            f.write(f"{si}," + ','.join(f"{got[m]:.3f}" for m in METHODS) + '\n')

    with open(os.path.join(out_dir, 'well_zone_enrichment.csv'), 'w') as f:
        f.write('sample,zone,area_fraction,' + ','.join(METHODS) + '\n')
        for si, got_z, areas in zone_rows:
            for z in ZONES:
                f.write(f"{si},{z},{areas[z]:.3f}," +
                        ','.join(f"{got_z[z][m]:.3f}" for m in METHODS) + '\n')

    lines = ["=" * 78,
             "  Sensitivity percentile of added wells (S = sigma_K * ||dC/dK||_w, FNO-2 based)",
             "  Uniform expectation ~50%; LHS/Random averaged over 10 layouts per sample",
             "=" * 78,
             f"  {'method':<12}{'n':>5}{'mean±std (%)':>20}"]
    for m in METHODS:
        v = arr[m][~np.isnan(arr[m])]
        if len(v):
            lines.append(f"  {LABELS[m]:<12}{len(v):>5}{np.mean(v):>13.3f} ± {np.std(v):<7.3f}")
    lines += ["", "  Paired test: Look-ahead vs baseline (two-sided Wilcoxon, Cohen's d)"]
    lines += _paired_lines(arr)

    # [a] 富集比
    lines += ["", "=" * 78,
              "  [a] Transport-zone enrichment ratio of added wells",
              "      ratio = fraction of added wells in zone / zone area fraction"
              "  (uniform expectation = 1)",
              f"      high_K: K_true >= P{K_HIGH_PCT};  low_K: K_true <= P{K_LOW_PCT};"
              f"  plume_fringe: {PLUME_BAND[0]} <= C_true(t=5)/max <= {PLUME_BAND[1]}",
              "      Zones defined from true fields (diagnostic only)",
              "=" * 78]
    for z in ZONES:
        areas = np.array([a[z] for _, _, a in zone_rows])
        lines.append(f"\n  {ZONE_LABELS[z]}  (mean area fraction = {areas.mean():.3f})")
        lines.append(f"  {'method':<12}{'n':>5}{'mean±std':>20}")
        for m in METHODS:
            v = zone_arr[z][m][~np.isnan(zone_arr[z][m])]
            if len(v):
                lines.append(f"  {LABELS[m]:<12}{len(v):>5}"
                             f"{np.mean(v):>13.3f} ± {np.std(v):<7.3f}")
        lines += _paired_lines(zone_arr[z], fmt_diff='{:>+12.3f}')
    lines += ["", "  d > 0: Look-ahead higher than baseline", "=" * 78]

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(out_dir, 'well_sensitivity_summary.txt'),
              'w', encoding='utf-8') as f:
        f.write(text + "\n")


def sample_sigma_f1(results_root, si):
    """该样本在主实验中实际使用的 σ_f1（保存在结果文件中）。

    主实验若在不同机器上运行，各自在验证集上算出的 σ_f1 存在微小浮点差异；
    逐样本重放使用该样本原来的值，保证与主实验采用相同的 R。
    """
    path = os.path.join(results_root, f'sample_{si:03d}', 'active.npz')
    with np.load(path, allow_pickle=True) as d:
        return float(d['sigma_f1_validation'])


# ==========================================
# [b] 集合方差贡献图（单样本，共享 3 井后验）
# ==========================================
def ensemble_variance_contribution(K_ens, C_ens, weights):
    """
    VC(x) = sum_t w_t sum_y Cov(K(x),C_t(y))^2 / Var(K(x))  /  sum_t w_t sum_y Var(C_t(y))

    Cov^2/Var(K) = R^2(x,y) * Var(C(y)) <= Var(C(y))，故 VC(x) ∈ [0, 1]：
    K(x) 单独线性解释的时间加权浓度方差占比。
    """
    N, G = K_ens.shape[0], K_ens.shape[-1]
    Ka = K_ens.reshape(N, -1).astype(np.float64)
    Ka -= Ka.mean(axis=0)
    var_k = (Ka ** 2).sum(axis=0) / (N - 1)
    num = np.zeros(Ka.shape[1])
    total = 0.0
    for t, w in enumerate(weights):
        Ca = C_ens[:, t].reshape(N, -1).astype(np.float64)
        Ca -= Ca.mean(axis=0)
        cov = Ka.T @ Ca / (N - 1)                       # (P, P)
        num += w * (cov ** 2).sum(axis=1)
        total += w * ((Ca ** 2).sum(axis=0) / (N - 1)).sum()
    vc = num / np.maximum(var_k, 1e-12) / max(total, 1e-12)
    return vc.reshape(G, G)


def run_variance_contribution(sample_idx, S, results_dir, out_dir, config,
                              sigma=1.0):
    print(f"\n[b] 复现样本 {sample_idx} 的共享 3 井后验，计算集合方差贡献 ...")
    vae, fno_kh, fno_kc = su.load_all_models(config)
    config.SIGMA_F1 = sample_sigma_f1(results_dir, sample_idx)   # 该样本在主实验中实际使用的 R
    K_test, H_test, C_test, K_train = su.load_test_data(config)
    truth = (K_test[sample_idx], H_test[sample_idx], C_test[sample_idx])
    models = (vae, fno_kh, fno_kc)

    z_prior = rl.get_prior(vae, K_train, config)
    z3, C3, _ = rl.prepare_initial(truth, z_prior, models, config, sample_idx)
    K3 = su.vae_decode(z3, vae, config)

    # 与主实验保存的 3 井方差场核对
    ref_path = result_path(results_dir, sample_idx, 'active')
    if os.path.exists(ref_path):
        with np.load(ref_path, allow_pickle=True) as d:
            if 'post3_C_variance_map' in d:
                ref = d['post3_C_variance_map']
                now = su.compute_weighted_C_variance(C3, config)[1]
                rel = float(np.abs(now - ref).max() / (np.abs(ref).max() + 1e-12))
                print(f"  [Check] post3 方差场与 active.npz 的最大相对差 = {rel:.2e}")

    VC = ensemble_variance_contribution(K3, C3, config.TIME_WEIGHTS)
    if sigma > 0:
        VC = gaussian_filter(VC, sigma=sigma)
    np.save(os.path.join(out_dir, f'variance_contribution_sample_{sample_idx:03d}.npy'), VC)

    layouts = pick_layouts_for_figure(sample_idx, results_dir)
    rho, p_rho = stats.spearmanr(S.ravel(), VC.ravel())
    lines = [f"Sample {sample_idx}: ensemble variance contribution VC(x) "
             f"(shared 3-well posterior, N_e = {len(K3)})",
             f"  Spearman rho(S, VC) over grid = {rho:.3f}  (p = {p_rho:.3e})",
             f"  {'method':<16}{'VC percentile of added wells (%)':>34}"]
    for m, info in layouts.items():
        pct = well_percentiles(VC, info['wells']).mean()
        lines.append(f"  {LABELS[m] + info['tag']:<16}{pct:>34.3f}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out_dir, f'variance_contribution_sample_{sample_idx:03d}.txt'),
              'w', encoding='utf-8') as f:
        f.write(text + "\n")

    _plot_variance_contribution(truth[0], truth[2], S, VC, layouts,
                                sample_idx, rho, out_dir)


def _plot_variance_contribution(K_true, C_true, S, VC, layouts, sample_idx,
                                rho, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})

    k_lo, k_hi = np.percentile(K_true, [K_LOW_PCT, K_HIGH_PCT])
    c_last = C_true[-1]
    c_rel = c_last / c_last.max() if c_last.max() > 0 else c_last

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    panels = [(K_true, 'viridis', 'True K with transport zones'),
              (S / S.max(), 'YlOrRd', 'K→C sensitivity S (normalized)'),
              (VC / VC.max(), 'YlOrRd',
               f'Variance contribution VC (normalized)\nSpearman ρ(S, VC) = {rho:.3f}')]
    for ax, (arr, cmap, title) in zip(axes, panels):
        im = ax.imshow(arr, origin='lower', cmap=cmap, aspect='equal')
        ax.contour(K_true, levels=[k_hi], colors='white', linewidths=0.9)
        ax.contour(K_true, levels=[k_lo], colors='cyan', linewidths=0.9)
        ax.contour(c_rel, levels=list(PLUME_BAND), colors='k',
                   linewidths=0.9, linestyles='--')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for m, marker in (('active', 'o'), ('lhs', 's')):
        if m in layouts:
            w = np.asarray(layouts[m]['wells'])[N_INIT:]
            axes[2].scatter(w[:, 1], w[:, 0], c=COLORS[m], marker=marker, s=40,
                            edgecolors='white', linewidths=0.8, zorder=10,
                            label=f"{LABELS[m]}{layouts[m]['tag']} added wells")
    axes[2].legend(loc='lower right', fontsize=9, framealpha=0.85)
    fig.suptitle(f'Sample {sample_idx} — white: K = P{K_HIGH_PCT}, cyan: K = P{K_LOW_PCT}, '
                 f'dashed: plume fringe ({PLUME_BAND[0]}–{PLUME_BAND[1]} of max C at t=5)',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, f'figS_variance_contribution_sample_{sample_idx:03d}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {path}")


# ==========================================
# 主函数
# ==========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=14)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--results', default='./test_100_results')
    ap.add_argument('--out', default='./sensitivity_results')
    ap.add_argument('--sigma', type=float, default=1.0, help='高斯平滑，0 关闭')
    ap.add_argument('--probes', type=int, default=64, help='Hutchinson 探针数')
    ap.add_argument('--no-sigma-k', action='store_true', help='不乘 sigma_K(x)')
    ap.add_argument('--skip-figure', action='store_true')
    ap.add_argument('--skip-stats', action='store_true')
    ap.add_argument('--skip-ensemble', action='store_true', help='跳过 [b] 集合方差贡献图')
    args = ap.parse_args()

    config = SensConfig()
    os.makedirs(args.out, exist_ok=True)

    print("[Info] 加载测试数据与 FNO-2 ...")
    test = np.load(config.TEST_DATA_PATH)
    K_all, C_all = test['K_test'], test['C_test']
    model = load_fno_kc(config)

    print("[Info] 计算 K→C 敏感性 ...")
    sigma_k = None if args.no_sigma_k else load_prior_sigma_k(config)
    S_all = compute_sensitivity(model, K_all, config, n_probe=args.probes,
                                sigma_k=sigma_k, sigma=args.sigma)
    np.save(os.path.join(args.out, 'sensitivity_maps.npy'), S_all)
    print(f"  S_all: {S_all.shape}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not args.skip_figure:
        layouts = pick_layouts_for_figure(args.sample, args.results)
        if layouts:
            plot_sample_figure(S_all[args.sample], C_all[args.sample],
                               layouts, args.sample, args.out)
        else:
            print(f"[Warning] 找不到样本 {args.sample} 的井位文件，跳过出图。")

    if not args.skip_stats:
        run_percentile_stats(S_all, K_all, C_all, args.results,
                             min(args.n, len(K_all)), args.out)

    if not args.skip_ensemble:
        run_variance_contribution(args.sample, S_all[args.sample], args.results,
                                  args.out, config, sigma=args.sigma)


if __name__ == "__main__":
    main()
