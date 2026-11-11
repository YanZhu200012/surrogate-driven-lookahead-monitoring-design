"""
figure_plot.py
代表性样本的空间场、井位与学习曲线图（默认 Sample 14）。
  Fig4_Wells.png 即正文 Fig. 12；spatial_stats.txt 为该样本的空间分布指标。
输入：run_100.py 为 PLOT_SAMPLE_IDS 中的样本保存的完整后验集合
      ./test_100_results/sample_XXX/{active,greedy,lhs,random}.npz
输出：./test_100_results/paper_figures_output/sample_XXX/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ============================================================
# 期刊图像全局样式
# 颜色约定：Look-ahead=绿、Greedy=蓝、LHS=橙、Random=红（LHS/Random 为虚线）
# ============================================================
JOURNAL_RC = {
    'font.family':        'Arial',
    'font.size':          12,
    'axes.titlesize':     13,
    'axes.labelsize':     12,
    'xtick.labelsize':    14,
    'ytick.labelsize':    11,
    'legend.fontsize':    12,
    'figure.titlesize':   14,
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
}

CBAR_KW = dict(fraction=0.046, pad=0.04)
SAVE_KW = dict(dpi=300, bbox_inches='tight')

# ------------------------------------------------------------
# 多行×多列场图的统一几何（与 evaluate_es_mda.plot_field_comparison
# 即论文 Figure 8 完全一致），保证 Figure 12 与 Figure 8 排版后
# 子图大小、字号比例相同。
#   figsize = (n_cols * 3.5, n_rows * 3.0)
#   GridSpec(hspace=0.12, wspace=0.30)，不调用 tight_layout
# ------------------------------------------------------------
FIELD_PANEL_W   = 3.5
FIELD_PANEL_H   = 3.0
FIELD_GS_KW     = dict(hspace=0.12, wspace=0.30)
FIELD_CBAR_TICK = 10      # = evaluate_es_mda 中 colorbar 刻度字号

# Fig4（Figure 12）与 Figure 8 使用完全相同的几何：
#   画布、GridSpec 间距、每个面板右侧 colorbar 占位（fraction=0.046, pad=0.04）
#   都相同；非最后一列的 colorbar 仅隐藏、仍占位，因此面板位置与大小逐一对齐。
FIG4_PANEL_H = FIELD_PANEL_H
FIG4_GS_KW   = FIELD_GS_KW


def apply_journal_style():
    """应用期刊图像全局样式"""
    plt.rcParams.update(JOURNAL_RC)


# ============================================================
# 配置
# ============================================================
class Config:
    """所有路径和网格参数集中管理"""

    BASE_DIR    = './test_100_results'
    # 需包含在 run_100.py 的 PLOT_SAMPLE_IDS 中（只有这些样本保存了完整后验集合）
    SAMPLE_IDS  = [14]
    OUTPUT_ROOT = './test_100_results/paper_figures_output'

    GRID_SIZE    = 51
    GRID_ORIGIN  = -4900
    GRID_SPACING = 196

    C_WEIGHTS    = [0.1, 0.15, 0.2, 0.25, 0.3]
    N_TIMES      = 5
    N_INIT_WELLS = 3

    # 固定随机试验索引（可视化用），None 表示取均值最好的那次
    FIXED_RAND_TRIAL = None

    # Fig4 主图是否绘制总标题（论文用图注代替，与 Figure 8 一致默认关闭）
    SHOW_FIG4_SUPTITLE = False

    # sensitivity_K_to_C.py 输出的 K→C 敏感性图 (N,51,51)，按样本编号索引，
    # 叠加为 Fig4 第 4 行；文件不存在或设为 None 时不绘制该行。
    SENS_MAPS_PATH  = './sensitivity_results/sensitivity_maps.npy'

    @staticmethod
    def get_sample_folder_name(sample_id):
        return f'sample_{sample_id:03d}'

    @staticmethod
    def get_input_paths(sample_id):
        folder_name = Config.get_sample_folder_name(sample_id)
        sample_dir  = os.path.join(Config.BASE_DIR, folder_name)
        return {
            'Active': os.path.join(sample_dir, 'active.npz'),
            'Greedy': os.path.join(sample_dir, 'greedy.npz'),
            'LHS':    os.path.join(sample_dir, 'lhs.npz'),
            'Random': os.path.join(sample_dir, 'random.npz'),
        }

    @staticmethod
    def get_output_dir(sample_id):
        folder_name = Config.get_sample_folder_name(sample_id)
        return os.path.join(Config.OUTPUT_ROOT, folder_name)


# ============================================================
# 全局方法样式映射（4 方法统一配置：颜色 / 标记 / 显示标签）
# ------------------------------------------------------------
# 所有图函数引用本字典，避免在多处硬编码颜色 / 标签。
# - color : 折线图、散点图、柱状图通用颜色
# - marker: 折线图标记
# - label : 图例 / 列标题中显示的方法标签
# - linestyle: 折线图线型（LHS 用虚线区分非自适应方法）
# ============================================================
METHOD_STYLE = {
    'Active': {'color': '#2ca02c', 'marker': 'o', 'label': 'Optimal (Lookahead)',
               'short_label': 'Optimal',  'linestyle': '-'},
    'Greedy': {'color': '#1f77b4', 'marker': '^', 'label': 'Greedy (MaxVar)',
               'short_label': 'MaxVar',   'linestyle': '-'},
    'LHS':    {'color': '#ff7f0e', 'marker': 'D', 'label': 'LHS-20 (space-filling)',
               'short_label': 'LHS-20',   'linestyle': '--'},
    'Random': {'color': '#d62728', 'marker': 's', 'label': 'Random',
               'short_label': 'Random',   'linestyle': '--'},
}

# 标准方法顺序（绘图、统计循环统一遵守）
METHOD_ORDER = ['Active', 'Greedy', 'LHS', 'Random']


# ============================================================
# 工具函数
# ============================================================
def add_colorbar(im, ax):
    """统一 colorbar，无单位标签，tick 字号匹配"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=JOURNAL_RC['xtick.labelsize'])
    return cbar


def index_to_coord(row, col):
    x = Config.GRID_ORIGIN + col * Config.GRID_SPACING
    y = Config.GRID_ORIGIN + row * Config.GRID_SPACING
    return x, y


def select_random_trial(random_data, sample_id=0):
    """
    选择用于空间场展示的 Random 试验索引。
    使用 sample_id 作为随机种子，保证同一样本每次选同一个 trial。
    """
    histories = _get_raw_histories(random_data)
    if not histories:
        return 0

    n_trials = len(histories)
    # 用 sample_id 作种子：同一样本永远选同一个 trial，不同样本选不同 trial
    rng      = np.random.default_rng(seed=sample_id)
    rand_idx = int(rng.integers(0, n_trials))
    print(f"  [Info] Selected Random Trial #{rand_idx} / {n_trials} "
          f"(seed={sample_id})")
    return rand_idx


def _get_raw_histories(d):
    """从 Random 数据字典中取出 histories 列表（兼容多种键名）。"""
    for key in ['all_histories', 'all_trial_histories', 'random_histories']:
        if key in d:
            h = d[key]
            return h.tolist() if hasattr(h, 'tolist') else h
    return []


def extract_metric(history, metric_key):
    return [step.get(metric_key, np.nan) for step in history]


def extract_random_metrics(histories, metric_key):
    """
    从多次随机试验中提取指标的均值和标准差。
    返回: (n_wells 列表, mean 数组, std 数组)
    """
    if not histories or len(histories) == 0:
        return None, None, None

    min_len = min(len(h) for h in histories)
    if min_len == 0:
        return None, None, None

    # 从第一个 trial 取 n_wells 序列
    n_wells = [step.get('n_wells', i + 1) for i, step in enumerate(histories[0][:min_len])]

    values_matrix = []
    for h in histories:
        val = [step.get(metric_key, np.nan) for step in h[:min_len]]
        values_matrix.append(val)

    values_matrix = np.array(values_matrix, dtype=float)
    mean = np.nanmean(values_matrix, axis=0)
    std  = np.nanstd(values_matrix,  axis=0)
    return n_wells, mean, std


# ============================================================
# 空间分布统计指标
# ============================================================

def compute_spatial_stats(wells, n_init, grid_size=51,
                          plume_mask=None, boundary_width=8):
    """
    计算单组井位的空间聚集/离散程度指标，用于 4.3.3 节定量描述。

    参数
    ----
    wells        : list of [row, col]  所有井位（初始 + 新增）
    n_init       : int                 初始井数，新增井 = wells[n_init:]
    grid_size    : int                 网格边长（默认 51）
    plume_mask   : ndarray (G,G) bool  污染羽流覆盖区域掩码（可为 None）
    boundary_width: int                边界区宽度（格子数，用于统计边界区覆盖率）

    返回
    ----
    stats : dict，包含以下字段
        n_added        : 新增井数量
        ann_distance   : 平均最近邻距离 ANN（格子单位）
        clark_evans    : Clark-Evans 指数 R（R<1 聚集，R=1 随机，R>1 均匀）
        coverage_rate  : 新增井在网格上的覆盖率
                         （以每口井为中心、半径 r=boundary_width/2 的圆覆盖面积 / 总格子数）
        boundary_rate  : 新增井中位于边界区域（距四边 ≤ boundary_width 格）的比例
        plume_inside   : 新增井落在 plume_mask 内的比例（若 plume_mask 为 None 则 None）
        sens_percentile: 新增井所在位置的 K→C 敏感性百分位均值（由调用者填入；默认 NaN）
    """
    wells_arr = np.array(wells)
    added     = wells_arr[n_init:]          # 新增井
    n_added   = len(added)
    G         = grid_size

    stats = dict(n_added=n_added, ann_distance=np.nan,
                 clark_evans=np.nan, coverage_rate=np.nan,
                 boundary_rate=np.nan, plume_inside=None,
                 sens_percentile=np.nan)

    if n_added == 0:
        return stats

    # ── ① 平均最近邻距离（Average Nearest-Neighbor Distance, ANN）────────
    # 对每口新增井，找它与其他所有新增井的最小距离，取均值
    if n_added >= 2:
        dists = []
        for i in range(n_added):
            diff = added - added[i]                        # (n_added, 2)
            d    = np.sqrt((diff ** 2).sum(axis=1))
            d[i] = np.inf                                  # 排除自身
            dists.append(d.min())
        ann = float(np.mean(dists))
        stats['ann_distance'] = ann

        # ── ② Clark-Evans 指数 R ─────────────────────────────────────────
        # R = ANN_observed / ANN_expected
        # ANN_expected（CSR下）= 0.5 / sqrt(密度)
        # 密度 = n_added / 研究区格子数  (以整个网格 G×G 为研究区)
        area     = float(G * G)
        density  = n_added / area
        ann_exp  = 0.5 / np.sqrt(density)
        stats['clark_evans'] = ann / ann_exp
    else:
        stats['ann_distance'] = np.nan
        stats['clark_evans']  = np.nan

    # ── ③ 覆盖率（Coverage Rate）────────────────────────────────────────
    # 以每口新增井为圆心、半径 r（格子）画圆，统计覆盖格子数 / 总格子数
    # r = boundary_width / 2，确保与边界区宽度参数一致（可按需调整）
    r         = boundary_width / 2.0
    covered   = np.zeros((G, G), dtype=bool)
    rows_grid = np.arange(G)
    cols_grid = np.arange(G)
    cc, rr    = np.meshgrid(cols_grid, rows_grid)   # (G, G)

    for w in added:
        dist2 = (rr - w[0]) ** 2 + (cc - w[1]) ** 2
        covered |= (dist2 <= r ** 2)

    stats['coverage_rate'] = float(covered.sum()) / (G * G)

    # ── ④ 边界区比例（Boundary Rate）────────────────────────────────────
    # 新增井中距任意一条边 ≤ boundary_width 格子的比例
    in_boundary = (
        (added[:, 0] <= boundary_width) |
        (added[:, 0] >= G - 1 - boundary_width) |
        (added[:, 1] <= boundary_width) |
        (added[:, 1] >= G - 1 - boundary_width)
    )
    stats['boundary_rate'] = float(in_boundary.sum()) / n_added

    # ── ⑤ 污染羽流内覆盖比例（Plume Inside Rate）────────────────────────
    if plume_mask is not None:
        plume_mask_bool = np.array(plume_mask, dtype=bool)
        inside = np.array([plume_mask_bool[int(w[0]), int(w[1])]
                           for w in added])
        stats['plume_inside'] = float(inside.sum()) / n_added

    # sens_percentile 由 compute_spatial_stats_all_methods 根据 S_map 填入

    return stats


def compute_spatial_stats_all_methods(data, rand_idx, rand_histories,
                                      n_init, grid_size, prior_C_var=None,
                                      S_map=None):
    """
    对 Active / Greedy / LHS / Random 四种方法分别调用 compute_spatial_stats()，
    返回汇总字典。

    plume_mask 由 prior_C_var 场的高方差区域（上四分位数）自动生成。
    若提供 S_map，则额外计算各方法新增井的 K→C 敏感性百分位（sens_percentile）。
    """
    # 用先验方差场的高方差区（>75th 百分位）作为 plume_mask 近似
    plume_mask = None
    if prior_C_var is not None:
        threshold  = np.percentile(prior_C_var, 75)
        plume_mask = prior_C_var >= threshold

    all_stats = {}

    # Active
    if 'Active' in data and 'final_wells' in data['Active']:
        wells = data['Active']['final_wells']
        if hasattr(wells, 'tolist'):
            wells = wells.tolist()
        all_stats['Optimal'] = compute_spatial_stats(
            wells, n_init, grid_size, plume_mask)

    # Greedy
    if 'Greedy' in data and 'final_wells' in data['Greedy']:
        wells = data['Greedy']['final_wells']
        if hasattr(wells, 'tolist'):
            wells = wells.tolist()
        all_stats['MaxVar'] = compute_spatial_stats(
            wells, n_init, grid_size, plume_mask)

    # LHS
    if 'LHS' in data and 'final_wells' in data['LHS']:
        wells = data['LHS']['final_wells']
        if hasattr(wells, 'tolist'):
            wells = wells.tolist()
        all_stats['LHS-20'] = compute_spatial_stats(
            wells, n_init, grid_size, plume_mask)

    # Random（取指定 trial）
    if rand_histories and len(rand_histories) > rand_idx:
        try:
            rand_wells = rand_histories[rand_idx][-1]['wells']
            all_stats[f'Random (Trial #{rand_idx})'] = compute_spatial_stats(
                rand_wells, n_init, grid_size, plume_mask)
        except (IndexError, KeyError) as e:
            print(f"    [Warning] Spatial stats for Random: {e}")

    # ── 填入 K→C 敏感性百分位（Sensitivity Percentile）──────────────
    if S_map is not None:
        _wells_map = {}
        if 'Active' in data and 'final_wells' in data['Active']:
            w = data['Active']['final_wells']
            _wells_map['Optimal'] = w.tolist() if hasattr(w, 'tolist') else w
        if 'Greedy' in data and 'final_wells' in data['Greedy']:
            w = data['Greedy']['final_wells']
            _wells_map['MaxVar'] = w.tolist() if hasattr(w, 'tolist') else w
        if 'LHS' in data and 'final_wells' in data['LHS']:
            w = data['LHS']['final_wells']
            _wells_map['LHS-20'] = w.tolist() if hasattr(w, 'tolist') else w
        if rand_histories and len(rand_histories) > rand_idx:
            try:
                _wells_map[f'Random (Trial #{rand_idx})'] = \
                    rand_histories[rand_idx][-1]['wells']
            except (IndexError, KeyError):
                pass

        for label, wells in _wells_map.items():
            if label in all_stats:
                all_stats[label]['sens_percentile'] = \
                    well_sens_percentile(S_map, wells, n_init)

    return all_stats


def format_spatial_stats_table(all_stats):
    """
    将空间统计结果格式化为期刊报告用的对齐表格字符串（ASCII）。
    同时返回每行的 dict 列表，供其他函数使用。
    """
    headers = ['Method', 'n_added', 'ANN dist', 'Clark-Evans R',
               'Coverage', 'Boundary %', 'Plume inside %', 'Sens. pct(%)']
    col_w   = [22, 9, 10, 15, 10, 12, 15, 14]

    sep  = '  ' + '-' * (sum(col_w) + len(col_w) - 1)
    head = '  ' + ''.join(f'{h:<{w}}' for h, w in zip(headers, col_w))

    lines = [sep, head, sep]
    for method, s in all_stats.items():
        pi_str = (f"{s['plume_inside']*100:5.1f}%"
                  if s['plume_inside'] is not None else '  N/A ')
        row = [
            method,
            str(s['n_added']),
            f"{s['ann_distance']:.2f}" if not np.isnan(s['ann_distance']) else 'N/A',
            f"{s['clark_evans']:.3f}"  if not np.isnan(s['clark_evans'])  else 'N/A',
            f"{s['coverage_rate']*100:.1f}%",
            f"{s['boundary_rate']*100:.1f}%",
            pi_str,
            f"{s['sens_percentile']:.1f}" if not np.isnan(s.get('sens_percentile', np.nan)) else 'N/A',
        ]
        lines.append('  ' + ''.join(f'{v:<{w}}' for v, w in zip(row, col_w)))
    lines.append(sep)

    # 解读：Clark-Evans R 意义注释
    lines.append('')
    lines.append('  Clark-Evans R: R < 1 → clustered  |  R = 1 → random  |  R > 1 → dispersed/uniform')
    lines.append('  ANN dist     : avg nearest-neighbor distance among added wells (grid units)')
    lines.append('  Coverage     : fraction of grid covered by circles of radius=boundary_width/2 around added wells')
    lines.append('  Boundary %   : fraction of added wells within boundary_width cells of any edge')
    lines.append('  Plume inside%: fraction of added wells inside high-variance region (>75th pct of prior C var)')
    lines.append('  Sens. pct(%) : mean sensitivity percentile of added wells on K→C sensitivity field (uniform ~50%)')

    return '\n'.join(lines)


# ============================================================
# 数据加载与标准化
# ============================================================
def load_data_for_sample(sample_id):
    data  = {}
    paths = Config.get_input_paths(sample_id)

    for name, path in paths.items():
        if os.path.exists(path):
            try:
                loaded      = np.load(path, allow_pickle=True)
                standardized = standardize_data_format(loaded, name)
                if standardized:
                    data[name] = standardized
                    print(f"    [Info] Loaded {name} from {path}")
                else:
                    print(f"    [Warning] Could not standardize {name} from {path}")
            except Exception as e:
                print(f"    [Warning] Failed to load {path}: {e}")
        else:
            print(f"    [Warning] File not found: {path}")

    return data


def standardize_data_format(loaded, method_name):
    result = {}

    # 复制所有原始字段
    for key in loaded.files:
        val = loaded[key]
        # numpy object 数组转 python list（history 等）
        if val.dtype == object:
            try:
                result[key] = val.tolist()
            except Exception:
                result[key] = val
        else:
            result[key] = val

    # ── 标准化 history ──
    if 'history' in result and not isinstance(result['history'], list):
        result['history'] = result['history'].tolist() if hasattr(result['history'], 'tolist') else result['history']

    # ── 标准化 final_wells ──
    if 'final_wells' in result:
        fw = result['final_wells']
        if hasattr(fw, 'tolist'):
            fw = fw.tolist()
        result['final_wells'] = np.array(fw)

    # ── 判断是否有完整样本 ──
    if method_name == 'Random':
        # 兼容多种键名
        for src_key, dst_key in [
            ('all_trial_histories', 'all_histories'),
            ('random_histories',    'all_histories'),
        ]:
            if src_key in result and 'all_histories' not in result:
                result['all_histories'] = result[src_key]

        # all_final_C_samples：run_100.py 保存为 (n_trials, N, 5, 51, 51)
        result['_has_full_samples'] = (
            'all_final_K_samples' in result or
            'all_final_C_samples' in result
        )
    else:
        result['_has_full_samples'] = (
            'K_samples_final' in result and
            'C_samples_final' in result
        )

    return result


def get_random_histories(data):
    if 'Random' not in data:
        return []
    return _get_raw_histories(data['Random'])


def check_data_completeness(data, sample_id):
    issues = []
    for method in ['Active', 'Greedy', 'LHS', 'Random']:
        if method not in data:
            issues.append(f"{method}: Not loaded")
            continue
        d = data[method]
        if 'K_true' not in d:
            issues.append(f"{method}: Missing K_true")
        if 'C_true' not in d:
            issues.append(f"{method}: Missing C_true")
        if not d.get('_has_full_samples', False):
            issues.append(f"{method}: Missing full samples for spatial plots")
    if issues:
        print(f"    [Data Issues for Sample {sample_id}]:")
        for issue in issues:
            print(f"      - {issue}")
    return len(issues) == 0


# ============================================================
# 辅助：获取方法的完整 K/C 样本集合
# ============================================================
def _get_k_samples(data, method, rand_idx):
    """返回 K_samples (N, 51, 51)"""
    d = data[method]
    if method == 'Random':
        arr = d.get('all_final_K_samples')
        if arr is None:
            return None
        arr = np.array(arr) if not isinstance(arr, np.ndarray) else arr
        # shape 可能是 (n_trials, N, 51, 51)
        if arr.ndim == 4:
            return arr[rand_idx]
        # shape (n_trials,) of object → already converted
        return np.array(arr[rand_idx])
    else:
        arr = d.get('K_samples_final')
        return np.array(arr) if arr is not None else None


def _get_c_samples(data, method, rand_idx):
    """返回 C_samples (N, 5, 51, 51)"""
    d = data[method]
    if method == 'Random':
        arr = d.get('all_final_C_samples')
        if arr is None:
            return None
        arr = np.array(arr) if not isinstance(arr, np.ndarray) else arr
        if arr.ndim == 5:
            return arr[rand_idx]
        return np.array(arr[rand_idx])
    else:
        arr = d.get('C_samples_final')
        return np.array(arr) if arr is not None else None


# ============================================================
# Fig1: K 场对比
# ============================================================
def plot_fig1_k_comparison(data, rand_idx, output_dir, sample_id):
    print(f"  Generating Fig 1: K Field Comparison for Sample {sample_id}...")
    apply_journal_style()  # 每张图确保风格一致

    methods = ['Active', 'Greedy', 'LHS', 'Random']
    titles  = ['Optimal', 'MaxVar', 'LHS-20', f'Random (Trial #{rand_idx})']

    valid_methods, method_data = [], {}
    k_trues, k_means, k_errors, k_stds = [], [], [], []

    for method in methods:
        if method not in data:
            continue
        d = data[method]
        if 'K_true' not in d:
            continue
        K_samples = _get_k_samples(data, method, rand_idx)
        if K_samples is None:
            print(f"    [Skip] {method}: No K_samples")
            continue

        K_true  = np.array(d['K_true'])
        K_mean  = np.mean(K_samples, axis=0)
        K_std   = np.std(K_samples, axis=0)
        K_error = np.abs(K_true - K_mean)

        k_trues.append(K_true);  k_means.append(K_mean)
        k_errors.append(K_error); k_stds.append(K_std)
        method_data[method] = dict(K_true=K_true, K_mean=K_mean,
                                   K_std=K_std, K_error=K_error)
        valid_methods.append(method)

    if not valid_methods:
        print("    [Skip] No valid data.")
        return

    k_min   = min(np.min(a) for a in k_trues + k_means)
    k_max   = max(np.max(a) for a in k_trues + k_means)
    err_max = max(np.max(a) for a in k_errors)
    std_max = max(np.max(a) for a in k_stds)

    n_rows  = len(valid_methods)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.5 * n_rows))  # 更紧凑期刊尺寸
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f'Sample {sample_id} - K Field Comparison',
                 fontsize=JOURNAL_RC['figure.titlesize'], fontweight='bold')

    for row, method in enumerate(valid_methods):
        title = titles[methods.index(method)]
        md = method_data[method]
        panels = [
            (md['K_true'],  'jet',     k_min,   k_max,   "K True"),
            (md['K_mean'],  'jet',     k_min,   k_max,   "Prediction (Mean)"),
            (md['K_error'], 'hot',     0,       err_max, "Error (|True−Pred|)"),
            (md['K_std'],   'viridis', 0,       std_max, "Posterior Std"),
        ]
        for col, (arr, cmap, vmin, vmax, col_title) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(arr, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
            add_colorbar(im, ax)
            if col == 0:
                ax.set_ylabel(title, fontsize=JOURNAL_RC['axes.labelsize'], fontweight='bold')
            if row == 0:
                ax.set_title(col_title, fontsize=JOURNAL_RC['axes.titlesize'])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(left=0.05, right=0.92, top=0.93, bottom=0.07)
    plt.savefig(os.path.join(output_dir, 'Fig1_K_Comparison.png'), **SAVE_KW)
    plt.close()
    print("    [Done] Fig1_K_Comparison.png")


# ============================================================
# Fig2: C 场加权平均对比
# ============================================================
def plot_fig2_c_weighted(data, rand_idx, output_dir, sample_id):
    print(f"  Generating Fig 2: C Field Weighted Average for Sample {sample_id}...")
    apply_journal_style()

    methods = ['Active', 'Greedy', 'LHS', 'Random']
    titles  = ['Optimal', 'MaxVar', 'LHS-20', f'Random (Trial #{rand_idx})']
    weights = Config.C_WEIGHTS

    valid_methods, method_data = [], {}
    c_trues, c_means, c_errors, c_stds = [], [], [], []

    for method in methods:
        if method not in data:
            continue
        d = data[method]
        if 'C_true' not in d:
            continue
        C_samples = _get_c_samples(data, method, rand_idx)
        if C_samples is None:
            print(f"    [Skip] {method}: No C_samples")
            continue

        C_true     = np.array(d['C_true'])
        C_true_avg = np.average(C_true, axis=0, weights=weights)

        # C_samples: (N, 5, 51, 51) → weighted → (N, 51, 51)
        C_sw   = np.average(C_samples, axis=1, weights=weights)
        C_mean = np.mean(C_sw, axis=0)
        C_std  = np.std(C_sw,  axis=0)
        C_err  = np.abs(C_true_avg - C_mean)

        c_trues.append(C_true_avg); c_means.append(C_mean)
        c_errors.append(C_err);     c_stds.append(C_std)
        method_data[method] = dict(C_true_avg=C_true_avg,
                                   C_mean=C_mean, C_std=C_std, C_error=C_err)
        valid_methods.append(method)

    if not valid_methods:
        print("    [Skip] No valid data.")
        return

    c_min   = min(np.min(a) for a in c_trues + c_means)
    c_max   = max(np.max(a) for a in c_trues + c_means)
    err_max = max(np.max(a) for a in c_errors)
    std_max = max(np.max(a) for a in c_stds)

    n_rows  = len(valid_methods)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f'Sample {sample_id} - C Field Weighted Comparison',
                 fontsize=JOURNAL_RC['figure.titlesize'], fontweight='bold')

    for row, method in enumerate(valid_methods):
        title = titles[methods.index(method)]
        md = method_data[method]
        panels = [
            (md['C_true_avg'], 'jet',     c_min,   c_max,   "C True (Weighted)"),
            (md['C_mean'],     'jet',     c_min,   c_max,   "C Mean (Weighted)"),
            (md['C_error'],    'hot',     0,       err_max, "C Error"),
            (md['C_std'],      'viridis', 0,       std_max, "C Posterior Std"),
        ]
        for col, (arr, cmap, vmin, vmax, col_title) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(arr, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
            add_colorbar(im, ax)
            if col == 0:
                ax.set_ylabel(title, fontsize=JOURNAL_RC['axes.labelsize'], fontweight='bold')
            if row == 0:
                ax.set_title(col_title, fontsize=JOURNAL_RC['axes.titlesize'])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(left=0.05, right=0.92, top=0.93, bottom=0.07)
    plt.savefig(os.path.join(output_dir, 'Fig2_C_Weighted_Comparison.png'), **SAVE_KW)
    plt.close()
    print("    [Done] Fig2_C_Weighted_Comparison.png")


# ============================================================
# Fig3: 收敛曲线
# ============================================================
def plot_fig3_curves(data, rand_histories, output_dir, sample_id):
    print(f"  Generating Fig 3: Convergence Curves for Sample {sample_id}...")
    apply_journal_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 更紧凑
    fig.suptitle(f'Sample {sample_id} - Convergence Curves',
                 fontsize=JOURNAL_RC['figure.titlesize'], fontweight='bold')

    configs_plot = [
        ('K_r2',  0, 'Hydraulic Conductivity (K) R² Score', 'R²',               False),
        ('C_r2',  1, 'Concentration (C) R² Score',         'R²',                False),
        ('C_var', 2, 'Concentration Variance (Uncertainty)','Variance (Log)',    True),
    ]
    has_data = False

    for metric, idx, title, ylabel, use_log in configs_plot:
        ax = axes[idx]

        if 'Active' in data and 'history' in data['Active']:
            history = data['Active']['history']
            if not isinstance(history, list):
                history = history.tolist()
            x = [h['n_wells'] for h in history]
            y = extract_metric(history, metric)
            ax.plot(x, y, 'g-o', label='Optimal (Lookahead)', linewidth=2)
            has_data = True

        if 'Greedy' in data and 'history' in data['Greedy']:
            history = data['Greedy']['history']
            if not isinstance(history, list):
                history = history.tolist()
            x = [h['n_wells'] for h in history]
            y = extract_metric(history, metric)
            ax.plot(x, y, 'b-^', label='Greedy (MaxVar)', linewidth=2)
            has_data = True

        # LHS-20: non-adaptive deterministic baseline (序贯协议下也有完整 history)
        if 'LHS' in data and 'history' in data['LHS']:
            history = data['LHS']['history']
            if not isinstance(history, list):
                history = history.tolist()
            x = [h['n_wells'] for h in history]
            y = extract_metric(history, metric)
            lhs_style = METHOD_STYLE['LHS']
            ax.plot(x, y,
                    color=lhs_style['color'],
                    marker=lhs_style['marker'],
                    linestyle=lhs_style['linestyle'],
                    label='LHS-20 (space-filling)', linewidth=2)
            has_data = True

        if rand_histories:
            x_rnd, y_mean, y_std = extract_random_metrics(rand_histories, metric)
            if x_rnd is not None:
                ax.plot(x_rnd, y_mean, 'r--', label='Random (Mean)', linewidth=2)
                ax.fill_between(x_rnd, y_mean - y_std, y_mean + y_std,
                                color='red', alpha=0.2)
                has_data = True

        if use_log:
            ax.set_yscale('log')
        ax.set_title(title, fontsize=JOURNAL_RC['axes.titlesize'])
        ax.set_xlabel('Number of Wells')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(Config.N_INIT_WELLS, 20)
        if idx == 0:
            ax.legend()

    if not has_data:
        print("    [Skip] No history data available.")
        plt.close()
        return

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'Fig3_Curves.png'), **SAVE_KW)
    plt.close()
    print("    [Done] Fig3_Curves.png")


# ============================================================
# Fig4: 井位分布（叠加背景场）
# ============================================================

# ============================================================
# LHS 与 Random 使用同一 trial 编号；K→C 敏感性图读取
# ============================================================
_SENS_CACHE = {}


def load_sensitivity_map(sample_id):
    """读取该样本的 K→C 敏感性图并归一化到 [0,1]；缺失则返回 None（对应行不绘制）。"""
    path = Config.SENS_MAPS_PATH
    if path is None:                      # 正文图不叠加敏感性
        return None
    if path not in _SENS_CACHE:
        _SENS_CACHE[path] = np.load(path) if os.path.exists(path) else None
        if _SENS_CACHE[path] is None:
            print(f"    [Warning] 未找到 {path}，Fig.4 将不绘制敏感性行；"
                  f"请先运行 sensitivity_K_to_C.py。")
    S = _SENS_CACHE[path]
    if S is None or sample_id >= len(S):
        return None
    m = np.array(S[sample_id], dtype=float)
    return m / m.max() if m.max() > 0 else m


def well_sens_percentile(S, wells, n_init):
    """新增井所在位置的敏感性在整幅网格中的百分位均值（均匀布井期望约 50%）。"""
    added = np.array(wells)[n_init:].astype(int)
    vals = S[added[:, 0], added[:, 1]]
    flat = S.ravel()
    return float(np.mean([(flat < v).mean() * 100.0 for v in vals]))


# ---------- 背景场提取辅助函数 ----------

_INIT_VAR_CACHE = {}


def load_initial_variance_map(sample_id):
    """读取共享初始 3 井同化后的浓度加权方差场（论文 Eq.13）。"""
    path = os.path.join(Config.BASE_DIR, f'sample_{sample_id:03d}', 'active.npz')
    if path in _INIT_VAR_CACHE:
        return _INIT_VAR_CACHE[path]

    var_map = None
    if os.path.exists(path):
        try:
            with np.load(path, allow_pickle=True) as d:
                if 'post3_C_variance_map' in d:
                    var_map = d['post3_C_variance_map']
                    print(f"    [Info] initial (3-well) C variance: loaded from {path} "
                          f"(max={var_map.max():.4f}, mean={var_map.mean():.4f})")
        except Exception as e:
            print(f"    [Warning] failed to read {path}: {e}")
    if var_map is None:
        print(f"    [WARNING] 未能读取 {path} 中的 post3_C_variance_map。")
    _INIT_VAR_CACHE[path] = var_map
    return var_map


def _get_background_fields(data, rand_idx, sample_id=None):
    """
    从任一可用方法中提取用于背景叠加的场数据：
      K_true        : (51, 51)   真实水力传导系数场
      C_true_last   : (51, 51)   最后时刻真实浓度羽流
      prior_C_var   : (51, 51)   3 口共享初始井同化后的浓度方差 sum_t w_t*Var_t（Eq.13）

    优先顺序：Active → Greedy → Random(rand_idx trial)
    """
    K_true       = None
    C_true_last  = None
    prior_C_var  = None

    # 优先从确定性方法取 K_true / C_true
    for method in ['Active', 'Greedy']:
        if method in data:
            d = data[method]
            if K_true is None and 'K_true' in d:
                K_true = np.array(d['K_true'])
            if C_true_last is None and 'C_true' in d:
                C_true_arr = np.array(d['C_true'])  # (5, 51, 51)
                C_true_last = C_true_arr[-1]         # 最后时刻

    # 如果确定性方法没有，从 Random 取
    if K_true is None or C_true_last is None:
        if 'Random' in data:
            d = data['Random']
            if K_true is None and 'K_true' in d:
                K_true = np.array(d['K_true'])
            if C_true_last is None and 'C_true' in d:
                C_true_arr = np.array(d['C_true'])
                C_true_last = C_true_arr[-1]

    # 选井前的浓度方差：3 口共享初始井同化后的结果（所有策略的共同起点）
    if sample_id is None:
        print("    [WARNING] _get_background_fields 未提供 sample_id，无法读取初始方差图。")
    else:
        prior_C_var = load_initial_variance_map(sample_id)

    return K_true, C_true_last, prior_C_var


def _overlay_background(ax, bg_arr, cmap, alpha, extent=None):
    """在子图上叠加半透明背景场，返回 AxesImage 用于添加 colorbar。"""
    if bg_arr is None:
        return None
    kw = dict(cmap=cmap, origin='lower', aspect='equal',
              alpha=alpha, interpolation='bilinear')
    if extent is not None:
        kw['extent'] = extent
    return ax.imshow(bg_arr, **kw)


def _scatter_wells(ax, wells, n_init, show_numbers=True,
                   s_init=120, lw_init=2.0, s_add=70, lw_add=0.8,
                   num_fontsize=7):
    """在子图上绘制井位散点（初始井 + 新增井）及编号。
    默认尺寸对应单行大图（Fig4a–e）；Fig4 主图按小面板传入缩小后的尺寸。"""
    all_w = np.array(wells)
    init  = all_w[:n_init]
    added = all_w[n_init:]

    h_init = ax.scatter(init[:, 1], init[:, 0],
                        c='white', marker='x', s=s_init, linewidths=lw_init,
                        label='Initial wells', zorder=10,
                        edgecolors='white')
    h_add = None
    if len(added) > 0:
        h_add = ax.scatter(added[:, 1], added[:, 0],
                           c='red', marker='o', s=s_add,
                           edgecolors='white', linewidths=lw_add,
                           label='Added wells', zorder=10)

    if show_numbers:
        for seq, well in enumerate(all_w, start=1):
            ax.text(well[1] + 0.6, well[0] - 0.6, str(seq),
                    fontsize=num_fontsize, color='white', ha='left', va='top',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', fc='black',
                              ec='none', alpha=0.45))
    return h_init, h_add


# ---------- 主函数：每种背景场一行，每种方法一列 ----------

def plot_fig4_wells(data, rand_idx, rand_histories, output_dir, sample_id):
    """
    井位空间分布叠加图（正文 Fig. 12）

    行：
      行 0  叠加真实 K 场 (viridis，alpha=0.7)
      行 1  叠加真实浓度羽流 C_true (t=最后) (turbo，alpha=0.65)
      行 2  叠加共享初始 3 井同化后的浓度方差场 (YlOrRd，alpha=0.65)
      行 3  叠加 K→C 敏感性图（仅当 SENS_MAPS_PATH 可读时）

    列：Look-ahead / Greedy / LHS / Random（LHS、Random 为展示 trial）

    井位符号：
      ✕ 白色叉号  → 初始 3 口井
      ● 红色实心  → 新增井（按选入顺序编号）
      编号白色加黑底标注，避免与背景混淆
    """
    print(f"  Generating Fig 4: Well Distributions with Background Fields "
          f"for Sample {sample_id}...")
    apply_journal_style()

    n_init = Config.N_INIT_WELLS
    G      = Config.GRID_SIZE

    # ── 收集所有方法的井位 ──────────────────────────────────────
    plots = []
    for method, label in [('Active', 'Optimal'), ('Greedy', 'MaxVar'), ('LHS', 'LHS-20')]:
        if method in data and 'final_wells' in data[method]:
            wells = data[method]['final_wells']
            if hasattr(wells, 'tolist'):
                wells = wells.tolist()
            plots.append((label, wells))

    if rand_histories and len(rand_histories) > rand_idx:
        try:
            rand_wells = rand_histories[rand_idx][-1]['wells']
            plots.append((f'Random (Trial #{rand_idx})', rand_wells))
        except (IndexError, KeyError) as e:
            print(f"    [Warning] Could not get Random wells: {e}")

    if not plots:
        print("    [Skip] No well data available.")
        return

    n_cols = len(plots)

    # ── 提取背景场 ────────────────────────────────────────────
    K_true, C_true_last, prior_C_var = _get_background_fields(data, rand_idx, sample_id)

    S_map = load_sensitivity_map(sample_id)

    # 行标签：单行（等值线等说明放入图注）
    bg_rows = [
        ('True K field',     K_true,      'viridis', 0.70),
        ('True C plume',     C_true_last, 'turbo',   0.65),
        ('Prior C variance', prior_C_var, 'YlOrRd',  0.65),
    ]
    row_contours = {}
    if S_map is not None:
        bg_rows.append(('K→C sensitivity', S_map, 'GnBu', 0.80))
        row_contours[len(bg_rows) - 1] = C_true_last

    n_rows = len(bg_rows)
    # 画布与网格与 Figure 8（evaluate_es_mda.plot_field_comparison）完全一致
    fig = plt.figure(figsize=(n_cols * FIELD_PANEL_W, n_rows * FIG4_PANEL_H))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, **FIG4_GS_KW)
    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(gs[r, c])

    # 列标题：仅方法名，单行（空间统计指标见 spatial_stats.txt / Table 5）
    col_names = []
    for method_label, _ in plots:
        disp = method_label
        if method_label == 'LHS-20' and data.get('LHS', {}).get('_aligned_trial') is not None:
            disp = f"LHS-20 (Trial #{data['LHS']['_aligned_trial']})"
        col_names.append(disp)

    if Config.SHOW_FIG4_SUPTITLE:
        fig.suptitle(
            f'Sample {sample_id} — Well Distributions with Spatial Context',
            fontsize=JOURNAL_RC['figure.titlesize'], fontweight='bold'
        )

    for row_i, (bg_label, bg_arr, cmap, alpha) in enumerate(bg_rows):
        for col_i, (method_label, wells) in enumerate(plots):
            ax = axes[row_i, col_i]

            # 灰色底（无背景时也好看）
            ax.set_facecolor('#e8e8e8')

            # ── 背景场 ──
            im = _overlay_background(ax, bg_arr, cmap, alpha)
            if row_i in row_contours and row_contours[row_i] is not None:
                ax.contour(row_contours[row_i], levels=[0.5], colors='k',
                           linewidths=1.0, linestyles='--', zorder=5)

            # ── 井位（面板与 Figure 8 同为约 2.0 in，标记按比例缩小）──
            h_init, h_add = _scatter_wells(ax, wells, n_init,
                                           show_numbers=True,
                                           s_init=45, lw_init=1.3,
                                           s_add=26, lw_add=0.5,
                                           num_fontsize=5)

            # ── 轴格式 ──
            ax.set_xlim(0, G - 1)
            ax.set_ylim(0, G - 1)
            ax.set_aspect('equal')
            # 去掉 0–50 横纵刻度数值（保留四周边框）
            ax.set_xticks([])
            ax.set_yticks([])
            # 不翻转 y 轴：imshow origin='lower' 已经把 row=0 放在底部

            # 列标题（仅第 0 行）：单行方法名
            if row_i == 0:
                ax.set_title(col_names[col_i],
                             fontsize=JOURNAL_RC['axes.titlesize'],
                             fontweight='bold', pad=6)

            # 行标签（仅第 0 列）
            if col_i == 0:
                ax.set_ylabel(bg_label,
                              fontsize=JOURNAL_RC['axes.labelsize'],
                              labelpad=8)
            else:
                ax.set_ylabel('')

            ax.set_xlabel('')

            # ── colorbar ──
            # 与 Figure 8 相同：每个面板都用 fig.colorbar(ax=ax, fraction=0.046, pad=0.04)
            # 挤出 colorbar 空间；只显示最后一列，其余隐藏但保留占位，
            # 从而面板尺寸、位置与 Figure 8 逐一相同。
            mappable = im
            if mappable is None:
                mappable = plt.cm.ScalarMappable(cmap=cmap)
            cb = fig.colorbar(mappable, ax=ax, **CBAR_KW)
            cb.ax.tick_params(labelsize=FIELD_CBAR_TICK)
            if col_i != n_cols - 1 or im is None:
                cb.ax.set_visible(False)

            # ── 图例（仅左上角子图） ──
            if row_i == 0 and col_i == 0:
                handles = [h_init]
                labels  = ['Initial wells']
                if h_add is not None:
                    handles.append(h_add)
                    labels.append('Added wells')
                ax.legend(handles, labels,
                          loc='lower right',
                          fontsize=JOURNAL_RC['legend.fontsize'] - 3,
                          framealpha=0.75,
                          edgecolor='gray',
                          borderpad=0.3, handletextpad=0.3,
                          labelspacing=0.3)

    # 不调用 tight_layout：间距完全由 GridSpec(hspace/wspace) 控制，
    # 与 Figure 8 一致；外围空白由 savefig(bbox_inches='tight') 裁掉

    # ── 保存主图 ──
    out_path = os.path.join(output_dir, 'Fig4_Wells.png')
    plt.savefig(out_path, **SAVE_KW)
    plt.close()
    print(f"    [Done] Fig4_Wells.png  ({n_rows} rows × {n_cols} cols)")

    # ── 附加：每种背景单独输出（方便论文单图引用） ──
    _plot_fig4_single_background(
        data, rand_idx, rand_histories, output_dir, sample_id,
        K_true, C_true_last, prior_C_var, n_init
    )


def _plot_fig4_single_background(data, rand_idx, rand_histories,
                                  output_dir, sample_id,
                                  K_true, C_true_last, prior_C_var, n_init):
    """
    对每种背景分别输出一张 1行×N列 子图，文件名：
      Fig4a_Wells_K.png
      Fig4b_Wells_C.png
      Fig4c_Wells_PriorVar.png
    """
    G = Config.GRID_SIZE

    plots = []
    for method, label in [('Active', 'Optimal'), ('Greedy', 'MaxVar'), ('LHS', 'LHS-20')]:
        if method in data and 'final_wells' in data[method]:
            wells = data[method]['final_wells']
            if hasattr(wells, 'tolist'):
                wells = wells.tolist()
            plots.append((label, wells))
    if rand_histories and len(rand_histories) > rand_idx:
        try:
            rand_wells = rand_histories[rand_idx][-1]['wells']
            plots.append((f'Random (Trial #{rand_idx})', rand_wells))
        except (IndexError, KeyError):
            pass

    if not plots:
        return

    n_cols = len(plots)
    single_bgs = [
        ('Fig4a_Wells_K.png',
         'Well locations on true K field',      K_true,      'viridis', 0.72),
        ('Fig4b_Wells_C.png',
         'Well locations on true C plume\n(last timestep)',
                                                C_true_last, 'turbo',   0.68),
        ('Fig4c_Wells_PriorVar.png',
         'Well locations on prior concentration\nvariance (after 3 initial wells)',
                                                prior_C_var, 'YlOrRd',  0.68),
    ]
    S_map = load_sensitivity_map(sample_id)
    if S_map is not None:
        single_bgs.append(
            ('Fig4e_Wells_Sensitivity.png',
             'Well locations on K→C sensitivity\n(dashed: C=0.5 front, last timestep)',
             S_map, 'GnBu', 0.80))

    for fname, sup_title, bg_arr, cmap, alpha in single_bgs:
        fig, axes = plt.subplots(1, n_cols,
                                 figsize=(4.8 * n_cols, 4.6),
                                 squeeze=False)
        fig.suptitle(f'Sample {sample_id} — {sup_title}',
                     fontsize=JOURNAL_RC['figure.titlesize'],
                     fontweight='bold')

        for col_i, (method_label, wells) in enumerate(plots):
            ax = axes[0, col_i]
            ax.set_facecolor('#e8e8e8')

            im = _overlay_background(ax, bg_arr, cmap, alpha)
            if fname == 'Fig4e_Wells_Sensitivity.png' and C_true_last is not None:
                ax.contour(C_true_last, levels=[0.5], colors='k',
                           linewidths=1.0, linestyles='--', zorder=5)
            h_init, h_add = _scatter_wells(ax, wells, n_init,
                                           show_numbers=True)

            ax.set_xlim(0, G - 1)
            ax.set_ylim(0, G - 1)
            ax.set_aspect('equal')
            # 去掉 0–50 横纵刻度数值（保留四周边框）
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(method_label,
                         fontsize=JOURNAL_RC['axes.titlesize'],
                         fontweight='bold', pad=6)
            # 去掉 "Column" / "Row" 标签；Fig4e 在面板下方标注敏感性百分位
            ax.set_xlabel('')
            ax.set_ylabel('')
            if fname == 'Fig4e_Wells_Sensitivity.png' and S_map is not None:
                pct = well_sens_percentile(S_map, wells, n_init)
                ax.set_xlabel(f"Mean sensitivity percentile = {pct:.0f}%",
                              fontsize=JOURNAL_RC['axes.labelsize'], labelpad=6)

            # colorbar 仅最后列
            if col_i == n_cols - 1 and im is not None:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cb  = plt.colorbar(im, cax=cax)
                cb.ax.tick_params(
                    labelsize=JOURNAL_RC['xtick.labelsize'])

            # 图例仅第 0 列
            if col_i == 0:
                handles = [h_init]
                labels  = ['Initial wells']
                if h_add is not None:
                    handles.append(h_add)
                    labels.append('Added wells')
                ax.legend(handles, labels,
                          loc='lower right',
                          fontsize=JOURNAL_RC['legend.fontsize'] + 1,
                          framealpha=0.75, edgecolor='gray')

        plt.tight_layout(rect=[0, 0.02, 1, 0.94])
        fig.subplots_adjust(wspace=0.10)
        out_path = os.path.join(output_dir, fname)
        plt.savefig(out_path, **SAVE_KW)
        plt.close()
        print(f"    [Done] {fname}")


# ============================================================
# Fig5: 每个时刻 C 场对比
# ============================================================
def plot_fig5_c_per_time(data, rand_idx, output_dir, sample_id):
    print(f"  Generating Fig 5: C Field Per Time Step for Sample {sample_id}...")
    apply_journal_style()

    methods = ['Active', 'Greedy', 'LHS', 'Random']
    titles  = ['Optimal', 'MaxVar', 'LHS-20', f'Random (Trial #{rand_idx})']
    n_times = Config.N_TIMES

    valid_methods = []
    for method in methods:
        if method not in data:
            continue
        if 'C_true' not in data[method]:
            continue
        if _get_c_samples(data, method, rand_idx) is None:
            print(f"    [Skip] {method}: No C_samples")
            continue
        valid_methods.append(method)

    if not valid_methods:
        print("    [Skip] No valid data.")
        return

    for t in range(n_times):
        c_trues, c_means, c_errors, c_stds = [], [], [], []
        method_data = {}

        for method in valid_methods:
            d = data[method]
            C_true    = np.array(d['C_true'])
            C_true_t  = C_true[t]
            C_samples = _get_c_samples(data, method, rand_idx)
            C_st      = C_samples[:, t]  # (N, 51, 51)

            C_mean = np.mean(C_st, axis=0)
            C_std  = np.std(C_st,  axis=0)
            C_err  = np.abs(C_true_t - C_mean)

            c_trues.append(C_true_t);  c_means.append(C_mean)
            c_errors.append(C_err);    c_stds.append(C_std)
            method_data[method] = dict(C_true_t=C_true_t,
                                       C_mean=C_mean, C_std=C_std, C_error=C_err)

        c_min   = min(np.min(a) for a in c_trues + c_means)
        c_max   = max(np.max(a) for a in c_trues + c_means)
        err_max = max(np.max(a) for a in c_errors)
        std_max = max(np.max(a) for a in c_stds)

        n_rows  = len(valid_methods)
        fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle(f'Sample {sample_id} - C Field at Time {t + 1}',
                     fontsize=JOURNAL_RC['figure.titlesize'], fontweight='bold')

        for row, method in enumerate(valid_methods):
            title = titles[methods.index(method)]
            md = method_data[method]
            panels = [
                (md['C_true_t'], 'jet',     c_min,   c_max,   f"C True (t={t+1})"),
                (md['C_mean'],   'jet',     c_min,   c_max,   f"C Mean (t={t+1})"),
                (md['C_error'],  'hot',     0,       err_max, f"C Error (t={t+1})"),
                (md['C_std'],    'viridis', 0,       std_max, f"C Std (t={t+1})"),
            ]
            for col, (arr, cmap, vmin, vmax, col_title) in enumerate(panels):
                ax = axes[row, col]
                im = ax.imshow(arr, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
                add_colorbar(im, ax)
                if col == 0:
                    ax.set_ylabel(title, fontsize=JOURNAL_RC['axes.labelsize'], fontweight='bold')
                if row == 0:
                    ax.set_title(col_title, fontsize=JOURNAL_RC['axes.titlesize'])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.subplots_adjust(left=0.05, right=0.92, top=0.93, bottom=0.07)
        fname = f'Fig5_C_Time_{t + 1}.png'
        plt.savefig(os.path.join(output_dir, fname), **SAVE_KW)
        plt.close()
        print(f"    [Done] {fname}")


# ============================================================
# Fig6: 综合对比摘要图
# ============================================================
def plot_fig6_summary(data, rand_histories, output_dir, sample_id):
    print(f"  Generating Fig 6: Comparison Summary for Sample {sample_id}...")
    apply_journal_style()

    # 检查是否有 Arrival Time 数据
    has_arrival = False
    if 'Active' in data and 'history' in data['Active']:
        h0 = data['Active']['history']
        if not isinstance(h0, list):
            h0 = h0.tolist()
        if h0 and 'arrival_r2' in h0[0]:
            has_arrival = True

    rows, cols = 2, 3
    fig, axes  = plt.subplots(rows, cols, figsize=(15, 9))  # 更紧凑期刊尺寸
    fig.suptitle(f'Sample {sample_id} - Comparison Summary',
                 fontsize=JOURNAL_RC['figure.titlesize'], fontweight='bold')
    axes_flat = axes.flatten()

    # 基准 n_wells 序列
    base_history = None
    for method in ['Active', 'Greedy']:
        if method in data and 'history' in data[method]:
            base_history = data[method]['history']
            if not isinstance(base_history, list):
                base_history = base_history.tolist()
            break
    if base_history is None and rand_histories:
        base_history = rand_histories[0]
    if not base_history:
        print("    [Skip] No history data.")
        plt.close()
        return

    n_wells_base = [h['n_wells'] for h in base_history]

    def _plot_metric(ax, metric, active_style, greedy_style,
                     active_label, greedy_label, title, ylabel):
        # Active
        if 'Active' in data and 'history' in data['Active']:
            h = data['Active']['history']
            if not isinstance(h, list):
                h = h.tolist()
            ax.plot(n_wells_base, extract_metric(h, metric),
                    active_style, label=active_label, linewidth=2)
        # Greedy
        if 'Greedy' in data and 'history' in data['Greedy']:
            h = data['Greedy']['history']
            if not isinstance(h, list):
                h = h.tolist()
            ax.plot(n_wells_base, extract_metric(h, metric),
                    greedy_style, label=greedy_label, linewidth=2)
        # LHS-20: 非自适应基线，与 active/greedy 同为 single history
        if 'LHS' in data and 'history' in data['LHS']:
            h = data['LHS']['history']
            if not isinstance(h, list):
                h = h.tolist()
            lhs_x = [hh['n_wells'] for hh in h]
            lhs_s = METHOD_STYLE['LHS']
            ax.plot(lhs_x, extract_metric(h, metric),
                    color=lhs_s['color'], marker=lhs_s['marker'],
                    linestyle=lhs_s['linestyle'],
                    label=lhs_s['short_label'], linewidth=2)
        # Random (mean ± std)
        if rand_histories:
            x_rnd, y_mean, y_std = extract_random_metrics(rand_histories, metric)
            if x_rnd is not None:
                ax.plot(x_rnd, y_mean, 'r--', label='Random (Mean)', linewidth=2)
                ax.fill_between(x_rnd, y_mean - y_std, y_mean + y_std,
                                color='red', alpha=0.1)
        ax.set_title(title, fontsize=JOURNAL_RC['axes.titlesize'])
        ax.set_xlabel('Number of Wells')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # ── 子图 0: K R² ──
    _plot_metric(axes_flat[0], 'K_r2', 'g-s', 'b-o',
                 'Optimal', 'Greedy (MaxVar)',
                 'Hydraulic Conductivity (K) R² Score', 'R²')
    axes_flat[0].legend()

    # ── 子图 1: C R² ──
    _plot_metric(axes_flat[1], 'C_r2', 'g-s', 'b-o',
                 'Optimal', 'Greedy',
                 'Concentration (C) R² Score', 'R²')

    # ── 子图 2: Arrival Time R2（可选）或留空 ──
    next_subplot_idx = 2
    if has_arrival:
        _plot_metric(axes_flat[2], 'arrival_r2', 'g-s', 'b-o',
                     'Optimal', 'Greedy',
                     'Arrival Time R² Score', 'R²')
        next_subplot_idx = 3
    else:
        axes_flat[2].axis('off')
        next_subplot_idx = 3

    # ── 子图 3: C Variance (Log) ──
    ax_var = axes_flat[3]
    if 'Active' in data and 'history' in data['Active']:
        h = data['Active']['history']
        if not isinstance(h, list):
            h = h.tolist()
        ax_var.semilogy(n_wells_base, extract_metric(h, 'C_var'),
                        'g-s', label='Optimal', linewidth=2)
    if 'Greedy' in data and 'history' in data['Greedy']:
        h = data['Greedy']['history']
        if not isinstance(h, list):
            h = h.tolist()
        ax_var.semilogy(n_wells_base, extract_metric(h, 'C_var'),
                        'b-o', label='Greedy', linewidth=2)
    if 'LHS' in data and 'history' in data['LHS']:
        h = data['LHS']['history']
        if not isinstance(h, list):
            h = h.tolist()
        lhs_x = [hh['n_wells'] for hh in h]
        lhs_s = METHOD_STYLE['LHS']
        ax_var.semilogy(lhs_x, extract_metric(h, 'C_var'),
                        color=lhs_s['color'], marker=lhs_s['marker'],
                        linestyle=lhs_s['linestyle'],
                        label=lhs_s['short_label'], linewidth=2)
    if rand_histories:
        x_rnd, y_mean, _ = extract_random_metrics(rand_histories, 'C_var')
        if x_rnd is not None:
            ax_var.semilogy(x_rnd, y_mean, 'r--', label='Random', linewidth=2)
    ax_var.set_title('Concentration Variance (Uncertainty)', fontsize=JOURNAL_RC['axes.titlesize'])
    ax_var.set_xlabel('Number of Wells')
    ax_var.set_ylabel('Variance (Log Scale)')
    ax_var.grid(True, alpha=0.3)
    ax_var.legend()

    # ── 子图 4: 最终 C R² 柱状图 ──
    ax_bar = axes_flat[4]
    bar_methods, bar_vals, bar_errs, bar_colors = [], [], [], []

    if 'Active' in data and 'history' in data['Active']:
        h = data['Active']['history']
        if not isinstance(h, list):
            h = h.tolist()
        bar_methods.append('Optimal')
        bar_vals.append(h[-1].get('C_r2', 0))
        bar_errs.append(0)
        bar_colors.append(METHOD_STYLE['Active']['color'])

    if 'Greedy' in data and 'history' in data['Greedy']:
        h = data['Greedy']['history']
        if not isinstance(h, list):
            h = h.tolist()
        bar_methods.append('Greedy')
        bar_vals.append(h[-1].get('C_r2', 0))
        bar_errs.append(0)
        bar_colors.append(METHOD_STYLE['Greedy']['color'])

    if 'LHS' in data and 'history' in data['LHS']:
        h = data['LHS']['history']
        if not isinstance(h, list):
            h = h.tolist()
        bar_methods.append('LHS-20')
        bar_vals.append(h[-1].get('C_r2', 0))
        bar_errs.append(0)
        bar_colors.append(METHOD_STYLE['LHS']['color'])

    if rand_histories:
        _, y_mean, y_std = extract_random_metrics(rand_histories, 'C_r2')
        if y_mean is not None:
            bar_methods.append('Random')
            bar_vals.append(float(y_mean[-1]))
            bar_errs.append(float(y_std[-1]))
            bar_colors.append(METHOD_STYLE['Random']['color'])

    if bar_methods:
        ax_bar.bar(bar_methods, bar_vals, yerr=bar_errs,
                   color=bar_colors, alpha=0.7, capsize=10)
        ax_bar.set_ylim(0, 1.05)
    ax_bar.set_title('Final C Field R² Comparison', fontsize=JOURNAL_RC['axes.titlesize'])
    ax_bar.set_ylabel('R²')
    ax_bar.grid(True, axis='y', alpha=0.3)

    # ── 子图 5: 井位叠加图 ──
    ax_w = axes_flat[5]
    ax_w.set_xlim(0, Config.GRID_SIZE)
    ax_w.set_ylim(0, Config.GRID_SIZE)
    ax_w.set_xlabel('Column')
    ax_w.set_ylabel('Row')
    ax_w.grid(True, alpha=0.3)

    init_wells = []
    for method in ['Active', 'Greedy']:
        if method in data and 'history' in data[method]:
            h = data[method]['history']
            if not isinstance(h, list):
                h = h.tolist()
            if h and 'wells' in h[0]:
                init_wells = h[0]['wells']
                break

    if init_wells:
        ax_w.scatter([w[1] for w in init_wells], [w[0] for w in init_wells],
                     c='black', marker='x', s=100, label='Initial', zorder=10)

    init_set = {tuple(w) for w in init_wells}

    colors_map = {
        'Active': (METHOD_STYLE['Active']['color'], 'o', 'Optimal'),
        'Greedy': (METHOD_STYLE['Greedy']['color'], '^', 'Greedy'),
        'LHS':    (METHOD_STYLE['LHS']['color'],    'D', 'LHS-20'),
    }
    for method, (color, marker, label) in colors_map.items():
        if method in data and 'final_wells' in data[method]:
            fw = data[method]['final_wells']
            if hasattr(fw, 'tolist'):
                fw = fw.tolist()
            new = [w for w in fw if tuple(w) not in init_set]
            if new:
                ax_w.scatter([w[1] for w in new], [w[0] for w in new],
                             c=color, marker=marker, label=label, alpha=0.7)

    ax_w.set_title('Final Well Locations', fontsize=JOURNAL_RC['axes.titlesize'])
    ax_w.legend(loc='upper right', fontsize=JOURNAL_RC['legend.fontsize'])
    ax_w.invert_yaxis()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'Fig6_Comparison_Summary.png'), **SAVE_KW)
    plt.close()
    print("    [Done] Fig6_Comparison_Summary.png")



def write_spatial_stats_report(stats_dict, out_dir, sample_id):
    """
    将空间分布统计量格式化输出到终端并保存为 spatial_stats.txt。
    使用原版 compute_spatial_stats 的字段名：
      ann_distance, clark_evans, coverage_rate, boundary_rate,
      plume_inside, sens_percentile
    同时输出"论文 4.3.3 节可用结论"文字块。
    """
    # 字段名 → (显示名称, 英文解读, 中文解读)
    metric_info = {
        'ann_distance':   ('ANN Distance (grid units)',
                           'higher=more dispersed',  '↑ 越大越分散'),
        'clark_evans':    ('Clark-Evans Ratio R',
                           'R<1=clustered, R=1=random, R>1=dispersed',
                           'R<1 聚集 / R=1 随机 / R>1 分散'),
        'coverage_rate':  ('Coverage Rate',
                           'higher=wider coverage of study area',
                           '↑ 越大覆盖面积越广'),
        'boundary_rate':  ('Boundary Rate',
                           'fraction of added wells in boundary zone',
                           '边界区井位比例'),
        'plume_inside':   ('Plume Inside Rate',
                           'fraction of added wells inside high-variance plume zone',
                           '落入高方差羽流区的比例'),
        'sens_percentile': ('Sensitivity Percentile (%)',
                           'higher=wells placed at more sensitive locations; uniform ~50%',
                           '↑ 越大越倾向高敏感区布井；均匀布井期望约 50%'),
    }
    metrics = list(metric_info.keys())
    methods = list(stats_dict.keys())

    col_m   = 26
    col_val = 14

    lines = []
    lines.append("=" * 90)
    lines.append(f"  Spatial Distribution Statistics — Sample {sample_id}")
    lines.append("  Based on added wells only (initial 3 wells excluded)")
    lines.append("=" * 90)

    # ── Table 1：绝对值 ────────────────────────────────────────
    lines.append("")
    lines.append("Table 1.  Absolute values of spatial distribution metrics")
    lines.append("-" * 90)
    header = f"  {'Metric':<{col_m}}" + "".join(
        f"{m:>{col_val}}" for m in methods)
    lines.append(header)
    lines.append("  " + "-" * (col_m + col_val * len(methods) + 2))

    for mk in metrics:
        label = metric_info[mk][0]
        row = f"  {label:<{col_m}}"
        for m in methods:
            val = stats_dict.get(m, {}).get(mk, np.nan)
            # plume_inside can be None when no plume_mask provided
            if val is None:
                val = np.nan
            try:
                row += f"{float(val):>{col_val}.4f}" if not np.isnan(float(val)) else f"{'N/A':>{col_val}}"
            except (TypeError, ValueError):
                row += f"{'N/A':>{col_val}}"
        lines.append(row)

    # ── Table 2：相对差异（以 Active/Optimal 为基准）─────────────
    lines.append("")
    lines.append("Table 2.  Relative difference vs. Optimal (Active) method  [%]")
    lines.append("  Positive = baseline HIGHER than Optimal (worse clustering = more dispersed)")
    lines.append("-" * 90)
    baselines = [m for m in methods if m != 'Optimal']
    if 'Optimal' in stats_dict and baselines:
        hdr2 = f"  {'Metric':<{col_m}}" + "".join(
            f"{m:>{col_val}}" for m in baselines)
        lines.append(hdr2)
        lines.append("  " + "-" * (col_m + col_val * len(baselines) + 2))
        for mk in metrics:
            label = metric_info[mk][0]
            row = f"  {label:<{col_m}}"
            raw = stats_dict['Optimal'].get(mk, np.nan)
            base_val = float(raw) if raw is not None else np.nan
            for m in baselines:
                cmp_val = stats_dict.get(m, {}).get(mk, np.nan)
                raw_c = stats_dict.get(m, {}).get(mk, np.nan)
                cmp_val = float(raw_c) if raw_c is not None else np.nan
                if np.isnan(base_val) or np.isnan(cmp_val) or base_val == 0:
                    row += f"{'N/A':>{col_val}}"
                else:
                    pct = (cmp_val - base_val) / abs(base_val) * 100
                    row += f"{pct:>+{col_val}.1f}%"[:-1] + "%"
                    # pad to col_val
            lines.append(row)

    # ── 解读说明 ──────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 90)
    lines.append("  Metric Interpretation")
    lines.append("=" * 90)
    for mk, (full_label, interp_en, interp_zh) in metric_info.items():
        lines.append(f"  {full_label}")
        lines.append(f"    EN: {interp_en}")
        lines.append(f"    ZH: {interp_zh}")
        lines.append("")

    # ── 自动生成关键结论（供论文 4.3.3 节参考）────────────────
    lines.append("=" * 90)
    lines.append("  Key Findings for Manuscript (Section 4.3.3)")
    lines.append("=" * 90)

    if 'Optimal' in stats_dict and 'MaxVar' in stats_dict:
        opt = stats_dict['Optimal']
        grd = stats_dict['MaxVar']

        # Use original field names from compute_spatial_stats()
        ce_opt = float(opt.get('clark_evans',  np.nan) or np.nan)
        ce_grd = float(grd.get('clark_evans',  np.nan) or np.nan)
        cv_opt = float(opt.get('coverage_rate',np.nan) or np.nan)
        cv_grd = float(grd.get('coverage_rate',np.nan) or np.nan)
        br_opt = float(opt.get('boundary_rate',np.nan) or np.nan)
        br_grd = float(grd.get('boundary_rate',np.nan) or np.nan)
        mn_opt = float(opt.get('ann_distance', np.nan) or np.nan)
        mn_grd = float(grd.get('ann_distance', np.nan) or np.nan)
        sp_opt = float(opt.get('sens_percentile',np.nan) or np.nan)
        sp_grd = float(grd.get('sens_percentile',np.nan) or np.nan)
        pi_opt = opt.get('plume_inside')
        pi_grd = grd.get('plume_inside')

        lines.append("")
        lines.append("  [Clustering: Clark-Evans Ratio]")
        if not (np.isnan(ce_opt) or np.isnan(ce_grd)):
            lines.append(
                f"  Optimal R={ce_opt:.3f} vs MaxVar R={ce_grd:.3f} "
                f"({'Greedy more clustered' if ce_grd < ce_opt else 'Optimal more clustered'})")
            lines.append(
                f"  Greedy clustering is "
                f"{abs(ce_opt - ce_grd) / max(ce_opt, 1e-9) * 100:.1f}% "
                f"{'stronger' if ce_grd < ce_opt else 'weaker'} than Optimal")

        lines.append("")
        lines.append("  [Coverage Rate]")
        if not (np.isnan(cv_opt) or np.isnan(cv_grd)):
            pct = (cv_opt - cv_grd) / max(cv_grd, 1e-9) * 100
            lines.append(
                f"  Optimal coverage={cv_opt*100:.1f}% vs MaxVar coverage={cv_grd*100:.1f}%")
            lines.append(
                f"  Optimal covers {pct:+.1f}% more of the study area than MaxVar")

        lines.append("")
        lines.append("  [Spread: Average Nearest-Neighbor Distance]")
        if not (np.isnan(mn_opt) or np.isnan(mn_grd)):
            lines.append(
                f"  Optimal ANN={mn_opt:.3f} vs MaxVar ANN={mn_grd:.3f} grid units")
            pct2 = (mn_opt - mn_grd) / max(mn_grd, 1e-9) * 100
            lines.append(
                f"  Optimal inter-well spacing is {pct2:+.1f}% "
                f"{'larger' if pct2 > 0 else 'smaller'} than MaxVar "
                f"→ {'more dispersed placement' if pct2 > 0 else 'tighter clustering'}")

        lines.append("")
        lines.append("  [Boundary Rate]")
        if not (np.isnan(br_opt) or np.isnan(br_grd)):
            lines.append(
                f"  Optimal boundary_rate={br_opt*100:.1f}% vs MaxVar={br_grd*100:.1f}%")

        if pi_opt is not None and pi_grd is not None:
            lines.append("")
            lines.append("  [Plume Coverage]")
            lines.append(
                f"  Optimal plume_inside={pi_opt*100:.1f}% vs MaxVar={pi_grd*100:.1f}%")
            lines.append(
                f"  → Optimal places {(pi_opt - pi_grd)*100:+.1f}% "
                f"{'more' if pi_opt > pi_grd else 'fewer'} wells inside the high-variance zone")

        lines.append("")
        lines.append("  [Sensitivity Percentile (%, uniform expectation ~50%)]")
        if not (np.isnan(sp_opt) or np.isnan(sp_grd)):
            lines.append(
                f"  Optimal sens_pct={sp_opt:.1f}% vs MaxVar={sp_grd:.1f}%")
            lines.append(
                f"  → Optimal wells are on average at {sp_opt - sp_grd:+.1f}% "
                f"higher sensitivity locations")

    lines.append("=" * 90)

    text = "\n".join(lines)
    print("\n" + text)

    path = os.path.join(out_dir, 'spatial_stats.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text + "\n")
    print(f"\n  [Saved] {path}")
    return text


def plot_spatial_stats_chart(stats_dict, out_dir, sample_id):
    """
    将五项空间统计指标绘制为雷达图 + 条形图组合，
    输出为 Fig4d_Spatial_Stats.png。

    布局：1 行 × 2 列
      左图：雷达图（Clark-Evans R / MNND归一化 / 1-QV归一化 / CH归一化 / 1-CDC归一化）
            各指标归一化到 [0,1]，值越大越"理想"（分散/均匀/覆盖广/居中）
      右图：条形图并排，展示各方法的 Clark-Evans R 和 Quadrat Variance 原始值
    """
    apply_journal_style()

    methods  = list(stats_dict.keys())
    colors   = {'Optimal': METHOD_STYLE['Active']['color'],
                'MaxVar':  METHOD_STYLE['Greedy']['color'],
                'LHS-20':  METHOD_STYLE['LHS']['color'],
                'Random':  METHOD_STYLE['Random']['color']}
    # 为 Random (Trial #N) 也映射颜色
    col_map = {}
    for m in methods:
        for key, c in colors.items():
            if key in m:
                col_map[m] = c
                break
        if m not in col_map:
            col_map[m] = '#aaaaaa'

    # ── 条形图：Clark-Evans R + Quadrat Variance ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    bar_metrics = [
        ('clark_evans',   'Clark-Evans Ratio R\n(R<1=clustered, R>1=dispersed)'),
        ('coverage_rate', 'Coverage Rate\n(higher = wider spatial coverage)'),
    ]
    for ax, (mk, ylabel) in zip(axes, bar_metrics):
        xs  = np.arange(len(methods))
        vals = [stats_dict[m].get(mk, np.nan) for m in methods]
        cols = [col_map[m] for m in methods]
        bars = ax.bar(xs, vals, color=cols, alpha=0.80, width=0.5,
                      edgecolor='white', linewidth=0.6)
        # 数值标注
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.02,
                        f'{v:.3f}', ha='center', va='bottom',
                        fontsize=JOURNAL_RC['xtick.labelsize'])
        ax.set_xticks(xs)
        ax.set_xticklabels(methods, rotation=12, ha='right',
                           fontsize=JOURNAL_RC['xtick.labelsize'])
        ax.set_ylabel(ylabel, fontsize=JOURNAL_RC['axes.labelsize'])
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)

        # 在 Clark-Evans 图上加参考线 R=1
        if mk == 'clark_evans':
            ax.axhline(y=1.0, color='gray', linestyle='--',
                       linewidth=1.0, alpha=0.7, label='R=1 (random)')
            ax.legend(fontsize=JOURNAL_RC['legend.fontsize'])

    fig.suptitle(
        f'Sample {sample_id} — Spatial Distribution Statistics\n'
        f'(added wells only, initial {Config.N_INIT_WELLS} wells excluded)',
        fontsize=JOURNAL_RC['figure.titlesize'], fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    path = os.path.join(out_dir, 'Fig4d_Spatial_Stats.png')
    plt.savefig(path, **SAVE_KW)
    plt.close()
    print(f"    [Done] Fig4d_Spatial_Stats.png")


# ============================================================
# 文本报告
# ============================================================
def generate_report(data, rand_histories, output_dir, sample_id):
    print(f"  Generating report for Sample {sample_id}...")
    path = os.path.join(output_dir, 'final_report.txt')

    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 55 + "\n")
        f.write(f"  Well-Placement Comparison Report - Sample {sample_id}\n")
        f.write("=" * 55 + "\n\n")

        f.write("1. Final Metrics Summary:\n")
        f.write(f"  {'Method':<15} {'K R²':<10} {'C R²':<10} {'C Var':<12}\n")
        f.write("  " + "-" * 48 + "\n")

        for method, label in [('Active', 'Optimal'), ('Greedy', 'Greedy'), ('LHS', 'LHS-20')]:
            if method in data and 'history' in data[method]:
                h = data[method]['history']
                if not isinstance(h, list):
                    h = h.tolist()
                if h:
                    last = h[-1]
                    f.write(f"  {label:<15} {last.get('K_r2', 0):<10.4f} "
                            f"{last.get('C_r2', 0):<10.4f} "
                            f"{last.get('C_var', 0):<12.4f}\n")

        if rand_histories:
            _, k_mean, _ = extract_random_metrics(rand_histories, 'K_r2')
            _, c_mean, _ = extract_random_metrics(rand_histories, 'C_r2')
            _, v_mean, _ = extract_random_metrics(rand_histories, 'C_var')
            if k_mean is not None:
                f.write(f"  {'Random(Avg)':<15} {k_mean[-1]:<10.4f} "
                        f"{c_mean[-1]:<10.4f} {v_mean[-1]:<12.4f}\n")

        f.write("\n2. Improvement (Final − Initial C_R²):\n")
        for method, label in [('Active', 'Optimal'), ('Greedy', 'Greedy'), ('LHS', 'LHS-20')]:
            if method in data and 'history' in data[method]:
                h = data[method]['history']
                if not isinstance(h, list):
                    h = h.tolist()
                if len(h) > 1:
                    imp = h[-1].get('C_r2', 0) - h[0].get('C_r2', 0)
                    f.write(f"  {label} Improvement: +{imp:.4f}\n")

        f.write("\n3. Final Well Coordinates (Row, Col) → (X, Y):\n")
        for method, label in [('Active', 'Optimal'), ('Greedy', 'Greedy'), ('LHS', 'LHS-20')]:
            if method in data and 'final_wells' in data[method]:
                fw = data[method]['final_wells']
                if hasattr(fw, 'tolist'):
                    fw = fw.tolist()
                f.write(f"  {label} Wells:\n")
                for w in fw:
                    x, y = index_to_coord(w[0], w[1])
                    f.write(f"    ({w[0]:>2d}, {w[1]:>2d})  →  "
                            f"({x:>6d}, {y:>6d})\n")

        # ── 4. 空间分布统计量 ──────────────────────────────────
        f.write("\n4. Spatial Distribution Statistics (added wells only):\n")
        f.write(f"  {'Method':<26} {'ANN':>8} {'CE_R':>8} "
                f"{'COV':>8} {'BDR':>8} {'PIN':>8} {'SEN':>8}\n")
        f.write("  " + "-" * 70 + "\n")

        rand_idx_rep = 0
        if rand_histories:
            rand_idx_rep = select_random_trial(data.get('Random', {}),
                                               sample_id)
        # Use the existing compute_spatial_stats_all_methods (original function)
        K_true, C_true_last, prior_C_var = _get_background_fields(data, rand_idx_rep, sample_id)
        S_map_rep = load_sensitivity_map(sample_id)
        stats_dict = compute_spatial_stats_all_methods(
            data, rand_idx_rep, rand_histories,
            Config.N_INIT_WELLS, Config.GRID_SIZE, prior_C_var,
            S_map=S_map_rep)

        def _fmt(v, w=8, d=3):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return f"{'N/A':>{w}}"
            try:
                return f"{float(v):>{w}.{d}f}"
            except (TypeError, ValueError):
                return f"{'N/A':>{w}}"

        for label, stats in stats_dict.items():
            ann  = stats.get('ann_distance',  np.nan)
            cer  = stats.get('clark_evans',   np.nan)
            cov  = stats.get('coverage_rate', np.nan)
            bdr  = stats.get('boundary_rate', np.nan)
            pin  = stats.get('plume_inside',  None)
            sen  = stats.get('sens_percentile', np.nan)
            f.write(f"  {label:<26}"
                    f"{_fmt(ann)}{_fmt(cer)}{_fmt(cov)}"
                    f"{_fmt(bdr)}{_fmt(pin)}{_fmt(sen, d=1)}\n")

        f.write("\n  Legend:\n"
                "    ANN  = Average Nearest-Neighbor Distance (grid units, higher=dispersed)\n"
                "    CE_R = Clark-Evans Ratio (R<1 clustered, R=1 random, R>1 dispersed)\n"
                "    COV  = Coverage Rate (fraction of grid covered)\n"
                "    BDR  = Boundary Rate (fraction of wells in edge zone)\n"
                "    PIN  = Plume Inside Rate (fraction in high-variance zone)\n"
                "    SEN  = Sensitivity Percentile (%) of added wells on K→C sensitivity field (uniform ~50%)\n")

    print(f"    [Done] {path}")


# ============================================================
# 处理单个样本
# ============================================================
def process_single_sample(sample_id):
    print(f"\n{'=' * 60}")
    print(f"Processing Sample {sample_id} "
          f"({Config.get_sample_folder_name(sample_id)})")
    print(f"{'=' * 60}")

    output_dir = Config.get_output_dir(sample_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output directory: {output_dir}")

    data = load_data_for_sample(sample_id)
    if not data:
        print(f"  [Error] No data found for Sample {sample_id}. Skipping...")
        return False

    check_data_completeness(data, sample_id)
    rand_histories = get_random_histories(data)

    rand_idx = 0
    if 'Random' in data and rand_histories:
        rand_idx = select_random_trial(data['Random'], sample_id)
    # LHS / Random 的完整后验集合由 run_100.py 为 PLOT_SAMPLE_IDS 中的样本保存；
    # 展示 trial 均由 default_rng(sample_id) 选出，LHS 与 Random 编号相同
    # （记录在 lhs.npz 的 _aligned_trial 字段中）。

    try:
        plot_fig1_k_comparison(data, rand_idx, output_dir, sample_id)
        plot_fig2_c_weighted(data, rand_idx, output_dir, sample_id)
        plot_fig3_curves(data, rand_histories, output_dir, sample_id)
        plot_fig4_wells(data, rand_idx, rand_histories,
                        output_dir, sample_id)
        plot_fig5_c_per_time(data, rand_idx, output_dir, sample_id)
        plot_fig6_summary(data, rand_histories, output_dir, sample_id)
        generate_report(data, rand_histories, output_dir, sample_id)

        # ── 空间分布定量统计（4.3.3 节）：与 Fig.4 使用相同的井位 ──
        print("  [空间统计] 计算井位的空间分布量化指标...")
        _, _, prior_C_var_pss = _get_background_fields(data, rand_idx, sample_id)
        S_map_pss = load_sensitivity_map(sample_id)
        stats_dict = compute_spatial_stats_all_methods(
            data, rand_idx, rand_histories,
            Config.N_INIT_WELLS, Config.GRID_SIZE, prior_C_var_pss,
            S_map=S_map_pss)
        write_spatial_stats_report(stats_dict, output_dir, sample_id)
        plot_spatial_stats_chart(stats_dict, output_dir, sample_id)

        print(f"\n  [Success] Sample {sample_id} completed!")
        return True
    except Exception as e:
        print(f"  [Error] Failed to process Sample {sample_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# 主函数
# ============================================================
def main():
    apply_journal_style()

    print("=" * 60)
    print("  Batch Processing: Integrated Figure Generation")
    print("=" * 60)
    print(f"Base directory : {Config.BASE_DIR}")
    print(f"Output root    : {Config.OUTPUT_ROOT}")
    print(f"Sample IDs     : {Config.SAMPLE_IDS}")

    os.makedirs(Config.OUTPUT_ROOT, exist_ok=True)

    success_count  = 0
    failed_samples = []

    for idx, sample_id in enumerate(Config.SAMPLE_IDS, start=1):
        print(f"\n[{idx}/{len(Config.SAMPLE_IDS)}] ", end="")
        if process_single_sample(sample_id):
            success_count += 1
        else:
            failed_samples.append(sample_id)

    print("\n" + "=" * 60)
    print("  Batch Processing Summary")
    print("=" * 60)
    print(f"Total    : {len(Config.SAMPLE_IDS)}")
    print(f"Success  : {success_count}")
    print(f"Failed   : {len(failed_samples)}")
    if failed_samples:
        print(f"Failed IDs: {failed_samples}")
    print(f"\nOutputs → {Config.OUTPUT_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()