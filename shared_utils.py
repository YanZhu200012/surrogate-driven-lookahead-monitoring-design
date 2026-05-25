"""
shared_utils.py
共享工具模块（支持 100 个测试场）

修改说明：
  1. ALPHA_LIST = [12.0]*12，Σ(1/αᵢ) = 1.0 ✓
  2. VIRTUAL_ALPHA_LIST = [4.0]*4，Σ(1/αᵢ) = 1.0 ✓
  3. VARIANCE_PERCENTILE = 30，保留前 70% 候选位置
  4. compute_weighted_C_variance() 新增返回 mean_var / mean_std
       mean_var = total_var / GRID_SIZE²  （平均每格点方差）
       mean_std = sqrt(mean_var)          （平均每格点标准差，论文展示用）
  5. evaluate_prediction() 新增 C_mean_std 字段
  6. 保留 check_variance_reduction() 定义但不在业务流程中调用
"""

import torch
import numpy as np
import scipy.io as sio
from scipy.stats import qmc
from scipy.linalg import inv
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity as ssim
import os

from train_vae import VAE
from train_f1 import FNO2d as FNO2d_KH
from train_f2 import FNO2d_KC

class BaseConfig:
    """基础配置参数"""
    SEED   = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GRID_SIZE = 51

    # 坐标参数
    GRID_ORIGIN  = -4900
    GRID_SPACING = 196

    # 模型路径
    VAE_PATH    = "./vae_results/best_vae.pth"
    FNO_KH_PATH = "./fno_kh_results/best_fno_kh.pth"
    FNO_KC_PATH = "./fno_kc_results/best_fno_kc.pth"

    # 训练数据路径
    K_DATA_PATH  = './data/DataK_6000_51_51.mat'
    H_DATA_PATH  = './data/DataH_6000_5_51_51.mat'
    KC_NORM_PATH = "./fno_kc_results/normalization_params.npz"

    # 测试数据路径
    TEST_DATA_PATH = './test_synthetic_data/synthetic_test_100.npz'
    N_TEST_SAMPLES = 100

    # 实验通用参数
    N_ENSEMBLE    = 500
    INITIAL_WELLS = 3
    TARGET_WELLS  = 20

    # ES-MDA 参数
    OBS_NOISE_STD = 0.3
    SIGMA_F1      = 0.0

    # 主 ES-MDA 膨胀因子（12次均值划分，Σ(1/αᵢ) = 1.0 ✓）
    ALPHA_LIST = [12.0] * 12

    # 浓度场参数
    N_TIME_STEPS = 5
    TIME_WEIGHTS = [0.1, 0.15, 0.2, 0.25, 0.3]

    # Look-ahead 参数
    # 虚拟 ES-MDA 膨胀因子（4次均值划分，Σ(1/αᵢ) = 1.0 ✓）
    VIRTUAL_ALPHA_LIST   = [4.0] * 4
    USE_CANDIDATE_FILTER = True
    MAX_CANDIDATES       = 500
    VARIANCE_PERCENTILE  = 30   # 排除最低 30%，保留前 70% 候选
    MIN_WELL_SPACING     = 2


# ================= 工具函数 =================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def index_to_coord(row, col, config):
    x = config.GRID_ORIGIN + col * config.GRID_SPACING
    y = config.GRID_ORIGIN + row * config.GRID_SPACING
    return (x, y)


def format_location(row, col, config):
    x, y = index_to_coord(row, col, config)
    return f"({row},{col}) -> ({x},{y})m"


# ================= 模型加载 =================
def load_all_models(config):
    print(f"[Info] Loading models on {config.DEVICE}...")
    vae = VAE(latent_dim=128).to(config.DEVICE)
    vae.load_state_dict(torch.load(config.VAE_PATH, map_location=config.DEVICE))
    vae.eval()

    h_mat  = sio.loadmat(config.H_DATA_PATH)
    h_key  = [k for k in h_mat.keys() if not k.startswith('__')][0]
    h_data = h_mat[h_key].astype(np.float32)
    h_mean, h_std = float(np.mean(h_data)), float(np.std(h_data))

    fno_kh = FNO2d_KH(modes1=12, modes2=12, width=64, in_channels=1, out_channels=5,
                       h_mean=h_mean, h_std=h_std).to(config.DEVICE)
    fno_kh.load_state_dict(torch.load(config.FNO_KH_PATH, map_location=config.DEVICE))
    fno_kh.eval()

    kc_norm = np.load(config.KC_NORM_PATH)
    c_mean, c_std = float(kc_norm['c_mean']), float(kc_norm['c_std'])

    fno_kc = FNO2d_KC(modes1=12, modes2=12, width=64, in_channels=1, out_channels=5,
                       c_mean=c_mean, c_std=c_std).to(config.DEVICE)
    fno_kc.load_state_dict(torch.load(config.FNO_KC_PATH, map_location=config.DEVICE))
    fno_kc.eval()

    return vae, fno_kh, fno_kc


# ================= 数据准备 =================
def load_test_data(config):
    """
    加载全部测试数据，并将 sigma_f1 写入 config。
    返回: K_all, H_all, C_all, k_train_data
    """
    print("[Info] Loading test data...")
    test_data = np.load(config.TEST_DATA_PATH)
    K_all = test_data['K_test']
    H_all = test_data['H_test']
    C_all = test_data['C_test']
    print(f"  K_all: {K_all.shape}, H_all: {H_all.shape}, C_all: {C_all.shape}")

    if 'sigma_f1' in test_data:
        config.SIGMA_F1 = float(test_data['sigma_f1'])
        print(f"  sigma_f1 = {config.SIGMA_F1:.5f}  (f1 代理误差，已写入 config)")
    else:
        print("  [Warning] sigma_f1 未找到，使用默认值 0.0")

    k_mat        = sio.loadmat(config.K_DATA_PATH)
    k_key        = [k for k in k_mat.keys() if not k.startswith('__')][0]
    k_train_data = k_mat[k_key].astype(np.float32)
    print(f"  k_train_data: {k_train_data.shape} (for prior generation)")

    return K_all, H_all, C_all, k_train_data


def get_test_sample(K_all, H_all, C_all, sample_idx):
    return K_all[sample_idx], H_all[sample_idx], C_all[sample_idx]


def prepare_synthetic_data(config, fno_kh=None, fno_kc=None, sample_idx=0,
                            K_all=None, H_all=None, C_all=None, k_train_data=None):
    """向后兼容接口"""
    if K_all is not None:
        K_true, H_true, C_true = get_test_sample(K_all, H_all, C_all, sample_idx)
        print(f"[Info] Test sample #{sample_idx}: K={K_true.shape}, "
              f"H={H_true.shape}, C={C_true.shape}")
        return K_true, H_true, C_true, k_train_data

    print("[Info] Preparing synthetic ground truth (legacy mode)...")
    k_mat  = sio.loadmat(config.K_DATA_PATH)
    k_key  = [k for k in k_mat.keys() if not k.startswith('__')][0]
    k_data = k_mat[k_key].astype(np.float32)

    TRUE_SAMPLE_INDEX = getattr(config, 'TRUE_SAMPLE_INDEX', 0)
    K_true   = k_data[TRUE_SAMPLE_INDEX]
    K_tensor = torch.tensor(K_true[np.newaxis, np.newaxis, :, :],
                            dtype=torch.float32).to(config.DEVICE)
    with torch.no_grad():
        H_true = fno_kh(K_tensor, return_normalized=False).cpu().numpy().squeeze(0)
        C_true = fno_kc(K_tensor, return_normalized=False).cpu().numpy().squeeze(0)

    print(f"  K_true: {K_true.shape}, H_true: {H_true.shape}, C_true: {C_true.shape}")
    return K_true, H_true, C_true, k_data


# ================= 井位与观测 =================
def init_wells_lhs(n_wells, grid_size, seed):
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    samples = sampler.random(n=n_wells)
    indices = np.round(samples * (grid_size - 1)).astype(int)
    seen, wells = set(), []
    for row, col in indices:
        coord = (int(row), int(col))
        if coord not in seen:
            seen.add(coord)
            wells.append(coord)
    return wells


def get_observations(H_true, wells, noise_std=0.0):
    """
    在井位取水头观测值。
    观测噪声通过 R 矩阵隐式处理，默认不加显式扰动。
    """
    n_wells = len(wells)
    h_obs   = np.zeros((5, n_wells), dtype=np.float32)
    for i, (row, col) in enumerate(wells):
        h_obs[:, i] = H_true[:, row, col]
    if noise_std > 0:
        h_obs += np.random.normal(0, noise_std, size=h_obs.shape)
    return h_obs


def extract_well_observations(H_full, wells):
    N       = H_full.shape[0]
    n_wells = len(wells)
    H_wells = np.zeros((N, 5, n_wells), dtype=np.float32)
    for w, (row, col) in enumerate(wells):
        H_wells[:, :, w] = H_full[:, :, row, col]
    return H_wells


# ================= 正演与编解码 =================
def vae_encode(K_samples, vae, config):
    K_tensor = torch.tensor(K_samples[:, np.newaxis, :, :],
                            dtype=torch.float32).to(config.DEVICE)
    with torch.no_grad():
        mu, logvar = vae.encode(K_tensor)
        z = vae.reparameterize(mu, logvar)
    return z.cpu().numpy()


def vae_decode(z, vae, config):
    z_tensor = torch.tensor(z, dtype=torch.float32).to(config.DEVICE)
    with torch.no_grad():
        K = vae.decode(z_tensor)
    return K.cpu().numpy().squeeze(1)


def predict_H_batch(K_samples, fno_kh, config):
    K_tensor = torch.tensor(K_samples[:, np.newaxis, :, :],
                            dtype=torch.float32).to(config.DEVICE)
    with torch.no_grad():
        H = fno_kh(K_tensor, return_normalized=False)
    return H.cpu().numpy()


def predict_C_batch(K_samples, fno_kc, config):
    K_tensor = torch.tensor(K_samples[:, np.newaxis, :, :],
                            dtype=torch.float32).to(config.DEVICE)
    with torch.no_grad():
        C = fno_kc(K_tensor, return_normalized=False)
    return C.cpu().numpy()


# ================= ES-MDA =================
def es_mda_update(z_f, obs, R, H_f, alpha, seed=None):
    """
    ES-MDA 单步更新。

    参数
    ----
    z_f   : (N, d_z)      先验潜变量集合
    obs   : (T, W)        观测矩阵，T=抽水试验次数，W=井数
    R     : (W, W)        单次抽水试验观测误差协方差
    H_f   : (N, T, W)     集合前向预测
    alpha : float         ES-MDA 膨胀系数（Σ(1/α)=1 时理论最优）
    seed  : int | None    随机种子
    """
    if seed is not None:
        np.random.seed(seed)

    num_samples, _       = z_f.shape
    num_times, num_wells = obs.shape

    H_f_mean = np.mean(H_f, axis=0)
    H_f_ana  = H_f - H_f_mean
    z_f_mean = np.mean(z_f, axis=0)
    z_f_ana  = z_f - z_f_mean

    H_f_ana_r = H_f_ana.reshape(num_samples, -1)

    C_md = np.dot(z_f_ana.T, H_f_ana_r) / (num_samples - 1)
    C_dd = np.dot(H_f_ana_r.T, H_f_ana_r) / (num_samples - 1)

    R_expanded  = np.kron(np.eye(num_times), R)
    C_dd       += alpha * R_expanded

    gain = np.dot(C_md, inv(C_dd))

    obs_flat          = obs.flatten()
    obs_pert_cov      = alpha * R_expanded
    obs_perturbations = np.random.multivariate_normal(
        np.zeros(obs_flat.shape[0]), obs_pert_cov, num_samples)

    z_a = np.zeros_like(z_f)
    for i in range(num_samples):
        d_perturbed = obs_flat + obs_perturbations[i]
        innovation  = d_perturbed - H_f[i].flatten()
        z_a[i]      = z_f[i] + np.dot(gain, innovation)

    return z_a


def generate_prior(vae, k_data, n_samples, config, seed=None):
    if seed is not None:
        np.random.seed(seed)
    idx       = np.random.choice(len(k_data), n_samples, replace=False)
    k_samples = k_data[idx]
    return vae_encode(k_samples, vae, config)


def run_esmda(vae, fno_kh, z_prior, wells, h_obs, config,
              alpha_list=None, base_seed=None):
    """
    ES-MDA 同化主函数。
    R = (sigma_obs² + sigma_f1²)·I  （model error inflation）
    """
    if alpha_list is None:
        alpha_list = config.ALPHA_LIST

    sum_inv_alpha = sum(1.0 / a for a in alpha_list)
    if abs(sum_inv_alpha - 1.0) > 0.01:
        print(f"  [Warning] ES-MDA Σ(1/αᵢ) = {sum_inv_alpha:.4f} ≠ 1.0")

    n_wells      = len(wells)
    sigma_eff_sq = config.OBS_NOISE_STD ** 2 + config.SIGMA_F1 ** 2
    R            = np.eye(n_wells) * sigma_eff_sq

    z = z_prior.copy()
    for i, alpha in enumerate(alpha_list):
        K_decoded = vae_decode(z, vae, config)
        H_full    = predict_H_batch(K_decoded, fno_kh, config)
        H_f       = extract_well_observations(H_full, wells)
        iter_seed = base_seed + i if base_seed is not None else None
        z = es_mda_update(z, h_obs, R, H_f, alpha, seed=iter_seed)

    K_post = vae_decode(z, vae, config)
    return K_post, z


# ================= Look-ahead 核心函数 =================
def virtual_es_mda_update(z_ensemble, existing_wells, new_well,
                           h_obs_existing, vae, fno_kh, config):
    """
    虚拟 ES-MDA 更新（look-ahead 候选井评估）。
    差异化 R：
      已有井  R_ii = sigma_obs² + sigma_f1²
      候选井  R_ii = max(sigma_f1², 1e-8)
    """
    virtual_alpha_list = config.VIRTUAL_ALPHA_LIST
    sum_inv_alpha = sum(1.0 / a for a in virtual_alpha_list)
    if abs(sum_inv_alpha - 1.0) > 0.01:
        print(f"  [Warning] Virtual ES-MDA Σ(1/αᵢ) = {sum_inv_alpha:.4f} ≠ 1.0")

    all_wells    = list(existing_wells) + [new_well]
    num_existing = len(existing_wells)
    num_wells    = len(all_wells)

    K_curr          = vae_decode(z_ensemble, vae, config)
    H_curr          = predict_H_batch(K_curr, fno_kh, config)
    r_new, c_new    = new_well
    h_pred_mean_new = H_curr[:, :, r_new, c_new].mean(axis=0)

    h_obs_virtual                   = np.zeros((5, num_wells), dtype=np.float32)
    h_obs_virtual[:, :num_existing] = h_obs_existing
    h_obs_virtual[:, num_existing]  = h_pred_mean_new

    sigma_eff_sq         = config.OBS_NOISE_STD ** 2 + config.SIGMA_F1 ** 2
    _sigma_f1_sq         = config.SIGMA_F1 ** 2
    R_diag               = np.full(num_wells, sigma_eff_sq)
    R_diag[num_existing] = max(_sigma_f1_sq, 1e-8)
    R = np.diag(R_diag)

    z = z_ensemble.copy()
    for alpha in virtual_alpha_list:
        K   = vae_decode(z, vae, config)
        H   = predict_H_batch(K, fno_kh, config)
        H_f = extract_well_observations(H, all_wells)
        z   = es_mda_update(z, h_obs_virtual, R, H_f, alpha)

    return z


def select_optimal_well(z_post, existing_wells, h_obs_existing,
                        fno_kh, fno_kc, vae, config):
    """
    Practical Look-ahead 最优井位选择。
    选择虚拟同化后浓度场总方差下降最大的候选位置。
    """
    K_curr = vae_decode(z_post, vae, config)
    C_curr = predict_C_batch(K_curr, fno_kc, config)
    curr_total_var, curr_var_map, _, _, _ = \
        compute_weighted_C_variance(C_curr, config)

    candidates = []
    if config.USE_CANDIDATE_FILTER:
        threshold    = np.percentile(curr_var_map, config.VARIANCE_PERCENTILE)
        rows, cols   = np.where(curr_var_map >= threshold)
        existing_set = set(existing_wells)
        for r, c in zip(rows, cols):
            if (r, c) in existing_set:
                continue
            too_close = any(
                abs(r - wr) <= config.MIN_WELL_SPACING and
                abs(c - wc) <= config.MIN_WELL_SPACING
                for wr, wc in existing_wells
            )
            if not too_close:
                candidates.append((r, c))
        candidates.sort(key=lambda p: curr_var_map[p[0], p[1]], reverse=True)
        candidates = candidates[:config.MAX_CANDIDATES]

    print(f"  [Select] 当前候选位置数: {len(candidates)} "
          f"(percentile={config.VARIANCE_PERCENTILE}, max={config.MAX_CANDIDATES})")

    results = []
    for cand in candidates:
        z_new = virtual_es_mda_update(
            z_post, existing_wells, cand, h_obs_existing, vae, fno_kh, config)
        K_new = vae_decode(z_new, vae, config)
        C_new = predict_C_batch(K_new, fno_kc, config)
        new_total_var, _, _, _, _ = compute_weighted_C_variance(C_new, config)
        results.append({
            'loc':     cand,
            'delta':   curr_total_var - new_total_var,
            'new_var': new_total_var,
        })

    if not results:
        return None, 0.0, curr_var_map

    results.sort(key=lambda x: x['delta'], reverse=True)
    best = results[0]
    return best['loc'], best['delta'], curr_var_map


# ================= 评估与方差计算 =================
def compute_weighted_C_variance(C_samples, config):
    """
    计算时间加权浓度场集合方差。

    返回
    ----
    total_var : float
        全域加权总方差 = Σ_{i,j} var_map[i,j]
        用于 Look-ahead 内部优化（最大化方差下降）

    var_map : ndarray (GRID_SIZE, GRID_SIZE)
        空间方差分布图，用于可视化和候选筛选

    var_per_time : list of float
        各时刻方差之和

    mean_var : float
        平均每格点方差 = total_var / GRID_SIZE²
        消除网格尺寸影响

    mean_std : float
        平均每格点标准差 = sqrt(mean_var)
        单位与浓度 C 一致，若 C∈[0,1] 则可直接解读为量程百分比
        用于论文展示
    """
    n_grid       = config.GRID_SIZE * config.GRID_SIZE
    var_map      = np.zeros((config.GRID_SIZE, config.GRID_SIZE), dtype=np.float32)
    var_per_time = []

    for t in range(config.N_TIME_STEPS):
        var_t = np.var(C_samples[:, t, :, :], axis=0)
        var_per_time.append(float(np.sum(var_t)))
        var_map += config.TIME_WEIGHTS[t] * var_t

    total_var = float(np.sum(var_map))
    mean_var  = total_var / n_grid
    mean_std  = float(np.sqrt(mean_var))

    return total_var, var_map, var_per_time, mean_var, mean_std


def evaluate_prediction(C_samples, C_true, K_samples, K_true, config):
    """
    评估集合预测质量。

    返回字典新增：
      C_mean_var : 平均每格点方差
      C_mean_std : 平均每格点标准差（论文展示用）
    """
    C_mean = np.mean(C_samples, axis=0)
    K_mean = np.mean(K_samples, axis=0)

    K_r2   = r2_score(K_true.flatten(), K_mean.flatten())
    K_rmse = np.sqrt(np.mean((K_true - K_mean) ** 2))
    K_ssim = ssim(K_true, K_mean, data_range=K_true.max() - K_true.min())

    C_r2   = r2_score(C_true.flatten(), C_mean.flatten())
    C_rmse = np.sqrt(np.mean((C_true - C_mean) ** 2))

    C_r2_time = [
        r2_score(C_true[t].flatten(), C_mean[t].flatten())
        for t in range(config.N_TIME_STEPS)
    ]

    total_var, _, var_per_time, mean_var, mean_std = \
        compute_weighted_C_variance(C_samples, config)

    return {
        'K_r2':          K_r2,
        'K_rmse':        K_rmse,
        'K_ssim':        K_ssim,
        'C_r2':          C_r2,
        'C_rmse':        C_rmse,
        'C_r2_per_time': C_r2_time,
        'C_var':         total_var,   # 全域总方差（内部优化用）
        'C_mean_var':    mean_var,    # 平均每格点方差
        'C_mean_std':    mean_std,    # 平均每格点标准差（论文展示用）
        'C_var_per_time': var_per_time,
    }


def check_variance_reduction(initial_C_var, final_C_var, method_name=""):
    """保留备用，业务流程中不调用。"""
    prefix = f"  [{method_name}] " if method_name else "  "
    if initial_C_var <= 0:
        print(f"{prefix}[Warning] initial_C_var <= 0，无法计算")
        return
    reduction_pct = (1.0 - final_C_var / initial_C_var) * 100.0
    print(f"{prefix}C Var: {initial_C_var:.4f} → {final_C_var:.4f} "
          f"(↓{reduction_pct:.1f}%)")