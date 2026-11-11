"""
shared_utils.py
地下水监测网络设计与 ES-MDA 实验的公共函数。
- 模型加载、先验集合生成、代理预测、同化更新与选井评价
- 顺序选井时，初始井同化一次，之后每次仅同化新增井
- 验证集校准 FNO-1 误差，正式与虚拟同化采用相同的 R
- 矩阵化 ES-MDA，支持分批代理模型预测
"""

import numpy as np
import torch
import scipy.io as sio
from scipy.stats import qmc
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity as ssim

from train_vae import VAE
from train_f1 import FNO2d as FNO2d_KH
from train_f2 import FNO2d_KC


class BaseConfig:
    """基础配置参数"""
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GRID_SIZE = 51
    N_TIME_STEPS = 5

    VAE_PATH = './vae_results/best_vae.pth'
    FNO_KH_PATH = './fno_kh_results/best_fno_kh.pth'
    FNO_KC_PATH = './fno_kc_results/best_fno_kc.pth'
    K_DATA_PATH = './data/DataK_6000_51_51.mat'
    H_DATA_PATH = './data/DataH_6000_5_51_51.mat'
    KC_NORM_PATH = './fno_kc_results/normalization_params.npz'
    TEST_DATA_PATH = './test_synthetic_data/synthetic_test_100.npz'
    VALIDATION_INDICES_PATH = './fno_kh_results/val_indices.npy'

    N_ENSEMBLE = 500
    INITIAL_WELLS = 3
    TARGET_WELLS = 20
    NUM_RANDOM_TRIALS = 10
    BATCH_SIZE = 24

    OBS_NOISE_STD = 0.3
    SIGMA_F1 = 0.0       # 运行时由 load_sigma_f1() 在 FNO-1 验证集上校准
    ALPHA_LIST = [12.0] * 12
    VIRTUAL_ALPHA_LIST = [4.0] * 4
    TIME_WEIGHTS = [0.1, 0.15, 0.2, 0.25, 0.3]
    MAX_CANDIDATES = 500
    MIN_WELL_SPACING = 2


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_mat(path):
    mat = sio.loadmat(path)
    keys = [k for k in mat if not k.startswith('__')]
    return mat[keys[0]].astype(np.float32)


def load_all_models(config):
    print(f'[Info] Loading models on {config.DEVICE}...')
    vae = VAE(latent_dim=128).to(config.DEVICE)
    vae.load_state_dict(torch.load(config.VAE_PATH, map_location=config.DEVICE))
    vae.eval()

    H_train = load_mat(config.H_DATA_PATH)
    fno_kh = FNO2d_KH(modes1=12, modes2=12, width=64, in_channels=1,
                       out_channels=5, h_mean=float(H_train.mean()),
                       h_std=float(H_train.std())).to(config.DEVICE)
    fno_kh.load_state_dict(torch.load(config.FNO_KH_PATH, map_location=config.DEVICE))
    fno_kh.eval()
    del H_train

    with np.load(config.KC_NORM_PATH) as d:
        c_mean, c_std = float(d['c_mean']), float(d['c_std'])
    fno_kc = FNO2d_KC(modes1=12, modes2=12, width=64, in_channels=1,
                       out_channels=5, c_mean=c_mean, c_std=c_std).to(config.DEVICE)
    fno_kc.load_state_dict(torch.load(config.FNO_KC_PATH, map_location=config.DEVICE))
    fno_kc.eval()
    return vae, fno_kh, fno_kc


def load_test_data(config):
    """读取测试真值与先验 K 数据；代理误差由独立的验证集校准函数计算。"""
    with np.load(config.TEST_DATA_PATH) as d:
        K = d['K_test'].astype(np.float32)
        H = d['H_test'].astype(np.float32)
        C = d['C_test'].astype(np.float32)
    K_train = load_mat(config.K_DATA_PATH)
    return K, H, C, K_train


def calibrate_sigma_f1(fno_kh, config):
    """按 FNO-1 保存的验证集索引计算物理水头尺度的 RMSE。

    返回全局 RMSE、各抽水试验的 RMSE 和验证样本数；不使用测试集。
    """
    indices = np.load(config.VALIDATION_INDICES_PATH)
    if indices.ndim != 1 or indices.size == 0 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('验证集索引必须为非空的一维整数数组')
    K_train = load_mat(config.K_DATA_PATH)
    H_train = load_mat(config.H_DATA_PATH)
    if (np.any(indices < 0) or np.any(indices >= len(K_train))
            or np.any(indices >= len(H_train))):
        raise ValueError('验证集索引超出 K/H 数据范围')
    residual_sum = 0.0
    n_values = 0
    rmse_t_sum = np.zeros(config.N_TIME_STEPS, dtype=np.float64)
    n_t = 0
    for start in range(0, len(indices), config.BATCH_SIZE):
        idx = indices[start:start + config.BATCH_SIZE]
        H_pred = predict_H_batch(K_train[idx], fno_kh, config)
        diff = H_pred.astype(np.float64) - H_train[idx].astype(np.float64)
        residual_sum += float(np.sum(diff ** 2))
        n_values += diff.size
        rmse_t_sum += np.sum(diff ** 2, axis=(0, 2, 3))
        n_t += len(idx) * config.GRID_SIZE ** 2
    sigma = np.sqrt(residual_sum / n_values)
    sigma_per_time = np.sqrt(rmse_t_sum / n_t)
    return float(sigma), sigma_per_time.tolist(), len(indices)


def _batched(x, batch_size, fn):
    return np.concatenate([fn(x[i:i + batch_size])
                           for i in range(0, len(x), batch_size)], axis=0)


def vae_encode(K_samples, vae, config):
    def run(part):
        x = torch.as_tensor(part[:, None], dtype=torch.float32, device=config.DEVICE)
        with torch.no_grad():
            mu, logvar = vae.encode(x)
            z = vae.reparameterize(mu, logvar)
        return z.cpu().numpy()
    return _batched(K_samples, config.BATCH_SIZE, run)


def vae_decode(z, vae, config):
    def run(part):
        x = torch.as_tensor(part, dtype=torch.float32, device=config.DEVICE)
        with torch.no_grad():
            out = vae.decode(x)
        return out.cpu().numpy()[:, 0]
    return _batched(z, config.BATCH_SIZE, run)


def predict_H_batch(K_samples, fno_kh, config):
    def run(part):
        x = torch.as_tensor(part[:, None], dtype=torch.float32, device=config.DEVICE)
        with torch.no_grad():
            out = fno_kh(x, return_normalized=False)
        return out.cpu().numpy()
    return _batched(K_samples, config.BATCH_SIZE, run)


def predict_C_batch(K_samples, fno_kc, config):
    def run(part):
        x = torch.as_tensor(part[:, None], dtype=torch.float32, device=config.DEVICE)
        with torch.no_grad():
            out = fno_kc(x, return_normalized=False)
        return out.cpu().numpy()
    return _batched(K_samples, config.BATCH_SIZE, run)


def init_wells_lhs(n_wells, grid_size, seed):
    samples = qmc.LatinHypercube(d=2, seed=seed).random(n_wells)
    cells = np.round(samples * (grid_size - 1)).astype(int)
    return list(dict.fromkeys((int(r), int(c)) for r, c in cells))


def initial_wells(config):
    """生成所有策略共享的初始 LHS 井序。"""
    wells = init_wells_lhs(config.INITIAL_WELLS, config.GRID_SIZE, config.SEED)
    if len(wells) != config.INITIAL_WELLS:
        raise ValueError('初始 LHS 井取整后发生重复，请核对实验井序')
    return wells


def stage_seed(config, sample_idx, n_wells):
    """按样本编号与当前井数确定顺序同化的扰动种子。"""
    return int(config.SEED + 1200 + sample_idx * 100 + n_wells)


def prepare_initial_latent(H_true, z_prior, vae, fno_kh, config, sample_idx):
    """从先验同化共享初始井，返回后验潜变量集合。"""
    wells = initial_wells(config)
    observations = get_observations(H_true, wells)
    return assimilate_stage(z_prior, wells, observations, vae, fno_kh, config,
                            stage_seed(config, sample_idx, config.INITIAL_WELLS))


def spacing_valid(wells, config):
    """检查新增井与已有井的 Chebyshev 距离是否大于 MIN_WELL_SPACING。"""
    for i in range(config.INITIAL_WELLS, len(wells)):
        r, c = wells[i]
        for r0, c0 in wells[:i]:
            if max(abs(r-r0), abs(c-c0)) <= config.MIN_WELL_SPACING:
                return False
    return len(set(map(tuple, wells))) == len(wells)


def feasible_wells(var_map, wells, config, limit=None):
    """按照浓度当前方差排序，并在排序前施加统一空间间距约束。"""
    coords = [(r, c) for r in range(config.GRID_SIZE)
             for c in range(config.GRID_SIZE)
             if all(max(abs(r-r0), abs(c-c0)) > config.MIN_WELL_SPACING
                    for r0, c0 in wells)]
    coords.sort(key=lambda p: (-float(var_map[p]), p[0], p[1]))
    return coords if limit is None else coords[:limit]


def get_observations(H_true, wells):
    rows, cols = np.asarray(wells, dtype=int).T
    return H_true[:, rows, cols]


def extract_well_observations(H_full, wells):
    rows, cols = np.asarray(wells, dtype=int).T
    return H_full[:, :, rows, cols]


def compute_weighted_C_variance(C_samples, config):
    """论文 sigma_bar_C = sqrt(空间平均时间加权集合方差)。"""
    v_time = np.var(C_samples, axis=0)
    v_map = np.tensordot(config.TIME_WEIGHTS, v_time, axes=(0, 0))
    total = float(v_map.sum())
    mean_var = float(v_map.mean())
    return total, v_map, v_time.sum(axis=(1, 2)).tolist(), mean_var, np.sqrt(mean_var)


def evaluate_prediction(C_samples, C_true, K_samples, K_true, config):
    C_mean = C_samples.mean(axis=0)
    K_mean = K_samples.mean(axis=0)
    total, _, v_time, mean_var, mean_std = compute_weighted_C_variance(C_samples, config)
    return {
        'K_r2': float(r2_score(K_true.ravel(), K_mean.ravel())),
        'K_rmse': float(np.sqrt(np.mean((K_true - K_mean) ** 2))),
        'K_ssim': float(ssim(K_true, K_mean, data_range=float(np.ptp(K_true)))),
        'C_r2': float(r2_score(C_true.ravel(), C_mean.ravel())),
        'C_rmse': float(np.sqrt(np.mean((C_true - C_mean) ** 2))),
        'C_r2_per_time': [float(r2_score(C_true[t].ravel(), C_mean[t].ravel()))
                          for t in range(config.N_TIME_STEPS)],
        'C_var': total,
        'C_mean_var': mean_var,
        'C_mean_std': float(mean_std),
        'C_var_per_time': v_time,
    }


def es_mda_update(z, observations, H_prediction, r_variance, alpha, seed):
    """使用 Cholesky 求解矩阵增益；无需显式求逆和逐集合成员循环。"""
    n = len(z)
    d = H_prediction.reshape(n, -1).astype(np.float64)
    z64 = z.astype(np.float64)
    za = z64 - z64.mean(axis=0)
    da = d - d.mean(axis=0)
    C_zd = za.T @ da / (n - 1)
    C_dd = da.T @ da / (n - 1)
    A = C_dd + alpha * r_variance * np.eye(d.shape[1])
    gain = cho_solve(cho_factor(A, lower=True, check_finite=False),
                     C_zd.T, check_finite=False).T

    rng = np.random.default_rng(seed)
    perturb = rng.standard_normal(d.shape) * np.sqrt(alpha * r_variance)
    innovation = observations.ravel()[None, :] + perturb - d
    return (z64 + innovation @ gain.T).astype(np.float32)


def assimilate_stage(z, wells, observations, vae, fno_kh, config, seed,
                     virtual=False, first_head=None, alpha_list=None):
    """从传入集合出发，完成给定井位观测的一轮 ES-MDA。

    顺序选井时仅传入尚未同化的观测；独立验证时可从先验一次同化全部井。
    alpha_list 可显式指定膨胀系数；默认使用配置中的正式或虚拟同化系数。
    """
    if alpha_list is not None:
        alphas = list(alpha_list)
    else:
        alphas = config.VIRTUAL_ALPHA_LIST if virtual else config.ALPHA_LIST
    r_variance = config.OBS_NOISE_STD ** 2 + config.SIGMA_F1 ** 2
    current = z.copy()
    for it, alpha in enumerate(alphas):
        if virtual and it == 0 and first_head is not None:
            h_wells = first_head[:, :, None]
        else:
            K = vae_decode(current, vae, config)
            H = predict_H_batch(K, fno_kh, config)
            h_wells = extract_well_observations(H, wells)
        current = es_mda_update(current, observations, h_wells,
                                r_variance, alpha, seed + it)
    return current


def select_look_ahead(z, wells, vae, fno_kh, fno_kc, config, seed,
                      current_K=None, current_C=None):
    """筛选最多 MAX_CANDIDATES 个高方差可行位置，以虚拟同化评价浓度方差缩减。"""
    K = vae_decode(z, vae, config) if current_K is None else current_K
    C = predict_C_batch(K, fno_kc, config) if current_C is None else current_C
    H = predict_H_batch(K, fno_kh, config)   # 候选初始水头只预测一次
    base_var, var_map, _, _, _ = compute_weighted_C_variance(C, config)
    candidates = feasible_wells(var_map, wells, config, config.MAX_CANDIDATES)
    best, best_score = None, -np.inf

    for r, c in candidates:
        head = H[:, :, r, c]
        pseudo_observation = head.mean(axis=0).reshape(config.N_TIME_STEPS, 1)
        z_virtual = assimilate_stage(z, [(r, c)], pseudo_observation,
                                     vae, fno_kh, config, seed,
                                     virtual=True, first_head=head)
        K_virtual = vae_decode(z_virtual, vae, config)
        C_virtual = predict_C_batch(K_virtual, fno_kc, config)
        new_var = compute_weighted_C_variance(C_virtual, config)[0]
        score = base_var - new_var
        if score > best_score:
            best, best_score = (r, c), float(score)
    return best, best_score, len(candidates)


# =====================================================================
# 实验公共接口：验证集误差校准、样本读取、先验生成与 ES-MDA 运行
# =====================================================================


def load_sigma_f1(config, fno_kh):
    """通过公共校准函数计算验证集误差，并写入 config.SIGMA_F1。

    每次启动实验时调用一次，不依赖选井策略的运行结果或测试集误差。
    正式和虚拟同化均使用 R = (OBS_NOISE_STD² + SIGMA_F1²) I。
    """
    sigma, _, n_validation = calibrate_sigma_f1(fno_kh, config)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError('验证集 FNO-1 RMSE 必须为有限的非负数')
    config.SIGMA_F1 = sigma
    print(f'[R] sigma_f1(val) = {sigma:.5f}; '
          f'R = {config.OBS_NOISE_STD**2 + sigma**2:.6f} I  '
          f'[validation samples={n_validation}]')
    return sigma


def get_test_sample(K_all, H_all, C_all, sample_idx):
    return K_all[sample_idx], H_all[sample_idx], C_all[sample_idx]


def generate_prior(vae, k_data, n_samples, config, seed=None):
    """从先验 K 数据中无放回抽样并编码；同时固定 NumPy 与 PyTorch 种子。"""
    set_seed(config.SEED if seed is None else int(seed))
    idx = np.random.choice(len(k_data), n_samples, replace=False)
    return vae_encode(k_data[idx], vae, config)


def lhs_wells(n_wells, config, seed):
    """LHS 井位；取整去重后不足 n_wells 时，用同一种子的更大 LHS 池补齐。

    不出现重复时与 init_wells_lhs(n_wells, ...) 完全相同。
    """
    wells = init_wells_lhs(n_wells, config.GRID_SIZE, seed)
    if len(wells) < n_wells:
        seen = set(wells)
        for extra in (n_wells * 2, n_wells * 4, config.GRID_SIZE ** 2):
            for w in init_wells_lhs(extra, config.GRID_SIZE, seed):
                if w not in seen:
                    wells.append(w)
                    seen.add(w)
                if len(wells) >= n_wells:
                    break
            if len(wells) >= n_wells:
                break
    return wells[:n_wells]


def run_esmda(vae, fno_kh, z_prior, wells, h_obs, config,
              alpha_list=None, base_seed=None):
    """一次性同化给定井位的全部观测（从 z_prior 出发），返回 (K_post, z_post)。

    内部调用 assimilate_stage，与主实验使用相同的 R = (sigma_obs² + sigma_f1²) I
    和相同的矩阵化 ES-MDA 更新。
    """
    if config.SIGMA_F1 <= 0:
        raise ValueError('config.SIGMA_F1 未设置：请先调用 load_sigma_f1(config, fno_kh)')
    seed = config.SEED if base_seed is None else int(base_seed)
    z_post = assimilate_stage(z_prior, wells, h_obs, vae, fno_kh, config, seed,
                              alpha_list=alpha_list)
    return vae_decode(z_post, vae, config), z_post
