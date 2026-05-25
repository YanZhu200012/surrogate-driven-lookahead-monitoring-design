"""
run_100.py
100 个测试场的完整实验流程

四方法对比（统一序贯 ES-MDA 协议）：
  - active : Look-ahead（自适应；look-ahead 虚拟同化）
  - greedy : MaxVar 贪心（自适应；当前后验方差最大处加井）
  - random : 随机选井（非自适应；10 次 trial 取 mean ± std）
  - lhs    : Latin Hypercube 空间填充（非自适应；10 次 trial 取 mean ± std）

协议一致性
----------
  D1 同化协议   : sequential ES-MDA（所有方法共用 shared_utils.run_esmda）
  D2 先验集合   : 同一 SEED、同一 N_ENSEMBLE，跨方法严格一致
  D3 起跑线     : 前 3 口初始井固定 seed = config.SEED，跨方法/跨样本共享
  D4 选井准则   : 四种方法在此唯一不同 —— 任何性能差异都可干净归因到准则本身

多 trial 设计
-------------
  Random 与 LHS 都是非自适应方法（井位与后验无关），单次运行的 spread
  无法刻画方法本身的性质，因此各跑 NUM_RANDOM_TRIALS (=10) 次：
    - random : 用 np.random.choice 抽签，seed = SEED + 1000 + sample_idx*100 + trial
    - lhs    : 用 LHS 拉丁超立方设计，seed = SEED + 3000 + sample_idx*100 + trial
  两者在 trial 维度上的统计口径完全一致（mean ± std），可干净对照
  "空间填充设计 vs 均匀随机抽样"的差异。

输出
----
  ./test_100_results/sample_{idx:03d}/{method}.npz   每样本-每方法一份结果
  ./test_100_results/progress.json                   断点续算与汇总进度
"""

import os
import time
import json
import numpy as np
import shared_utils as su


class TestConfig(su.BaseConfig):
    OUTPUT_ROOT          = './test_100_results'
    VIRTUAL_ALPHA_LIST   = [4.0] * 4
    USE_CANDIDATE_FILTER = True
    MAX_CANDIDATES       = 500
    VARIANCE_PERCENTILE  = 30
    MIN_WELL_SPACING     = 2
    NUM_RANDOM_TRIALS    = 10


# ==========================================
# Look-ahead
# ==========================================
def run_look_ahead(sample_idx, K_true, H_true, C_true, k_train_data,
                        vae, fno_kh, fno_kc, config):
    wells   = su.init_wells_lhs(config.INITIAL_WELLS, config.GRID_SIZE, config.SEED)
    z_prior = su.generate_prior(vae, k_train_data, config.N_ENSEMBLE, config, config.SEED)
    z_curr  = z_prior.copy()
    history = []

    K_prior = su.vae_decode(z_prior, vae, config)
    C_prior = su.predict_C_batch(K_prior, fno_kc, config)
    initial_C_var, _, _, _, initial_mean_std = \
        su.compute_weighted_C_variance(C_prior, config)

    while len(wells) < config.TARGET_WELLS:
        h_obs          = su.get_observations(H_true, wells)
        K_post, z_post = su.run_esmda(vae, fno_kh, z_curr, wells, h_obs, config)
        z_curr         = z_post.copy()

        C_post  = su.predict_C_batch(K_post, fno_kc, config)
        metrics = su.evaluate_prediction(C_post, C_true, K_post, K_true, config)
        metrics['n_wells'] = len(wells)
        metrics['wells']   = [list(w) for w in wells]

        if len(wells) < config.TARGET_WELLS:
            new_well, delta, _ = su.select_optimal_well(
                z_post, wells, h_obs, fno_kh, fno_kc, vae, config)
            if new_well:
                wells.append(new_well)
            else:
                break
        history.append(metrics)

    # 最终评估
    h_obs          = su.get_observations(H_true, wells)
    K_post, z_post = su.run_esmda(vae, fno_kh, z_curr, wells, h_obs, config)
    C_post         = su.predict_C_batch(K_post, fno_kc, config)
    final_metrics  = su.evaluate_prediction(C_post, C_true, K_post, K_true, config)
    final_metrics['n_wells'] = len(wells)
    final_metrics['wells']   = [list(w) for w in wells]
    history.append(final_metrics)

    return {
        'history':          history,
        'final_wells':      [list(w) for w in wells],
        'K_post_mean':      np.mean(K_post, axis=0),
        'K_post_std':       np.std(K_post, axis=0),
        'C_post_mean':      np.mean(C_post, axis=0),
        'C_post_std':       np.std(C_post, axis=0),
        'initial_C_var':    initial_C_var,
        'initial_mean_std': initial_mean_std,
    }


# ==========================================
# Greedy (MaxVar)
# ==========================================
def select_maxvar_well(variance_map, existing_wells, config):
    mask = np.ones_like(variance_map, dtype=bool)
    for r, c in existing_wells:
        r_min = max(0, r - config.MIN_WELL_SPACING)
        r_max = min(config.GRID_SIZE, r + config.MIN_WELL_SPACING + 1)
        c_min = max(0, c - config.MIN_WELL_SPACING)
        c_max = min(config.GRID_SIZE, c + config.MIN_WELL_SPACING + 1)
        mask[r_min:r_max, c_min:c_max] = False
    masked_var = np.where(mask, variance_map, -1.0)
    idx = np.argmax(masked_var)
    r, c = np.unravel_index(idx, variance_map.shape)
    return (int(r), int(c)), masked_var[r, c]


def run_greedy(sample_idx, K_true, H_true, C_true, k_train_data,
               vae, fno_kh, fno_kc, config):
    wells   = su.init_wells_lhs(config.INITIAL_WELLS, config.GRID_SIZE, config.SEED)
    z_prior = su.generate_prior(vae, k_train_data, config.N_ENSEMBLE, config, config.SEED)
    z_curr  = z_prior.copy()
    history = []

    K_prior = su.vae_decode(z_prior, vae, config)
    C_prior = su.predict_C_batch(K_prior, fno_kc, config)
    initial_C_var, _, _, _, initial_mean_std = \
        su.compute_weighted_C_variance(C_prior, config)

    while len(wells) < config.TARGET_WELLS:
        h_obs          = su.get_observations(H_true, wells)
        K_post, z_post = su.run_esmda(vae, fno_kh, z_curr, wells, h_obs, config)
        z_curr         = z_post.copy()

        C_post  = su.predict_C_batch(K_post, fno_kc, config)
        metrics = su.evaluate_prediction(C_post, C_true, K_post, K_true, config)
        metrics['n_wells'] = len(wells)
        metrics['wells']   = [list(w) for w in wells]

        _, var_map, _, _, _ = su.compute_weighted_C_variance(C_post, config)
        if len(wells) < config.TARGET_WELLS:
            new_well, _ = select_maxvar_well(var_map, wells, config)
            wells.append(new_well)
        history.append(metrics)

    h_obs          = su.get_observations(H_true, wells)
    K_post, z_post = su.run_esmda(vae, fno_kh, z_curr, wells, h_obs, config)
    C_post         = su.predict_C_batch(K_post, fno_kc, config)
    final_metrics  = su.evaluate_prediction(C_post, C_true, K_post, K_true, config)
    final_metrics['n_wells'] = len(wells)
    final_metrics['wells']   = [list(w) for w in wells]
    history.append(final_metrics)

    return {
        'history':          history,
        'final_wells':      [list(w) for w in wells],
        'K_post_mean':      np.mean(K_post, axis=0),
        'K_post_std':       np.std(K_post, axis=0),
        'C_post_mean':      np.mean(C_post, axis=0),
        'C_post_std':       np.std(C_post, axis=0),
        'initial_C_var':    initial_C_var,
        'initial_mean_std': initial_mean_std,
    }


# ==========================================
# Random（10 次 trial，与 LHS 完全对齐的统计口径）
# ==========================================
def run_random_single_trial(trial_seed, K_true, H_true, C_true, k_train_data,
                            vae, fno_kh, fno_kc, config):
    su.set_seed(trial_seed)
    wells      = su.init_wells_lhs(config.INITIAL_WELLS, config.GRID_SIZE, config.SEED)
    existing   = set(wells)
    candidates = [(r, c) for r in range(config.GRID_SIZE)
                          for c in range(config.GRID_SIZE)
                          if (r, c) not in existing]

    n_needed     = config.TARGET_WELLS - config.INITIAL_WELLS
    rand_idx     = np.random.choice(len(candidates), n_needed, replace=False)
    future_wells = [candidates[i] for i in rand_idx]

    z_prior  = su.generate_prior(vae, k_train_data, config.N_ENSEMBLE, config, config.SEED)
    z_curr   = z_prior.copy()
    history  = []
    full_seq = wells + future_wells

    for i in range(len(full_seq) - config.INITIAL_WELLS + 1):
        step_wells     = full_seq[:config.INITIAL_WELLS + i]
        h_obs          = su.get_observations(H_true, step_wells)
        K_post, z_post = su.run_esmda(vae, fno_kh, z_curr, step_wells, h_obs, config)
        z_curr         = z_post.copy()

        C_post  = su.predict_C_batch(K_post, fno_kc, config)
        metrics = su.evaluate_prediction(C_post, C_true, K_post, K_true, config)
        metrics['n_wells'] = len(step_wells)
        metrics['wells']   = [list(w) for w in step_wells]
        history.append(metrics)

    return history, np.mean(K_post, axis=0), np.mean(C_post, axis=0)


def run_random(sample_idx, K_true, H_true, C_true, k_train_data,
               vae, fno_kh, fno_kc, config):
    z_prior_check = su.generate_prior(
        vae, k_train_data, config.N_ENSEMBLE, config, config.SEED)
    K_prior_check = su.vae_decode(z_prior_check, vae, config)
    C_prior_check = su.predict_C_batch(K_prior_check, fno_kc, config)
    initial_C_var, _, _, _, initial_mean_std = \
        su.compute_weighted_C_variance(C_prior_check, config)

    all_hist, all_K, all_C = [], [], []
    for trial in range(config.NUM_RANDOM_TRIALS):
        seed = config.SEED + 1000 + sample_idx * 100 + trial
        h, K_m, C_m = run_random_single_trial(
            seed, K_true, H_true, C_true, k_train_data, vae, fno_kh, fno_kc, config)
        all_hist.append(h)
        all_K.append(K_m)
        all_C.append(C_m)

    return {
        'all_trial_histories': all_hist,
        'all_trial_K_means':   np.array(all_K),
        'all_trial_C_means':   np.array(all_C),
        'initial_C_var':       initial_C_var,
        'initial_mean_std':    initial_mean_std,
    }


# ==========================================
# LHS（10 次 trial，非自适应空间填充基线，与 Random 完全对齐）
# ==========================================
# 设计原理
#   D1 同化协议   : sequential ES-MDA（与所有方法一致）
#   D2 选井时机   : pre-determined（每次 trial 内部井位一次性确定）
#   D3 选井准则   : space-filling（Latin Hypercube）
#   D4 试验次数   : NUM_RANDOM_TRIALS (=10) —— 与 random 完全对齐
#
# 井位序列构造（每次 trial）：
#   - 前 3 口 = init_wells_lhs(INITIAL_WELLS, ..., config.SEED)
#       ↑ 仍固定用 config.SEED：所有方法/所有 trial/所有样本共享同一起跑线，
#         保证 LHS 与 Random 的对比纯净（差异只来自后 17 口选井准则）
#   - 后 17 口 = init_wells_lhs(TARGET_WELLS, ..., trial_seed) 与前 3 口的差集
#       ↑ trial_seed = config.SEED + 3000 + sample_idx*100 + trial
#         跨样本独立（sample_idx 维度），跨 trial 独立（trial 维度）
#         offset 3000 与 random(1000) 错开避免 seed 重合
#
# 与 Random 的核心可比性：
#   - 同样运行 NUM_RANDOM_TRIALS 次；
#   - 同样使用 mean ± std 作为汇总统计；
#   - 起跑线（前 3 口）完全一致；
#   - 唯一差异：random 用 np.random.choice 抽签；lhs 用 LHS 拉丁超立方设计
#   → 性能差异可干净归因到"空间填充设计 vs 均匀随机"本身
# ==========================================
def run_lhs_single_trial(trial_seed, K_true, H_true, C_true, k_train_data,
                         vae, fno_kh, fno_kc, config):
    """单次 LHS trial：给定 trial_seed，生成一组确定性的 LHS-20 井位并跑完整序贯 ES-MDA。"""
    # 前 3 口：固定用 config.SEED，与所有方法/所有 trial 共享起跑线
    init_wells = su.init_wells_lhs(config.INITIAL_WELLS, config.GRID_SIZE, config.SEED)

    # 母 LHS-20 设计：由 trial_seed 控制，每个 (sample, trial) 组合一组独立设计
    lhs20 = su.init_wells_lhs(config.TARGET_WELLS, config.GRID_SIZE, trial_seed)

    init_set     = set(init_wells)
    future_wells = [w for w in lhs20 if w not in init_set]

    # 兜底：若 LHS 自身去重 + 与前 3 口重叠后井数不足，从更大 LHS 池补齐
    n_needed = config.TARGET_WELLS - config.INITIAL_WELLS
    if len(future_wells) < n_needed:
        existing = init_set | set(future_wells)
        for extra_n in (config.TARGET_WELLS * 2,
                        config.TARGET_WELLS * 4,
                        config.GRID_SIZE * config.GRID_SIZE):
            pool = su.init_wells_lhs(extra_n, config.GRID_SIZE, trial_seed)
            for w in pool:
                if w not in existing:
                    future_wells.append(w)
                    existing.add(w)
                    if len(future_wells) >= n_needed:
                        break
            if len(future_wells) >= n_needed:
                break
    future_wells = future_wells[:n_needed]
    full_seq     = list(init_wells) + future_wells

    # 先验集合（同所有方法严格一致）
    z_prior = su.generate_prior(vae, k_train_data, config.N_ENSEMBLE, config, config.SEED)
    z_curr  = z_prior.copy()
    history = []

    for i in range(len(full_seq) - config.INITIAL_WELLS + 1):
        step_wells     = full_seq[:config.INITIAL_WELLS + i]
        h_obs          = su.get_observations(H_true, step_wells)
        K_post, z_post = su.run_esmda(vae, fno_kh, z_curr, step_wells, h_obs, config)
        z_curr         = z_post.copy()

        C_post  = su.predict_C_batch(K_post, fno_kc, config)
        metrics = su.evaluate_prediction(C_post, C_true, K_post, K_true, config)
        metrics['n_wells'] = len(step_wells)
        metrics['wells']   = [list(w) for w in step_wells]
        history.append(metrics)

    return history, np.mean(K_post, axis=0), np.mean(C_post, axis=0)


def run_lhs(sample_idx, K_true, H_true, C_true, k_train_data,
            vae, fno_kh, fno_kc, config):
    """LHS：每样本运行 NUM_RANDOM_TRIALS 次，输出与 random 完全对齐的多 trial 结构。"""
    # 计算初始（先验）的 C 方差，仅用 config.SEED 一次即可（先验在所有 trial 间共享）
    z_prior_check = su.generate_prior(
        vae, k_train_data, config.N_ENSEMBLE, config, config.SEED)
    K_prior_check = su.vae_decode(z_prior_check, vae, config)
    C_prior_check = su.predict_C_batch(K_prior_check, fno_kc, config)
    initial_C_var, _, _, _, initial_mean_std = \
        su.compute_weighted_C_variance(C_prior_check, config)

    all_hist, all_K, all_C = [], [], []
    for trial in range(config.NUM_RANDOM_TRIALS):
        # offset 3000 与 random(1000) 错开，避免任何 seed 重合
        seed = config.SEED + 3000 + sample_idx * 100 + trial
        h, K_m, C_m = run_lhs_single_trial(
            seed, K_true, H_true, C_true, k_train_data, vae, fno_kh, fno_kc, config)
        all_hist.append(h)
        all_K.append(K_m)
        all_C.append(C_m)

    return {
        'all_trial_histories': all_hist,
        'all_trial_K_means':   np.array(all_K),
        'all_trial_C_means':   np.array(all_C),
        'initial_C_var':       initial_C_var,
        'initial_mean_std':    initial_mean_std,
    }


# ==========================================
# 保存 / 进度
# ==========================================
def save_sample_result(output_dir, sample_idx, method_name, result):
    sample_dir = os.path.join(output_dir, f'sample_{sample_idx:03d}')
    os.makedirs(sample_dir, exist_ok=True)
    save_path  = os.path.join(sample_dir, f'{method_name}.npz')
    save_dict  = {}
    for key, val in result.items():
        if isinstance(val, np.ndarray):
            save_dict[key] = val
        elif isinstance(val, list):
            save_dict[key] = np.array(val, dtype=object)
        else:
            save_dict[key] = val
    np.savez(save_path, **save_dict)


def check_sample_done(output_dir, sample_idx, method_name):
    path = os.path.join(output_dir, f'sample_{sample_idx:03d}', f'{method_name}.npz')
    return os.path.exists(path)


def save_progress(output_dir, progress):
    with open(os.path.join(output_dir, 'progress.json'), 'w') as f:
        json.dump(progress, f, indent=2)


def load_progress(output_dir):
    path = os.path.join(output_dir, 'progress.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {'completed_samples': {}, 'start_time': None}


# ==========================================
# 主函数
# ==========================================
def main(methods=('active', 'greedy', 'random', 'lhs'),
         start=0, end=100, resume=True):
    """
    100-sample 实验主入口。

    参数
    ----
    methods : tuple/list of str
        要运行的方法子集，可选 'active' / 'greedy' / 'random' / 'lhs'；其中 'active' 为历史内部键，对外对应 Look-ahead。
        默认全部四种。
    start, end : int
        样本索引范围，半开区间 [start, end)。默认 0~100。
    resume : bool
        是否启用断点续算（跳过已存在的 npz 文件）。默认 True。
        resume=True 时已生成的 npz 不会被重跑、不会被覆盖。
    """
    methods = list(methods)
    valid   = {'active', 'greedy', 'random', 'lhs'}
    invalid = [m for m in methods if m not in valid]
    if invalid:
        raise ValueError(f"未知方法: {invalid}，可选 {sorted(valid)}")

    config = TestConfig()
    os.makedirs(config.OUTPUT_ROOT, exist_ok=True)

    print("=" * 70)
    print(f"  100-Sample Test  |  methods={methods}  "
          f"samples={start}~{end-1}  resume={resume}")
    print("=" * 70)

    vae, fno_kh, fno_kc = su.load_all_models(config)
    K_all, H_all, C_all, k_train_data = su.load_test_data(config)
    print(f"  [Info] sigma_eff = sqrt({config.OBS_NOISE_STD**2:.4f} + "
          f"{config.SIGMA_F1**2:.4f}) = "
          f"{(config.OBS_NOISE_STD**2 + config.SIGMA_F1**2)**0.5:.4f}")

    total_start = time.time()
    progress    = load_progress(config.OUTPUT_ROOT)
    if progress['start_time'] is None:
        progress['start_time'] = time.strftime('%Y-%m-%d %H:%M:%S')

    for sample_idx in range(start, end):
        print(f"\n{'='*70}")
        print(f"  Sample {sample_idx}/{end-1}  "
              f"[{(time.time()-total_start)/60:.1f} min elapsed]")
        print(f"{'='*70}")

        K_true, H_true, C_true = su.get_test_sample(K_all, H_all, C_all, sample_idx)
        sk = f'sample_{sample_idx:03d}'
        progress['completed_samples'].setdefault(sk, {})

        # ---- Look-ahead ----
        if 'active' in methods:
            if resume and check_sample_done(config.OUTPUT_ROOT, sample_idx, 'active'):
                print(f"  [Look-ahead] already done, skipping.")
            else:
                print(f"\n  --- Look-ahead (sample {sample_idx}) ---")
                t0 = time.time()
                su.set_seed(config.SEED)
                result  = run_look_ahead(sample_idx, K_true, H_true, C_true,
                                              k_train_data, vae, fno_kh, fno_kc, config)
                elapsed = time.time() - t0
                save_sample_result(config.OUTPUT_ROOT, sample_idx, 'active', result)
                final = result['history'][-1]
                print(f"  [Look-ahead] {elapsed/60:.1f} min | "
                      f"K_R2={final['K_r2']:.4f}  C_R2={final['C_r2']:.4f}  "
                      f"mean_std={final['C_mean_std']:.4f} "
                      f"(量程{final['C_mean_std']*100:.1f}%)")
                progress['completed_samples'][sk]['active'] = {
                    'time_min':         round(elapsed / 60, 2),
                    'K_r2':             round(final['K_r2'], 4),
                    'C_r2':             round(final['C_r2'], 4),
                    'final_C_var':      round(float(final['C_var']), 4),
                    'final_mean_std':   round(float(final['C_mean_std']), 6),
                    'initial_mean_std': round(float(result['initial_mean_std']), 6),
                }

        # ---- Greedy ----
        if 'greedy' in methods:
            if resume and check_sample_done(config.OUTPUT_ROOT, sample_idx, 'greedy'):
                print(f"  [Greedy] already done, skipping.")
            else:
                print(f"\n  --- Greedy MaxVar (sample {sample_idx}) ---")
                t0 = time.time()
                su.set_seed(config.SEED)
                result  = run_greedy(sample_idx, K_true, H_true, C_true,
                                     k_train_data, vae, fno_kh, fno_kc, config)
                elapsed = time.time() - t0
                save_sample_result(config.OUTPUT_ROOT, sample_idx, 'greedy', result)
                final = result['history'][-1]
                print(f"  [Greedy] {elapsed/60:.1f} min | "
                      f"K_R2={final['K_r2']:.4f}  C_R2={final['C_r2']:.4f}  "
                      f"mean_std={final['C_mean_std']:.4f} "
                      f"(量程{final['C_mean_std']*100:.1f}%)")
                progress['completed_samples'][sk]['greedy'] = {
                    'time_min':         round(elapsed / 60, 2),
                    'K_r2':             round(final['K_r2'], 4),
                    'C_r2':             round(final['C_r2'], 4),
                    'final_C_var':      round(float(final['C_var']), 4),
                    'final_mean_std':   round(float(final['C_mean_std']), 6),
                    'initial_mean_std': round(float(result['initial_mean_std']), 6),
                }

        # ---- Random (10 trials) ----
        if 'random' in methods:
            if resume and check_sample_done(config.OUTPUT_ROOT, sample_idx, 'random'):
                print(f"  [Random] already done, skipping.")
            else:
                print(f"\n  --- Random ({config.NUM_RANDOM_TRIALS} trials, "
                      f"sample {sample_idx}) ---")
                t0 = time.time()
                result  = run_random(sample_idx, K_true, H_true, C_true,
                                     k_train_data, vae, fno_kh, fno_kc, config)
                elapsed = time.time() - t0
                save_sample_result(config.OUTPUT_ROOT, sample_idx, 'random', result)
                r2s       = [h[-1]['C_r2']       for h in result['all_trial_histories']]
                mean_stds = [h[-1]['C_mean_std']  for h in result['all_trial_histories']]
                print(f"  [Random] {elapsed/60:.1f} min | "
                      f"C_R2={np.mean(r2s):.4f}±{np.std(r2s):.4f}  "
                      f"mean_std={np.mean(mean_stds):.4f}±{np.std(mean_stds):.4f} "
                      f"(量程{np.mean(mean_stds)*100:.1f}%)")
                progress['completed_samples'][sk]['random'] = {
                    'time_min':         round(elapsed / 60, 2),
                    'C_r2_mean':        round(float(np.mean(r2s)), 4),
                    'C_r2_std':         round(float(np.std(r2s)), 4),
                    'mean_std_mean':    round(float(np.mean(mean_stds)), 6),
                    'mean_std_std':     round(float(np.std(mean_stds)), 6),
                    'initial_mean_std': round(float(result['initial_mean_std']), 6),
                }

        # ---- LHS (10 trials, 与 Random 完全对齐的非自适应基线) ----
        if 'lhs' in methods:
            if resume and check_sample_done(config.OUTPUT_ROOT, sample_idx, 'lhs'):
                print(f"  [LHS] already done, skipping.")
            else:
                print(f"\n  --- LHS ({config.NUM_RANDOM_TRIALS} trials, "
                      f"sample {sample_idx}) ---")
                t0 = time.time()
                result  = run_lhs(sample_idx, K_true, H_true, C_true,
                                  k_train_data, vae, fno_kh, fno_kc, config)
                elapsed = time.time() - t0
                save_sample_result(config.OUTPUT_ROOT, sample_idx, 'lhs', result)
                r2s       = [h[-1]['C_r2']       for h in result['all_trial_histories']]
                mean_stds = [h[-1]['C_mean_std']  for h in result['all_trial_histories']]
                print(f"  [LHS]    {elapsed/60:.1f} min | "
                      f"C_R2={np.mean(r2s):.4f}±{np.std(r2s):.4f}  "
                      f"mean_std={np.mean(mean_stds):.4f}±{np.std(mean_stds):.4f} "
                      f"(量程{np.mean(mean_stds)*100:.1f}%)")
                progress['completed_samples'][sk]['lhs'] = {
                    'time_min':         round(elapsed / 60, 2),
                    'C_r2_mean':        round(float(np.mean(r2s)), 4),
                    'C_r2_std':         round(float(np.std(r2s)), 4),
                    'mean_std_mean':    round(float(np.mean(mean_stds)), 6),
                    'mean_std_std':     round(float(np.std(mean_stds)), 6),
                    'initial_mean_std': round(float(result['initial_mean_std']), 6),
                }

        save_progress(config.OUTPUT_ROOT, progress)

    # ── 汇总 ──
    total_h = (time.time() - total_start) / 3600
    print(f"\n{'='*70}")
    print(f"  Done!  Total: {total_h:.2f} h  →  {config.OUTPUT_ROOT}/")
    print(f"{'='*70}")
    print(f"\n{'Method':>10}  {'C_R2':>18}  {'mean_std':>22}  {'n':>4}")
    print("-" * 60)
    # 与 random 一样使用 mean±std 汇总的方法：random / lhs
    multi_trial_methods = {'random', 'lhs'}
    for method in methods:
        r2_vals, std_vals = [], []
        for si in range(start, end):
            info = progress['completed_samples'].get(
                f'sample_{si:03d}', {}).get(method)
            if info:
                if method in multi_trial_methods:
                    r2_vals.append(info.get('C_r2_mean', np.nan))
                    std_vals.append(info.get('mean_std_mean', np.nan))
                else:
                    r2_vals.append(info.get('C_r2', np.nan))
                    std_vals.append(info.get('final_mean_std', np.nan))
        if r2_vals:
            r2_arr  = np.array(r2_vals)
            std_arr = np.array(std_vals)
            print(f"  {method:>8s}:  "
                  f"C_R2={np.mean(r2_arr):.4f}±{np.std(r2_arr):.4f}  "
                  f"mean_std={np.mean(std_arr):.4f}±{np.std(std_arr):.4f}  "
                  f"n={len(r2_arr)}")
    save_progress(config.OUTPUT_ROOT, progress)


# ==========================================
# 代码启动配置区（直接运行本文件即可）
# ==========================================
# 修改以下变量来控制运行行为：
#   METHODS   : 要跑的方法子集。已有结果会被自动跳过（RESUME=True）。
#               默认 ['active', 'greedy', 'random', 'lhs'] —— 四方法全跑；'active' 对外显示为 Look-ahead
#   START_IDX, END_IDX : 样本范围（半开区间），默认 0~100
#   RESUME    : 断点续算开关。
#               -- True  : 跳过已存在的 npz（不会覆盖之前的结果）
#               -- False : 重跑所有指定方法（会覆盖旧 npz，慎用！）
# ==========================================
if __name__ == "__main__":
    METHODS   = ['active', 'greedy', 'random', 'lhs']
    START_IDX = 0
    END_IDX   = 100
    RESUME    = True

    main(methods=METHODS, start=START_IDX, end=END_IDX, resume=RESUME)
