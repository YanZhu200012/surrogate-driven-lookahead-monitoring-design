"""
run_selected_samples.py
对选定的少量样本生成完整数据（用于论文空间场展示）

"""

import os
import time
import numpy as np
import shared_utils as su


# ==========================================
# 用户自定义运行参数（直接修改这里即可，无需命令行输入）
# ==========================================
SELECTED_SAMPLES   = [14]          # ← 这里修改你要运行的样本索引列表（可任意增删）
SELECTED_METHODS   = ['active', 'greedy', 'random', 'lhs']  # 可选: 'active'/'greedy'/'random'/'lhs'
RESUME_MODE        = True                   # True = 跳过已生成的文件
CUSTOM_OUTPUT_ROOT = None                   # None = 使用 SelectedConfig 默认路径


# ==========================================
# 控制台输出样式（与 evaluate_all.py 完全一致）
# ==========================================
def print_big_header(title):
    """大标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_small_header(title):
    """小标题"""
    print("\n" + "=" * 52)
    print(f"  {title}")
    print("=" * 52)


class SelectedConfig(su.BaseConfig):
    OUTPUT_ROOT          = './test_selected_results'
    VIRTUAL_ALPHA_LIST   = [4.0] * 4
    USE_CANDIDATE_FILTER = True
    MAX_CANDIDATES       = 500
    VARIANCE_PERCENTILE  = 30
    MIN_WELL_SPACING     = 2
    NUM_RANDOM_TRIALS    = 10


# ==========================================
# Active Learning（逻辑保持不变）
# ==========================================
def run_active_learning(sample_idx, K_true, H_true, C_true, k_train_data,
                        vae, fno_kh, fno_kc, config):
    wells   = su.init_wells_lhs(config.INITIAL_WELLS, config.GRID_SIZE, config.SEED)
    z_prior = su.generate_prior(vae, k_train_data, config.N_ENSEMBLE, config, config.SEED)
    z_curr  = z_prior.copy()
    history = []

    K_prior = su.vae_decode(z_prior, vae, config)
    C_prior = su.predict_C_batch(K_prior, fno_kc, config)
    initial_C_var, _, _, _, initial_mean_std = \
        su.compute_weighted_C_variance(C_prior, config)
    print(f"  [Info] 初始先验 mean_std={initial_mean_std:.4f} "
          f"(量程{initial_mean_std*100:.1f}%)")

    while len(wells) < config.TARGET_WELLS:
        h_obs          = su.get_observations(H_true, wells)
        K_post, z_post = su.run_esmda(vae, fno_kh, z_curr, wells, h_obs, config)
        z_curr         = z_post.copy()

        C_post  = su.predict_C_batch(K_post, fno_kc, config)
        metrics = su.evaluate_prediction(C_post, C_true, K_post, K_true, config)
        metrics['n_wells'] = len(wells)
        metrics['wells']   = [list(w) for w in wells]

        if len(wells) < config.TARGET_WELLS:
            new_well, _, _ = su.select_optimal_well(
                z_post, wells, h_obs, fno_kh, fno_kc, vae, config)
            if new_well:
                wells.append(new_well)
            else:
                break
        history.append(metrics)

    h_obs          = su.get_observations(H_true, wells)
    K_post, z_post = su.run_esmda(vae, fno_kh, z_curr, wells, h_obs, config)
    C_post         = su.predict_C_batch(K_post, fno_kc, config)
    final_metrics  = su.evaluate_prediction(C_post, C_true, K_post, K_true, config)
    final_metrics['n_wells'] = len(wells)
    final_metrics['wells']   = [list(w) for w in wells]
    history.append(final_metrics)

    print(f"  [Active] 最终 mean_std={final_metrics['C_mean_std']:.4f} "
          f"(量程{final_metrics['C_mean_std']*100:.1f}%)")

    return {
        'history':          history,
        'final_wells':      [list(w) for w in wells],
        'K_samples_final':  K_post,
        'C_samples_final':  C_post,
        'K_true':           K_true,
        'H_true':           H_true,
        'C_true':           C_true,
        'initial_C_var':    initial_C_var,
        'initial_mean_std': initial_mean_std,
    }


# ==========================================
# Greedy（逻辑保持不变）
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
    print(f"  [Info] 初始先验 mean_std={initial_mean_std:.4f} "
          f"(量程{initial_mean_std*100:.1f}%)")

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

    print(f"  [Greedy] 最终 mean_std={final_metrics['C_mean_std']:.4f} "
          f"(量程{final_metrics['C_mean_std']*100:.1f}%)")

    return {
        'history':          history,
        'final_wells':      [list(w) for w in wells],
        'K_samples_final':  K_post,
        'C_samples_final':  C_post,
        'K_true':           K_true,
        'H_true':           H_true,
        'C_true':           C_true,
        'initial_C_var':    initial_C_var,
        'initial_mean_std': initial_mean_std,
    }


# ==========================================
# Random（逻辑保持不变）
# ==========================================
def run_random_single_trial(trial_seed, K_true, H_true, C_true, k_train_data,
                            vae, fno_kh, fno_kc, config):
    su.set_seed(trial_seed)
    wells    = su.init_wells_lhs(config.INITIAL_WELLS, config.GRID_SIZE, config.SEED)
    existing = set(wells)
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

    return history, K_post, C_post


def run_random(sample_idx, K_true, H_true, C_true, k_train_data,
               vae, fno_kh, fno_kc, config):
    z_prior_check = su.generate_prior(
        vae, k_train_data, config.N_ENSEMBLE, config, config.SEED)
    K_prior_check = su.vae_decode(z_prior_check, vae, config)
    C_prior_check = su.predict_C_batch(K_prior_check, fno_kc, config)
    initial_C_var, _, _, _, initial_mean_std = \
        su.compute_weighted_C_variance(C_prior_check, config)
    print(f"  [Info] 初始先验 mean_std={initial_mean_std:.4f} "
          f"(量程{initial_mean_std*100:.1f}%)")

    all_hist, all_K, all_C = [], [], []
    for trial in range(config.NUM_RANDOM_TRIALS):
        seed = config.SEED + 1000 + sample_idx * 100 + trial
        h, K_p, C_p = run_random_single_trial(
            seed, K_true, H_true, C_true, k_train_data, vae, fno_kh, fno_kc, config)
        all_hist.append(h)
        all_K.append(K_p)
        all_C.append(C_p)

    final_mean_stds = [h[-1]['C_mean_std'] for h in all_hist]
    print(f"  [Random] 最终 mean_std={np.mean(final_mean_stds):.4f}"
          f"±{np.std(final_mean_stds):.4f} "
          f"(量程{np.mean(final_mean_stds)*100:.1f}%)")

    return {
        'all_trial_histories':  all_hist,
        'all_final_K_samples':  np.array(all_K),
        'all_final_C_samples':  np.array(all_C),
        'K_true':               K_true,
        'H_true':               H_true,
        'C_true':               C_true,
        'initial_C_var':        initial_C_var,
        'initial_mean_std':     initial_mean_std,
    }


# ==========================================
# LHS-20 (non-adaptive, 序贯 ES-MDA)
# ==========================================
# 输出 schema 与 active/greedy 完全对齐：
#   K_samples_final / C_samples_final / history / final_wells / K_true / H_true / C_true
# 因此下游 figure_plot.py 能用统一的接口加载和可视化 LHS 结果，
# 且 check_done() 检测的字段 'K_samples_final' 也能被正确识别。
# ==========================================
def run_lhs(sample_idx, K_true, H_true, C_true, k_train_data,
            vae, fno_kh, fno_kc, config):
    z_prior = su.generate_prior(vae, k_train_data, config.N_ENSEMBLE, config, config.SEED)
    z_curr  = z_prior.copy()

    K_prior = su.vae_decode(z_prior, vae, config)
    C_prior = su.predict_C_batch(K_prior, fno_kc, config)
    initial_C_var, _, _, _, initial_mean_std = \
        su.compute_weighted_C_variance(C_prior, config)
    print(f"  [Info] 初始先验 mean_std={initial_mean_std:.4f} "
          f"(量程{initial_mean_std*100:.1f}%)")

    # ── 构造 20 口井位序列 ──
    # 前 3 口：seed 固定为 config.SEED，与 active/greedy/random 跨样本共享起跑线
    init_wells = su.init_wells_lhs(config.INITIAL_WELLS, config.GRID_SIZE, config.SEED)

    # 母 LHS-20 设计：seed 与样本绑定，使每个样本得到独立 LHS 设计
    # （v3 修复：与 run_100.py 保持一致，避免 100 样本共享同一组设计的偏倚）
    lhs_sample_seed = config.SEED + 2000 + sample_idx * 100
    lhs20 = su.init_wells_lhs(config.TARGET_WELLS, config.GRID_SIZE, lhs_sample_seed)

    init_set     = set(init_wells)
    future_wells = [w for w in lhs20 if w not in init_set]

    # 兜底：去重后井数不足时从更大 LHS 池补齐（同样使用 lhs_sample_seed）
    n_needed = config.TARGET_WELLS - config.INITIAL_WELLS
    if len(future_wells) < n_needed:
        existing = init_set | set(future_wells)
        for extra_n in (config.TARGET_WELLS * 2,
                        config.TARGET_WELLS * 4,
                        config.GRID_SIZE * config.GRID_SIZE):
            pool = su.init_wells_lhs(extra_n, config.GRID_SIZE, lhs_sample_seed)
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

    # ── 序贯 ES-MDA：逐口加井、每步同化、记录学习曲线 ──
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

    final_metrics = history[-1]
    print(f"  [LHS] 最终 mean_std={final_metrics['C_mean_std']:.4f} "
          f"(量程{final_metrics['C_mean_std']*100:.1f}%)")

    return {
        'history':          history,
        'final_wells':      [list(w) for w in full_seq],
        'K_samples_final':  K_post,
        'C_samples_final':  C_post,
        'K_true':           K_true,
        'H_true':           H_true,
        'C_true':           C_true,
        'initial_C_var':    initial_C_var,
        'initial_mean_std': initial_mean_std,
    }


# ==========================================
# 保存与检查（逻辑保持不变）
# ==========================================
def save_result(output_dir, sample_idx, method_name, result):
    sample_dir = os.path.join(output_dir, f'sample_{sample_idx:03d}')
    os.makedirs(sample_dir, exist_ok=True)
    save_path = os.path.join(sample_dir, f'{method_name}.npz')
    save_dict = {}
    for key, val in result.items():
        if isinstance(val, np.ndarray):
            save_dict[key] = val
        elif isinstance(val, list):
            save_dict[key] = np.array(val, dtype=object)
        else:
            save_dict[key] = val
    np.savez(save_path, **save_dict)
    size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f"    Saved: {save_path} ({size_mb:.1f} MB)")


def check_done(output_dir, sample_idx, method_name):
    path = os.path.join(output_dir, f'sample_{sample_idx:03d}', f'{method_name}.npz')
    if not os.path.exists(path):
        return False
    try:
        data = np.load(path, allow_pickle=True)
        key  = 'all_final_K_samples' if method_name == 'random' else 'K_samples_final'
        ok   = key in data and 'K_true' in data
        data.close()
        return ok
    except Exception:
        return False


# ==========================================
# 主函数
# ==========================================
def main():
    config = SelectedConfig()
    if CUSTOM_OUTPUT_ROOT:
        config.OUTPUT_ROOT = CUSTOM_OUTPUT_ROOT
    os.makedirs(config.OUTPUT_ROOT, exist_ok=True)

    print_big_header("Selected Samples Data Generation")
    print(f"  samples = {SELECTED_SAMPLES} | methods = {SELECTED_METHODS} | resume = {RESUME_MODE}")

    vae, fno_kh, fno_kc = su.load_all_models(config)
    K_all, H_all, C_all, k_train_data = su.load_test_data(config)
    print(f"  [Info] sigma_f1={config.SIGMA_F1:.5f}  "
          f"sigma_eff={(config.OBS_NOISE_STD**2 + config.SIGMA_F1**2)**0.5:.5f}")

    total_start = time.time()
    summary     = []

    for idx, sample_idx in enumerate(SELECTED_SAMPLES):
        print_big_header(f"Sample {sample_idx}  [{idx+1}/{len(SELECTED_SAMPLES)}]")

        K_true, H_true, C_true = su.get_test_sample(K_all, H_all, C_all, sample_idx)
        s = {'sample_idx': sample_idx}

        if 'active' in SELECTED_METHODS:
            if RESUME_MODE and check_done(config.OUTPUT_ROOT, sample_idx, 'active'):
                print("  [Active] already done, skipping.")
            else:
                print_small_header("Active Learning")
                print("  [Active] Running...")
                t0 = time.time()
                su.set_seed(config.SEED)
                result  = run_active_learning(sample_idx, K_true, H_true, C_true,
                                              k_train_data, vae, fno_kh, fno_kc, config)
                elapsed = time.time() - t0
                save_result(config.OUTPUT_ROOT, sample_idx, 'active', result)
                final = result['history'][-1]
                print(f"  → {elapsed/60:.1f} min | "
                      f"K_R² = {final['K_r2']:.4f}   "
                      f"C_R² = {final['C_r2']:.4f}   "
                      f"mean_std = {final['C_mean_std']:.4f}")
                s['active_K_r2']     = final['K_r2']
                s['active_C_r2']     = final['C_r2']
                s['active_mean_std'] = final['C_mean_std']

        if 'greedy' in SELECTED_METHODS:
            if RESUME_MODE and check_done(config.OUTPUT_ROOT, sample_idx, 'greedy'):
                print("  [Greedy] already done, skipping.")
            else:
                print_small_header("Greedy (MaxVar)")
                print("  [Greedy] Running...")
                t0 = time.time()
                su.set_seed(config.SEED)
                result  = run_greedy(sample_idx, K_true, H_true, C_true,
                                     k_train_data, vae, fno_kh, fno_kc, config)
                elapsed = time.time() - t0
                save_result(config.OUTPUT_ROOT, sample_idx, 'greedy', result)
                final = result['history'][-1]
                print(f"  → {elapsed/60:.1f} min | "
                      f"K_R² = {final['K_r2']:.4f}   "
                      f"C_R² = {final['C_r2']:.4f}   "
                      f"mean_std = {final['C_mean_std']:.4f}")
                s['greedy_K_r2']     = final['K_r2']
                s['greedy_C_r2']     = final['C_r2']
                s['greedy_mean_std'] = final['C_mean_std']

        if 'random' in SELECTED_METHODS:
            if RESUME_MODE and check_done(config.OUTPUT_ROOT, sample_idx, 'random'):
                print("  [Random] already done, skipping.")
            else:
                print_small_header("Random (10 trials)")
                print(f"  [Random] Running {config.NUM_RANDOM_TRIALS} trials...")
                t0 = time.time()
                result  = run_random(sample_idx, K_true, H_true, C_true,
                                     k_train_data, vae, fno_kh, fno_kc, config)
                elapsed = time.time() - t0
                save_result(config.OUTPUT_ROOT, sample_idx, 'random', result)
                r2s       = [h[-1]['C_r2']      for h in result['all_trial_histories']]
                mean_stds = [h[-1]['C_mean_std'] for h in result['all_trial_histories']]
                print(f"  → {elapsed/60:.1f} min | "
                      f"C_R² = {np.mean(r2s):.4f}±{np.std(r2s):.4f}   "
                      f"mean_std = {np.mean(mean_stds):.4f}±{np.std(mean_stds):.4f}")
                s['random_C_r2_mean']    = float(np.mean(r2s))
                s['random_mean_std_mean'] = float(np.mean(mean_stds))

        if 'lhs' in SELECTED_METHODS:
            if RESUME_MODE and check_done(config.OUTPUT_ROOT, sample_idx, 'lhs'):
                print("  [LHS] already done, skipping.")
            else:
                print_small_header(f"LHS-{config.TARGET_WELLS} (non-adaptive)")
                print("  [LHS] Running...")
                t0 = time.time()
                su.set_seed(config.SEED)
                result  = run_lhs(sample_idx, K_true, H_true, C_true,
                                  k_train_data, vae, fno_kh, fno_kc, config)
                elapsed = time.time() - t0
                save_result(config.OUTPUT_ROOT, sample_idx, 'lhs', result)
                final = result['history'][-1]
                print(f"  → {elapsed/60:.1f} min | "
                      f"K_R² = {final['K_r2']:.4f}   "
                      f"C_R² = {final['C_r2']:.4f}   "
                      f"mean_std = {final['C_mean_std']:.4f}")
                s['lhs_K_r2']     = final['K_r2']
                s['lhs_C_r2']     = final['C_r2']
                s['lhs_mean_std'] = final['C_mean_std']

        summary.append(s)

    total_min = (time.time() - total_start) / 60
    print_big_header(f"Batch Processing Completed ({total_min:.1f} min)")

    # ==================== 最终指标汇总（4 方法） ====================
    print("\n" + "=" * 110)
    print("  最终指标汇总（与 evaluate_all.py 风格一致）")
    print("=" * 110)
    print(f"{'Sample':<7} "
          f"{'AL K':<10} {'AL C':<10} {'AL std':<10} "
          f"{'Gd K':<10} {'Gd C':<10} {'Gd std':<10} "
          f"{'Rd C':<10} {'Rd std':<10} "
          f"{'LHS K':<10} {'LHS C':<10} {'LHS std':<10}")
    print("-" * 110)

    active_k, active_c, active_std = [], [], []
    greedy_k, greedy_c, greedy_std = [], [], []
    random_c, random_std           = [], []
    lhs_k, lhs_c, lhs_std          = [], [], []

    def _fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)

    for s in summary:
        print(f"{s['sample_idx']:<7} "
              f"{_fmt(s.get('active_K_r2',         '-')):<10} "
              f"{_fmt(s.get('active_C_r2',         '-')):<10} "
              f"{_fmt(s.get('active_mean_std',     '-')):<10} "
              f"{_fmt(s.get('greedy_K_r2',         '-')):<10} "
              f"{_fmt(s.get('greedy_C_r2',         '-')):<10} "
              f"{_fmt(s.get('greedy_mean_std',     '-')):<10} "
              f"{_fmt(s.get('random_C_r2_mean',    '-')):<10} "
              f"{_fmt(s.get('random_mean_std_mean','-')):<10} "
              f"{_fmt(s.get('lhs_K_r2',            '-')):<10} "
              f"{_fmt(s.get('lhs_C_r2',            '-')):<10} "
              f"{_fmt(s.get('lhs_mean_std',        '-')):<10}")

        if 'active_K_r2' in s:
            active_k.append(s['active_K_r2'])
            active_c.append(s['active_C_r2'])
            active_std.append(s['active_mean_std'])
        if 'greedy_K_r2' in s:
            greedy_k.append(s['greedy_K_r2'])
            greedy_c.append(s['greedy_C_r2'])
            greedy_std.append(s['greedy_mean_std'])
        if 'random_C_r2_mean' in s:
            random_c.append(s['random_C_r2_mean'])
            random_std.append(s['random_mean_std_mean'])
        if 'lhs_K_r2' in s:
            lhs_k.append(s['lhs_K_r2'])
            lhs_c.append(s['lhs_C_r2'])
            lhs_std.append(s['lhs_mean_std'])

    # 平均行
    print("-" * 110)
    def _avg(lst):
        return f"{np.mean(lst):.4f}" if lst else "-"
    print(f"{'Avg':<7} "
          f"{_avg(active_k):<10} {_avg(active_c):<10} {_avg(active_std):<10} "
          f"{_avg(greedy_k):<10} {_avg(greedy_c):<10} {_avg(greedy_std):<10} "
          f"{_avg(random_c):<10} {_avg(random_std):<10} "
          f"{_avg(lhs_k):<10} {_avg(lhs_c):<10} {_avg(lhs_std):<10}")
    print("=" * 110)
    print(f"\nAll outputs → {config.OUTPUT_ROOT}")


if __name__ == "__main__":
    main()