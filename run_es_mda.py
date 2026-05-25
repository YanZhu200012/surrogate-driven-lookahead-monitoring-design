"""
run_es_mda.py
ES-MDA 运行脚本（训练/评估分离版本）

目标：验证 ES-MDA 同化算法本身的有效性，与选井策略解耦。
      对 100 个测试场分别计算：
        1. 先验（Prior）：未同化，仅基于训练集先验分布
        2. 后验-3井（Posterior-3w）：用 3 口 LHS 初始井同化
        3. 后验-20井（Posterior-20w）：用 20 口 LHS 随机井同化

      三组井位均由 LHS 随机生成，与 Active Learning 中的自适应选井无关，
      从而单独验证 ES-MDA 本身带来的不确定性缩减效果。

输出目录：./es_mda_results/
  sample_000.npz, sample_001.npz, ...
  progress.json
"""

import os
import time
import json
import numpy as np
import shared_utils as su


class EsMdaConfig(su.BaseConfig):
    OUTPUT_ROOT  = './es_mda_results'
    INITIAL_WELLS = 3    # 先验→后验-3井
    TARGET_WELLS  = 20   # 先验→后验-20井（全部 LHS 随机，非自适应选井）


# ==========================================
# 单样本运行
# ==========================================
def run_single_sample(sample_idx, K_true, H_true, C_true, k_train_data,
                      vae, fno_kh, fno_kc, config):
    """
    对一个测试场计算先验与两组后验。

    返回字典包含各阶段的均值场、标准差场及评估指标。
    """
    # ── 先验集合 ──
    z_prior  = su.generate_prior(vae, k_train_data, config.N_ENSEMBLE, config,
                                  seed=config.SEED)
    K_prior  = su.vae_decode(z_prior, vae, config)
    C_prior  = su.predict_C_batch(K_prior, fno_kc, config)
    prior_C_var, _, _, _, prior_mean_std = su.compute_weighted_C_variance(C_prior, config)

    # ── 3 口 LHS 井 ──
    wells_3 = su.init_wells_lhs(config.INITIAL_WELLS, config.GRID_SIZE, config.SEED)
    h_obs_3 = su.get_observations(H_true, wells_3)
    K_post3, z_post3 = su.run_esmda(vae, fno_kh, z_prior.copy(), wells_3, h_obs_3, config)
    C_post3  = su.predict_C_batch(K_post3, fno_kc, config)
    metrics3 = su.evaluate_prediction(C_post3, C_true, K_post3, K_true, config)

    # ── 20 口 LHS 井（全部随机布置，非自适应） ──
    wells_20 = su.init_wells_lhs(config.TARGET_WELLS, config.GRID_SIZE, config.SEED)
    h_obs_20 = su.get_observations(H_true, wells_20)
    K_post20, z_post20 = su.run_esmda(vae, fno_kh, z_prior.copy(), wells_20, h_obs_20, config)
    C_post20  = su.predict_C_batch(K_post20, fno_kc, config)
    metrics20 = su.evaluate_prediction(C_post20, C_true, K_post20, K_true, config)

    return {
        # ── 先验 ──
        'prior_C_mean_std': float(prior_mean_std),
        'prior_C_var':      float(prior_C_var),
        'prior_K_mean':     np.mean(K_prior, axis=0).astype(np.float32),
        'prior_K_std':      np.std(K_prior,  axis=0).astype(np.float32),
        'prior_C_mean':     np.mean(C_prior, axis=0).astype(np.float32),
        'prior_C_std':      np.std(C_prior,  axis=0).astype(np.float32),

        # ── 后验-3井 ──
        'post3_K_r2':      float(metrics3['K_r2']),
        'post3_K_rmse':    float(metrics3['K_rmse']),
        'post3_C_r2':      float(metrics3['C_r2']),
        'post3_C_rmse':    float(metrics3['C_rmse']),
        'post3_C_var':     float(metrics3['C_var']),
        'post3_C_mean_std': float(metrics3['C_mean_std']),
        'post3_K_mean':    np.mean(K_post3, axis=0).astype(np.float32),
        'post3_K_std':     np.std(K_post3,  axis=0).astype(np.float32),
        'post3_C_mean':    np.mean(C_post3, axis=0).astype(np.float32),
        'post3_C_std':     np.std(C_post3,  axis=0).astype(np.float32),
        'wells_3':         np.array(wells_3, dtype=np.int32),

        # ── 后验-20井 ──
        'post20_K_r2':      float(metrics20['K_r2']),
        'post20_K_rmse':    float(metrics20['K_rmse']),
        'post20_C_r2':      float(metrics20['C_r2']),
        'post20_C_rmse':    float(metrics20['C_rmse']),
        'post20_C_var':     float(metrics20['C_var']),
        'post20_C_mean_std': float(metrics20['C_mean_std']),
        'post20_K_mean':    np.mean(K_post20, axis=0).astype(np.float32),
        'post20_K_std':     np.std(K_post20,  axis=0).astype(np.float32),
        'post20_C_mean':    np.mean(C_post20, axis=0).astype(np.float32),
        'post20_C_std':     np.std(C_post20,  axis=0).astype(np.float32),
        'wells_20':         np.array(wells_20, dtype=np.int32),

        # ── 真实值（供 evaluate_es_mda.py 可视化用） ──
        'K_true': K_true.astype(np.float32),
        'C_true': C_true.astype(np.float32),
    }


# ==========================================
# 保存 / 进度工具
# ==========================================
def save_sample_result(output_dir, sample_idx, result):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'sample_{sample_idx:03d}.npz')
    np.savez(path, **result)


def check_sample_done(output_dir, sample_idx):
    return os.path.exists(
        os.path.join(output_dir, f'sample_{sample_idx:03d}.npz'))


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
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Run ES-MDA on test samples (prior vs. 3-well vs. 20-well posteriors)')
    parser.add_argument('--start',     type=int,  default=0,
                        help='起始样本编号（含）')
    parser.add_argument('--end',       type=int,  default=100,
                        help='终止样本编号（不含）')
    parser.add_argument('--no-resume', action='store_false', dest='resume',
                        default=True,
                        help='禁用断点续跑，强制重新计算所有样本')
    args = parser.parse_args()

    config = EsMdaConfig()
    os.makedirs(config.OUTPUT_ROOT, exist_ok=True)

    print("=" * 70)
    print(f"  ES-MDA Run  |  samples={args.start}~{args.end-1}"
          f"  resume={args.resume}")
    print(f"  初始井数={config.INITIAL_WELLS}（后验-3井）"
          f"  目标井数={config.TARGET_WELLS}（后验-20井）")
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

    for sample_idx in range(args.start, args.end):
        print(f"\n{'='*70}")
        print(f"  Sample {sample_idx}/{args.end-1}  "
              f"[{(time.time()-total_start)/60:.1f} min elapsed]")
        print(f"{'='*70}")

        if args.resume and check_sample_done(config.OUTPUT_ROOT, sample_idx):
            print(f"  [Skip] 已存在，跳过。")
            continue

        K_true, H_true, C_true = su.get_test_sample(K_all, H_all, C_all, sample_idx)
        su.set_seed(config.SEED)

        t0     = time.time()
        result = run_single_sample(
            sample_idx, K_true, H_true, C_true, k_train_data,
            vae, fno_kh, fno_kc, config)
        elapsed = time.time() - t0

        save_sample_result(config.OUTPUT_ROOT, sample_idx, result)

        sk = f'sample_{sample_idx:03d}'
        progress['completed_samples'][sk] = {
            'time_min':        round(elapsed / 60, 2),
            'prior_C_mean_std': round(result['prior_C_mean_std'], 6),
            'post3_K_r2':      round(result['post3_K_r2'],  4),
            'post3_C_r2':      round(result['post3_C_r2'],  4),
            'post3_C_mean_std': round(result['post3_C_mean_std'],  6),
            'post20_K_r2':     round(result['post20_K_r2'], 4),
            'post20_C_r2':     round(result['post20_C_r2'], 4),
            'post20_C_mean_std': round(result['post20_C_mean_std'], 6),
        }
        save_progress(config.OUTPUT_ROOT, progress)

        print(f"  [Done] {elapsed/60:.1f} min")
        print(f"    先验   C_mean_std={result['prior_C_mean_std']:.4f}")
        print(f"    后验-3 K_R2={result['post3_K_r2']:.4f}  "
              f"C_R2={result['post3_C_r2']:.4f}  "
              f"C_mean_std={result['post3_C_mean_std']:.4f}")
        print(f"    后验-20 K_R2={result['post20_K_r2']:.4f}  "
              f"C_R2={result['post20_C_r2']:.4f}  "
              f"C_mean_std={result['post20_C_mean_std']:.4f}")

    # ── 汇总 ──
    total_h = (time.time() - total_start) / 3600
    print(f"\n{'='*70}")
    print(f"  Done!  Total: {total_h:.2f} h  →  {config.OUTPUT_ROOT}/")
    print(f"{'='*70}")

    prior_stds, post3_r2s, post3_stds, post20_r2s, post20_stds = [], [], [], [], []
    for si in range(args.start, args.end):
        info = progress['completed_samples'].get(f'sample_{si:03d}')
        if info:
            prior_stds.append(info['prior_C_mean_std'])
            post3_r2s.append(info['post3_C_r2'])
            post3_stds.append(info['post3_C_mean_std'])
            post20_r2s.append(info['post20_C_r2'])
            post20_stds.append(info['post20_C_mean_std'])

    n = len(prior_stds)
    if n > 0:
        print(f"\n{'条件':>12}  {'C_mean_std':>22}  {'C_R2':>18}  n")
        print("-" * 62)
        print(f"  {'先验':>10}:  "
              f"mean_std={np.mean(prior_stds):.4f}±{np.std(prior_stds):.4f}  "
              f"{'—':>18}  {n}")
        print(f"  {'后验-3井':>8}:  "
              f"mean_std={np.mean(post3_stds):.4f}±{np.std(post3_stds):.4f}  "
              f"C_R2={np.mean(post3_r2s):.4f}±{np.std(post3_r2s):.4f}  {n}")
        print(f"  {'后验-20井':>7}:  "
              f"mean_std={np.mean(post20_stds):.4f}±{np.std(post20_stds):.4f}  "
              f"C_R2={np.mean(post20_r2s):.4f}±{np.std(post20_r2s):.4f}  {n}")

    save_progress(config.OUTPUT_ROOT, progress)


if __name__ == "__main__":
    main()