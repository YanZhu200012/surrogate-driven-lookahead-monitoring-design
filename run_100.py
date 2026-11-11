"""
run_100.py
100 个测试场上四种布井策略（Look-ahead / Greedy / LHS / Random）的对比实验。

- 四种策略共用同一先验、同一组 3 口 LHS 初始井、同一间距约束
  （Chebyshev 距离 > MIN_WELL_SPACING）和同一观测误差协方差
  R = (σ_obs² + σ_f1²) I，σ_f1 在 FNO-1 验证集上校准；
- 初始 3 井从先验联合同化一次，之后每轮只同化新增井的 5 个水头值；
- LHS / Random 各 10 个布局，不满足间距约束时整体重抽（初始井保留）。

输出：./test_100_results/
  sample_XXX/{active,greedy,lhs,random}.npz   （active.npz 对应 Look-ahead）
  R_calibration.json / run_config.json / progress.json（断点续跑）
初始三井后的浓度方差保存在各结果文件的 post3_C_variance_map 中；
PLOT_SAMPLE_IDS 中的样本额外保存完整后验集合，供 figure_plot.py 绘图。
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import shared_utils as su


class TestConfig(su.BaseConfig):
    OUTPUT_ROOT = './test_100_results'
    # 仅代表性样本保留全部最终后验集合，供 figure_plot.py 绘制空间场。
    PLOT_SAMPLE_IDS = (14,)


METHOD_FILES = {
    'active': 'active.npz',
    'greedy': 'greedy.npz',
    'lhs':    'lhs.npz',
    'random': 'random.npz',
}


# ==========================================
# 结果保存与随机数
# ==========================================
def save_npz(path, **data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, **data)
    os.replace(tmp, path)


def file_hash(path):
    sha = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(part)
    return sha.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def stage_seed(config, sample_idx, n_wells, virtual=False):
    """正式同化种子即 shared_utils.stage_seed；虚拟同化（候选评价）使用独立偏移。"""
    if virtual:
        return config.SEED + 8000 + sample_idx * 100 + n_wells
    return su.stage_seed(config, sample_idx, n_wells)


def initial_wells(config):
    """四种策略共享的 3 口 LHS 初始井（shared_utils.initial_wells）。"""
    return su.initial_wells(config)


def get_prior(vae, K_train, config):
    """每个测试场、每种方法和 trial 均使用同一份潜空间先验（种子 config.SEED）。"""
    return su.generate_prior(vae, K_train, config.N_ENSEMBLE, config)


# ==========================================
# LHS / Random：完整布局拒绝采样
# ==========================================
def lhs_layout(seed, config):
    init = initial_wells(config)
    future = [w for w in su.init_wells_lhs(config.TARGET_WELLS, config.GRID_SIZE, seed)
              if w not in set(init)]
    needed = config.TARGET_WELLS - len(init)
    if len(future) < needed:
        seen = set(init) | set(future)
        for extra in (config.TARGET_WELLS * 2, config.TARGET_WELLS * 4,
                      config.GRID_SIZE ** 2):
            for w in su.init_wells_lhs(extra, config.GRID_SIZE, seed):
                if w not in seen:
                    future.append(w)
                    seen.add(w)
                if len(future) >= needed:
                    break
            if len(future) >= needed:
                break
    return init + future[:needed]


def generate_layout(method, sample_idx, trial, config):
    """整体拒绝采样；所有无效提案耗时计入布局生成时间。"""
    init = initial_wells(config)
    seed_offset = 1000 if method == 'random' else 3000
    seed = config.SEED + seed_offset + sample_idx * 100 + trial
    rng = np.random.RandomState(seed)
    if method == 'random':
        cells = [(r, c) for r in range(config.GRID_SIZE)
                 for c in range(config.GRID_SIZE) if (r, c) not in set(init)]

    t0 = time.perf_counter()
    for attempt in range(10000):
        if method == 'random':
            indices = rng.choice(len(cells), config.TARGET_WELLS - len(init),
                                 replace=False)
            wells = init + [cells[i] for i in indices]
        else:
            proposal_seed = (seed if attempt == 0 else
                             int(np.random.SeedSequence([seed, attempt])
                                 .generate_state(1)[0]))
            wells = lhs_layout(proposal_seed, config)
        if len(wells) == config.TARGET_WELLS and su.spacing_valid(wells, config):
            return wells, attempt, time.perf_counter() - t0
    raise RuntimeError(f'{method}: 10000 次尝试后仍未找到满足间距约束的布局')


# ==========================================
# 序贯同化与评价
# ==========================================
def get_metrics(z, truth, models, config):
    vae, _, fno_kc = models
    K_true, _, C_true = truth
    K = su.vae_decode(z, vae, config)
    C = su.predict_C_batch(K, fno_kc, config)
    metrics = su.evaluate_prediction(C, C_true, K, K_true, config)
    return K, C, metrics


def prepare_initial(truth, z_prior, models, config, sample_idx):
    """共享初始三井后验；四方法无需重复计算相同的 3 井阶段。"""
    vae, fno_kh, _ = models
    wells = initial_wells(config)
    start = time.perf_counter()
    t0 = time.perf_counter()
    z3 = su.prepare_initial_latent(truth[1], z_prior, vae, fno_kh, config, sample_idx)
    assimilation_seconds = time.perf_counter() - t0
    _, C3, metrics = get_metrics(z3, truth, models, config)
    metrics.update({
        'n_wells': 3, 'wells': [list(w) for w in wells],
        'assimilated_wells': [list(w) for w in wells],
        'n_assimilated_head_values': 15,
        'selection_seconds': 0.0,
        'assimilation_seconds': assimilation_seconds,
        'stage_total_seconds': time.perf_counter() - start,
    })
    return z3, C3, metrics


def run_trajectory(method, sample_idx, truth, models, config,
                   z3, initial_metrics, fixed_wells=None, checkpoint=None,
                   save_full=False):
    """从共同 3 井后验出发；每轮只同化新增井的 5 个水头值。"""
    vae, fno_kh, fno_kc = models
    wells = initial_wells(config)
    z = z3.copy()
    history = [dict(initial_metrics)]
    if checkpoint is not None and checkpoint.exists():
        with np.load(checkpoint, allow_pickle=True) as d:
            z = d['z'].copy()
            wells = [tuple(w) for w in d['wells']]
            history = d['history'].tolist()
        print(f'    [Resume] {method}: {len(wells)} wells')

    K_final, C_final = None, None
    for n in range(len(wells) + 1, config.TARGET_WELLS + 1):
        stage_start = time.perf_counter()
        select_seconds = 0.0
        expected_reduction = np.nan
        n_candidates = 0
        if fixed_wells is not None:
            new_well = tuple(fixed_wells[n-1])
        else:
            t_select = time.perf_counter()
            K_now = su.vae_decode(z, vae, config)
            C_now = su.predict_C_batch(K_now, fno_kc, config)
            if method == 'greedy':
                var_map = su.compute_weighted_C_variance(C_now, config)[1]
                new_well = su.feasible_wells(var_map, wells, config, limit=1)[0]
            else:
                new_well, expected_reduction, n_candidates = su.select_look_ahead(
                    z, wells, vae, fno_kh, fno_kc, config,
                    stage_seed(config, sample_idx, n, virtual=True),
                    current_K=K_now, current_C=C_now)
            select_seconds = time.perf_counter() - t_select

        wells.append(new_well)
        if not su.spacing_valid(wells, config):
            raise ValueError(f'Well spacing violated: {new_well}')
        obs = su.get_observations(truth[1], [new_well])
        t_assim = time.perf_counter()
        z = su.assimilate_stage(z, [new_well], obs, vae, fno_kh, config,
                                 stage_seed(config, sample_idx, n))
        assimilation_seconds = time.perf_counter() - t_assim
        K_final, C_final, metrics = get_metrics(z, truth, models, config)
        metrics.update({
            'n_wells': n,
            'wells': [list(w) for w in wells],
            'assimilated_wells': [list(new_well)],
            'n_assimilated_head_values': 5,
            'selection_seconds': select_seconds,
            'assimilation_seconds': assimilation_seconds,
            'stage_total_seconds': time.perf_counter() - stage_start,
            'expected_variance_reduction': expected_reduction,
            'n_candidates_evaluated': n_candidates,
        })
        history.append(metrics)
        print(f'    [{method}] {n:2d}/{config.TARGET_WELLS}  C_R2={metrics["C_r2"]:.4f}  '
              f'C_std={metrics["C_mean_std"]:.5f}', flush=True)

        if checkpoint is not None:
            save_npz(checkpoint, z=z, wells=np.array(wells),
                     history=np.array(history, dtype=object))

    if K_final is None or C_final is None:
        K_final = su.vae_decode(z, vae, config)
        C_final = su.predict_C_batch(K_final, fno_kc, config)
    result = {
        'history': history,
        'final_wells': np.asarray(wells, dtype=np.int16),
        'final_K_mean': K_final.mean(axis=0).astype(np.float32),
        'final_C_mean': C_final.mean(axis=0).astype(np.float32),
    }
    if save_full:
        result['K_samples_final'] = K_final.astype(np.float32)
        result['C_samples_final'] = C_final.astype(np.float32)
    return result


def run_method(method, sample_idx, truth, models, config, z3, C3,
               initial_metrics, prior_metrics, sample_dir):
    common = {
        'K_true': truth[0].astype(np.float32),
        'C_true': truth[2].astype(np.float32),
        'initial_C_var': prior_metrics[0],
        'initial_mean_std': prior_metrics[1],
        'post3_C_std': C3.std(axis=0).astype(np.float32),
        'post3_C_variance_map': su.compute_weighted_C_variance(C3, config)[1].astype(np.float32),
        'R_diagonal': config.OBS_NOISE_STD**2 + config.SIGMA_F1**2,
        'sigma_f1_validation': config.SIGMA_F1,
    }
    save_full = sample_idx in config.PLOT_SAMPLE_IDS

    if method in ('active', 'greedy'):
        state_path = sample_dir / f'.{method}.state.npz'
        out = run_trajectory(method, sample_idx, truth, models, config,
                             z3, initial_metrics, checkpoint=state_path,
                             save_full=save_full)
        history = out['history']
        elapsed = sum(h['stage_total_seconds'] for h in history)
        out.update(common)
        out.update({
            'history': np.asarray(history, dtype=object),
            'elapsed_seconds': elapsed,
            'time_min': elapsed / 60,
        })
        path = sample_dir / METHOD_FILES[method]
        save_npz(path, **out)
        state_path.unlink(missing_ok=True)
        return elapsed / 60

    histories, K_means, C_means = [], [], []
    layout_times, trial_times, rejected = [], [], []
    full_K, full_C = [], []
    representative = int(np.random.default_rng(sample_idx).integers(config.NUM_RANDOM_TRIALS))
    representative_data = None

    for trial in range(config.NUM_RANDOM_TRIALS):
        trial_path = sample_dir / f'.{method}-trial{trial:02d}.npz'
        if trial_path.exists():
            with np.load(trial_path, allow_pickle=True) as d:
                trial_data = {k: d[k] for k in d.files}
            print(f'    [{method}] trial {trial+1}/10 (cached)')
        else:
            wells, count, layout_time = generate_layout(method, sample_idx, trial, config)
            result = run_trajectory(method, sample_idx, truth, models, config,
                                    z3, initial_metrics, fixed_wells=wells,
                                    save_full=save_full and
                                    (method == 'random' or trial == representative))
            stage_time = sum(h['stage_total_seconds'] for h in result['history'])
            trial_data = {
                'history': np.asarray(result['history'], dtype=object),
                'final_K_mean': result['final_K_mean'],
                'final_C_mean': result['final_C_mean'],
                'layout_seconds': layout_time,
                'trial_seconds': layout_time + stage_time,
                'rejected_count': count,
            }
            if 'K_samples_final' in result:
                trial_data['K_samples_final'] = result['K_samples_final']
                trial_data['C_samples_final'] = result['C_samples_final']
            save_npz(trial_path, **trial_data)
        histories.append(trial_data['history'].tolist())
        K_means.append(trial_data['final_K_mean'])
        C_means.append(trial_data['final_C_mean'])
        layout_times.append(float(trial_data['layout_seconds']))
        trial_times.append(float(trial_data['trial_seconds']))
        rejected.append(int(trial_data['rejected_count']))
        if save_full:
            if method == 'random':
                full_K.append(trial_data['K_samples_final'])
                full_C.append(trial_data['C_samples_final'])
            elif trial == representative:
                representative_data = trial_data

    total_seconds = float(np.sum(trial_times))
    result = dict(common)
    result.update({
        'all_trial_histories': np.asarray(histories, dtype=object),
        'all_trial_K_means': np.asarray(K_means, dtype=np.float32),
        'all_trial_C_means': np.asarray(C_means, dtype=np.float32),
        'layout_generation_seconds_per_trial': np.asarray(layout_times),
        'trial_algorithm_seconds': np.asarray(trial_times),
        'rejected_layout_counts': np.asarray(rejected),
        'elapsed_seconds': total_seconds,
        'mean_trial_seconds': total_seconds / config.NUM_RANDOM_TRIALS,
        'time_min': total_seconds / 60 / config.NUM_RANDOM_TRIALS,
    })
    if method == 'lhs' and representative_data is not None:
        result['history'] = representative_data['history']
        result['final_wells'] = np.asarray(histories[representative][-1]['wells'])
        result['K_samples_final'] = representative_data['K_samples_final']
        result['C_samples_final'] = representative_data['C_samples_final']
        result['_aligned_trial'] = representative
    if method == 'random' and save_full:
        result['all_final_K_samples'] = np.asarray(full_K, dtype=np.float32)
        result['all_final_C_samples'] = np.asarray(full_C, dtype=np.float32)

    save_npz(sample_dir / METHOD_FILES[method], **result)
    for trial in range(config.NUM_RANDOM_TRIALS):
        (sample_dir / f'.{method}-trial{trial:02d}.npz').unlink(missing_ok=True)
    return total_seconds / 60 / config.NUM_RANDOM_TRIALS


# ==========================================
# 主函数
# ==========================================
def main(start=0, end=100, methods=('active', 'greedy', 'lhs', 'random'),
         output='./test_100_results', batch_size=24, force=False):
    config = TestConfig()
    config.OUTPUT_ROOT = output
    config.BATCH_SIZE = batch_size
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    vae, fno_kh, fno_kc = su.load_all_models(config)
    calibration_path = output_dir / 'R_calibration.json'
    weight_hash = file_hash(config.FNO_KH_PATH)
    vae_hash = file_hash(config.VAE_PATH)
    fno2_hash = file_hash(config.FNO_KC_PATH)
    validation_hash = file_hash(config.VALIDATION_INDICES_PATH)
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
        if (calibration['fno_weights_sha256'] != weight_hash or
                calibration['validation_indices_sha256'] != validation_hash):
            raise ValueError('验证集校准配置发生变化，请改用新的结果目录')
        sigma = calibration['sigma_f1']
    else:
        sigma, sigma_time, n_calibration = su.calibrate_sigma_f1(fno_kh, config)
        calibration = {
            'validation_indices_path': config.VALIDATION_INDICES_PATH,
            'validation_indices_sha256': validation_hash,
            'fno_weights_sha256': weight_hash,
            'n_calibration': n_calibration,
            'sigma_f1': sigma,
            'sigma_f1_per_time': sigma_time,
            'obs_noise_std': config.OBS_NOISE_STD,
            'R_diagonal': config.OBS_NOISE_STD**2 + sigma**2,
        }
        save_json(calibration_path, calibration)
    config.SIGMA_F1 = float(sigma)
    print(f'[R] sigma_f1(val) = {sigma:.5f}; R = {config.OBS_NOISE_STD**2 + sigma**2:.6f} I')

    settings_path = output_dir / 'run_config.json'
    settings = {
        'seed': config.SEED, 'n_ensemble': config.N_ENSEMBLE,
        'alpha': config.ALPHA_LIST, 'virtual_alpha': config.VIRTUAL_ALPHA_LIST,
        'sigma_f1': config.SIGMA_F1, 'R_diagonal': calibration['R_diagonal'],
        'n_candidates': config.MAX_CANDIDATES,
        'fno_weights_sha256': weight_hash,
        'vae_weights_sha256': vae_hash,
        'fno2_weights_sha256': fno2_hash,
        'validation_indices_sha256': validation_hash,
        'min_spacing': config.MIN_WELL_SPACING, 'n_trials': config.NUM_RANDOM_TRIALS,
    }
    if settings_path.exists():
        old = json.loads(settings_path.read_text(encoding='utf-8'))
        if old != settings:
            raise ValueError('当前目录包含不同的实验配置，请更换输出目录')
    else:
        save_json(settings_path, settings)

    K_test, H_test, C_test, K_train = su.load_test_data(config)
    progress_path = output_dir / 'progress.json'
    progress = json.loads(progress_path.read_text(encoding='utf-8')) \
        if progress_path.exists() else {'completed_samples': {}}

    for sample_idx in range(start, end):
        sample_dir = output_dir / f'sample_{sample_idx:03d}'
        sample_dir.mkdir(parents=True, exist_ok=True)
        pending = [m for m in methods if force or
                   not (sample_dir / METHOD_FILES[m]).exists()]
        if not pending:
            print(f'[Sample {sample_idx:03d}] complete, skip')
            continue
        if force:
            for m in pending:
                (sample_dir / f'.{m}.state.npz').unlink(missing_ok=True)
                for trial in range(config.NUM_RANDOM_TRIALS):
                    (sample_dir / f'.{m}-trial{trial:02d}.npz').unlink(missing_ok=True)

        print(f'\n[Sample {sample_idx:03d}] {pending}')
        truth = (K_test[sample_idx], H_test[sample_idx], C_test[sample_idx])
        models = (vae, fno_kh, fno_kc)
        z_prior = get_prior(vae, K_train, config)
        K_prior = su.vae_decode(z_prior, vae, config)
        C_prior = su.predict_C_batch(K_prior, fno_kc, config)
        prior_var = su.compute_weighted_C_variance(C_prior, config)
        z3, C3, metrics3 = prepare_initial(truth, z_prior, models,
                                               config, sample_idx)
        for method in pending:
            avg_time_min = run_method(method, sample_idx, truth, models, config,
                                      z3, C3, metrics3,
                                      (prior_var[0], prior_var[4]), sample_dir)
            progress['completed_samples'].setdefault(f'sample_{sample_idx:03d}', {})[
                METHOD_FILES[method][:-4]] = {'time_min': round(avg_time_min, 4)}
            save_json(progress_path, progress)
            print(f'  [Saved] {sample_dir / METHOD_FILES[method]}')

    print(f'\n[Done] {output_dir.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--methods', default='active,greedy,lhs,random')
    parser.add_argument('--output', default='./test_100_results')
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    methods = tuple(s.strip() for s in args.methods.split(','))
    if any(m not in METHOD_FILES for m in methods):
        parser.error(f'Methods must be within {tuple(METHOD_FILES)}')
    if not (0 <= args.start < args.end <= 100):
        parser.error('Samples must satisfy 0 <= start < end <= 100')
    main(args.start, args.end, methods, args.output, args.batch_size, args.force)
