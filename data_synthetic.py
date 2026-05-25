"""
data_synthetic.py
批量合成实验数据准备

修改说明（设定 A）：
  H_test 和 C_test 直接从物理模拟器真值文件加载，
  不再用 FNO 代理生成，避免 inverse crime。
  FNO f1 仅作为同化算子，与观测数据来源分离。
  sigma_f1 写入 npz，供 shared_utils.run_esmda / virtual_es_mda_update 使用。
"""

import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
import os

from train_f1 import FNO2d as FNO2d_KH
from train_f2 import FNO2d_KC

# ==========================================
# 配置
# ==========================================
PATH_K_TEST  = './data/test/DataK_test_100_51_51.mat'
PATH_H_TEST  = './data/test/DataH_test_100_5_51_51.mat'       # 物理模拟器 H 真值
PATH_C_TEST  = './data/test/DataC_100_5_51_51_Original.mat'   # 物理模拟器 C 真值（原始量纲）

# 训练数据路径（用于加载归一化参数，计算 sigma_f1）
PATH_H_TRAIN = './data/DataH_6000_5_51_51.mat'
PATH_K_TRAIN = './data/DataK_6000_51_51.mat'
KC_NORM_PATH = './fno_kc_results/normalization_params.npz'

# 模型路径
FNO_KH_PATH  = './fno_kh_results/best_fno_kh.pth'
FNO_KC_PATH  = './fno_kc_results/best_fno_kc.pth'

# 输出
OUTPUT_DIR   = './test_synthetic_data'
OUTPUT_PATH  = os.path.join(OUTPUT_DIR, 'synthetic_test_100.npz')

GRID_SIZE    = 51
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TEST       = 100
BATCH_FNO    = 20   # 用于估计 sigma_f1 时的推理 batch size


# ==========================================
# 工具函数
# ==========================================
def load_mat_data(path, expected_keyword=None):
    """通用 mat 文件加载"""
    mat_data = sio.loadmat(path)
    keys = [k for k in mat_data.keys() if not k.startswith('__')]
    if expected_keyword:
        matched = [k for k in keys if expected_keyword.lower() in k.lower()]
        key = matched[0] if matched else keys[0]
    else:
        key = keys[0]
    data = mat_data[key].astype(np.float32)
    print(f"  加载 {path}: key='{key}', shape={data.shape}, "
          f"range=[{data.min():.4f}, {data.max():.4f}]")
    return data


def load_fno_kh(h_mean, h_std):
    fno = FNO2d_KH(modes1=12, modes2=12, width=64, in_channels=1, out_channels=5,
                   h_mean=h_mean, h_std=h_std).to(DEVICE)
    fno.load_state_dict(torch.load(FNO_KH_PATH, map_location=DEVICE))
    fno.eval()
    return fno


def batch_predict_H(model, K_all, batch_size):
    """分批次用 FNO f1 预测 H（用于估计 sigma_f1）"""
    N = len(K_all)
    results = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        K_batch = torch.tensor(
            K_all[start:end, np.newaxis, :, :], dtype=torch.float32
        ).to(DEVICE)
        with torch.no_grad():
            out = model(K_batch, return_normalized=False)
        results.append(out.cpu().numpy())
        print(f"    sigma_f1 推理进度: {end}/{N}")
    return np.concatenate(results, axis=0)


# ==========================================
# 核心：估计 sigma_f1（f1 代理误差标准差）
# ==========================================
def estimate_sigma_f1(fno_kh, K_test, H_test_physics):
    """
    在全场空间估计 f1 代理误差。

    做法：对测试集所有样本，比较 f1(K_true) 与物理模拟器 H_true 的逐点残差，
    取全局 RMSE 作为 sigma_f1（标量）。

    sigma_f1_per_time 保存各时刻误差，便于分析是否满足各时刻同分布假设。
    若各时刻差异较大，可考虑在 run_esmda 中使用块对角 R。

    这个值将写入 npz，供 shared_utils 中的 run_esmda 和
    virtual_es_mda_update 构造 R 矩阵使用。
    """
    print("  用 FNO f1 推理测试集，估计代理误差 sigma_f1...")
    H_pred = batch_predict_H(fno_kh, K_test, BATCH_FNO)   # (100, 5, 51, 51)

    residuals = H_pred - H_test_physics                    # (100, 5, 51, 51)
    sigma_f1  = float(np.sqrt(np.mean(residuals ** 2)))    # 全局 RMSE

    sigma_f1_per_time = []
    for t in range(5):
        rmse_t = float(np.sqrt(np.mean(residuals[:, t, :, :] ** 2)))
        sigma_f1_per_time.append(rmse_t)
        print(f"    t={t+1}: RMSE = {rmse_t:.5f}")

    print(f"  sigma_f1 (全局 RMSE) = {sigma_f1:.5f}")
    print(f"  各时刻 RMSE 范围: [{min(sigma_f1_per_time):.5f}, "
          f"{max(sigma_f1_per_time):.5f}]  "
          f"（差异{'较大，可考虑块对角 R' if max(sigma_f1_per_time)/max(min(sigma_f1_per_time),1e-9) > 2 else '较小，全局 RMSE 近似合理'}）")
    return sigma_f1, sigma_f1_per_time


# ==========================================
# 主函数
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  批量测试数据准备（设定 A：物理模拟器真值）")
    print("=" * 60)

    # -------- Step 1: 加载 K 真值 --------
    print("\n[Step 1] 加载测试 K 场（物理模拟器）...")
    K_test = load_mat_data(PATH_K_TEST)
    assert K_test.shape[0] >= N_TEST
    K_test = K_test[:N_TEST]
    print(f"  使用 {N_TEST} 个测试 K 场，shape: {K_test.shape}")

    # -------- Step 2: 加载 H 真值（物理模拟器，非 FNO 生成） --------
    print("\n[Step 2] 加载测试 H 场（物理模拟器真值，非代理）...")
    H_test = load_mat_data(PATH_H_TEST)
    assert H_test.shape[0] >= N_TEST
    H_test = H_test[:N_TEST]   # (100, 5, 51, 51)
    print(f"  H_test shape: {H_test.shape}")

    # -------- Step 3: 加载 C 真值（物理模拟器，原始量纲） --------
    print("\n[Step 3] 加载测试 C 场（物理模拟器真值，Original 量纲）...")
    C_test = load_mat_data(PATH_C_TEST)
    assert C_test.shape[0] >= N_TEST
    C_test = C_test[:N_TEST]   # (100, 5, 51, 51)
    print(f"  C_test shape: {C_test.shape}")

    # -------- Step 4: 加载归一化参数 --------
    print("\n[Step 4] 加载归一化参数...")
    h_mat  = sio.loadmat(PATH_H_TRAIN)
    h_key  = [k for k in h_mat.keys() if not k.startswith('__')][0]
    h_data = h_mat[h_key].astype(np.float32)
    h_mean = float(np.mean(h_data))
    h_std  = float(np.std(h_data))
    print(f"  H 归一化: mean={h_mean:.4f}, std={h_std:.4f}")

    kc_norm = np.load(KC_NORM_PATH)
    c_mean  = float(kc_norm['c_mean'])
    c_std   = float(kc_norm['c_std'])
    print(f"  C 归一化: mean={c_mean:.4f}, std={c_std:.4f}")

    # -------- Step 5: 估计 sigma_f1 --------
    print("\n[Step 5] 加载 FNO f1，估计代理误差 sigma_f1...")
    fno_kh = load_fno_kh(h_mean, h_std)
    sigma_f1, sigma_f1_per_time = estimate_sigma_f1(fno_kh, K_test, H_test)
    del fno_kh
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------- Step 6: 保存 --------
    print("\n[Step 6] 保存...")
    np.savez(OUTPUT_PATH,
             K_test            = K_test,             # (100, 51, 51)   物理模拟器真值
             H_test            = H_test,             # (100, 5, 51, 51) 物理模拟器真值
             C_test            = C_test,             # (100, 5, 51, 51) 物理模拟器真值
             h_mean            = h_mean,
             h_std             = h_std,
             c_mean            = c_mean,
             c_std             = c_std,
             sigma_f1          = sigma_f1,           # f1 全局代理误差（标量）
             sigma_f1_per_time = sigma_f1_per_time,  # 各时刻代理误差（供分析）
             grid_size         = GRID_SIZE,
             n_test            = N_TEST)

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"  保存至: {OUTPUT_PATH}  ({size_mb:.1f} MB)")
    print(f"  sigma_f1 = {sigma_f1:.5f}  （已写入 npz，供 R 矩阵使用）")

    # -------- Step 7: 抽检可视化 --------
    print("\n[Step 7] 抽检可视化...")
    check_indices = [0, 25, 50, 75, 99]
    fig, axes = plt.subplots(len(check_indices), 4,
                             figsize=(16, 4 * len(check_indices)))

    for row, idx in enumerate(check_indices):
        im = axes[row, 0].imshow(K_test[idx], origin='lower', cmap='jet',
                                 vmin=0, vmax=1)
        axes[row, 0].set_title(f'K_true [{idx}]')
        plt.colorbar(im, ax=axes[row, 0])

        im = axes[row, 1].imshow(H_test[idx, 0], origin='lower', cmap='coolwarm')
        axes[row, 1].set_title(f'H_true [{idx}] t=1 (physics)')
        plt.colorbar(im, ax=axes[row, 1])

        im = axes[row, 2].imshow(H_test[idx, 4], origin='lower', cmap='coolwarm')
        axes[row, 2].set_title(f'H_true [{idx}] t=5 (physics)')
        plt.colorbar(im, ax=axes[row, 2])

        im = axes[row, 3].imshow(C_test[idx, 4], origin='lower', cmap='hot')
        axes[row, 3].set_title(f'C_true [{idx}] t=5 (physics)')
        plt.colorbar(im, ax=axes[row, 3])

    plt.suptitle(f'Test Data Spot Check — Physics Simulator True Values\n'
                 f'sigma_f1={sigma_f1:.5f}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'test_data_check.png'), dpi=150)
    plt.show()
    print("\nDone!")


if __name__ == "__main__":
    main()