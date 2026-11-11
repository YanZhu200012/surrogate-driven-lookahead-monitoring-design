"""
data_synthetic.py
打包 100 个测试样本 → ./test_synthetic_data/synthetic_test_100.npz

K、H、C 均直接取自物理模拟器（COMSOL）计算的真值文件；
H 与 C 不由 FNO 代理生成，以避免 inverse crime。
FNO-1 只在同化中作为正演算子使用，其代理误差 σ_f1 由同化脚本
在 FNO-1 验证集（./fno_kh_results/val_indices.npy）上校准，
测试集不参与 σ_f1 的估计，本脚本也不计算 σ_f1。
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# ==========================================
# 配置
# ==========================================
PATH_K_TEST  = './data/test/DataK_test_100_51_51.mat'
PATH_H_TEST  = './data/test/DataH_test_100_5_51_51.mat'       # 物理模拟器 H 真值（5 次抽水试验）
PATH_C_TEST  = './data/test/DataC_100_5_51_51_Original.mat'   # 物理模拟器 C 真值（5 个时步，原始量纲）

# 代理模型训练时保存的归一化参数（随测试数据一并存档，便于查阅）
KH_NORM_PATH = './fno_kh_results/normalization_params.npz'
KC_NORM_PATH = './fno_kc_results/normalization_params.npz'

# 输出
OUTPUT_DIR   = './test_synthetic_data'
OUTPUT_PATH  = os.path.join(OUTPUT_DIR, 'synthetic_test_100.npz')

GRID_SIZE    = 51
N_TEST       = 100


# ==========================================
# 工具函数
# ==========================================
def load_mat_data(path):
    """读取 .mat 文件中的第一个数据变量"""
    mat_data = sio.loadmat(path)
    key  = [k for k in mat_data.keys() if not k.startswith('__')][0]
    data = mat_data[key].astype(np.float32)
    print(f"  加载 {path}: key='{key}', shape={data.shape}, "
          f"range=[{data.min():.4f}, {data.max():.4f}]")
    return data


# ==========================================
# 主函数
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  测试数据打包（物理模拟器真值）")
    print("=" * 60)

    # -------- Step 1: K 真值 --------
    print("\n[Step 1] 加载测试 K 场...")
    K_test = load_mat_data(PATH_K_TEST)
    assert K_test.shape[0] >= N_TEST
    K_test = K_test[:N_TEST]                  # (100, 51, 51)

    # -------- Step 2: H 真值 --------
    print("\n[Step 2] 加载测试 H 场（物理模拟器）...")
    H_test = load_mat_data(PATH_H_TEST)
    assert H_test.shape[0] >= N_TEST
    H_test = H_test[:N_TEST]                  # (100, 5, 51, 51)

    # -------- Step 3: C 真值 --------
    print("\n[Step 3] 加载测试 C 场（物理模拟器，原始量纲）...")
    C_test = load_mat_data(PATH_C_TEST)
    assert C_test.shape[0] >= N_TEST
    C_test = C_test[:N_TEST]                  # (100, 5, 51, 51)

    # -------- Step 4: 归一化参数 --------
    print("\n[Step 4] 读取代理模型归一化参数...")
    with np.load(KH_NORM_PATH) as d:
        h_mean, h_std = float(d['h_mean']), float(d['h_std'])
    with np.load(KC_NORM_PATH) as d:
        c_mean, c_std = float(d['c_mean']), float(d['c_std'])
    print(f"  H 归一化: mean={h_mean:.4f}, std={h_std:.4f}")
    print(f"  C 归一化: mean={c_mean:.4f}, std={c_std:.4f}")

    # -------- Step 5: 保存 --------
    print("\n[Step 5] 保存...")
    np.savez(OUTPUT_PATH,
             K_test    = K_test,      # (100, 51, 51)
             H_test    = H_test,      # (100, 5, 51, 51)
             C_test    = C_test,      # (100, 5, 51, 51)
             h_mean    = h_mean,
             h_std     = h_std,
             c_mean    = c_mean,
             c_std     = c_std,
             grid_size = GRID_SIZE,
             n_test    = N_TEST)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"  保存至: {OUTPUT_PATH}  ({size_mb:.1f} MB)")

    # -------- Step 6: 抽检可视化 --------
    print("\n[Step 6] 抽检可视化...")
    check_indices = [0, 25, 50, 75, 99]
    fig, axes = plt.subplots(len(check_indices), 4,
                             figsize=(16, 4 * len(check_indices)))

    for row, idx in enumerate(check_indices):
        im = axes[row, 0].imshow(K_test[idx], origin='lower', cmap='jet',
                                 vmin=0, vmax=1)
        axes[row, 0].set_title(f'K_true [{idx}]')
        plt.colorbar(im, ax=axes[row, 0])

        im = axes[row, 1].imshow(H_test[idx, 0], origin='lower', cmap='coolwarm')
        axes[row, 1].set_title(f'H_true [{idx}] p=1')
        plt.colorbar(im, ax=axes[row, 1])

        im = axes[row, 2].imshow(H_test[idx, 4], origin='lower', cmap='coolwarm')
        axes[row, 2].set_title(f'H_true [{idx}] p=5')
        plt.colorbar(im, ax=axes[row, 2])

        im = axes[row, 3].imshow(C_test[idx, 4], origin='lower', cmap='hot')
        axes[row, 3].set_title(f'C_true [{idx}] t=5')
        plt.colorbar(im, ax=axes[row, 3])

    plt.suptitle('Test Data Spot Check — Physics Simulator Reference Fields',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'test_data_check.png'), dpi=150)
    plt.show()
    print("\nDone!")


if __name__ == "__main__":
    main()
