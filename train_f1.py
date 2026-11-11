"""
train_f1.py
FNO-1: K → H（5 次抽水试验的水头场）训练脚本
- 固定随机种子（SEED = 42），训练/验证按 9:1 划分
- 保存验证集最优权重、归一化参数、训练历史与验证集索引
  （val_indices.npy 用于同化阶段在验证集上校准 FNO-1 代理误差 σ_f1）
- 输出训练损失曲线
结果保存至 ./fno_kh_results/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import time
import random
from torch.utils.data import DataLoader, Dataset, random_split

# ==========================================
# 0. 随机种子
# ==========================================
SEED = 42

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed(SEED)

# ==========================================
# 1. 参数配置
# ==========================================
PATH_K   = './data/DataK_6000_51_51.mat'
PATH_H   = './data/DataH_6000_5_51_51.mat'
SAVE_DIR = './fno_kh_results'
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LR         = 1e-3
EPOCHS     = 300
MODES      = 12
WIDTH      = 64
OUT_CH     = 5


# ==========================================
# 2. LpLoss
# ==========================================
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        self.d = d; self.p = p
        self.reduction    = reduction
        self.size_average = size_average

    def __call__(self, x, y):
        n          = x.size(0)
        diff_norms = torch.norm(
            x.reshape(n, -1) - y.reshape(n, -1), self.p, 1)
        y_norms    = torch.norm(y.reshape(n, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms


# ==========================================
# 3. FNO 模型
# ==========================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale  = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, inp, w):
        return torch.einsum("bixy,ioxy->boxy", inp, w)

    def forward(self, x):
        B     = x.shape[0]
        x_ft  = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_channels,
                             x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :,  :self.modes1,  :self.modes2] = self.compl_mul2d(
            x_ft[:, :,  :self.modes1,  :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:,  :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:,  :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width,
                 in_channels=1, out_channels=5,
                 h_mean=0.0, h_std=1.0):
        super().__init__()
        self.modes1  = modes1
        self.modes2  = modes2
        self.width   = width
        self.padding = 9

        self.register_buffer('h_mean',
                             torch.tensor(h_mean, dtype=torch.float32))
        self.register_buffer('h_std',
                             torch.tensor(h_std,  dtype=torch.float32))

        self.p     = nn.Conv2d(in_channels, width, 1)
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.w0    = nn.Conv2d(width, width, 1)
        self.w1    = nn.Conv2d(width, width, 1)
        self.w2    = nn.Conv2d(width, width, 1)
        self.w3    = nn.Conv2d(width, width, 1)
        self.bn0   = nn.BatchNorm2d(width)
        self.bn1   = nn.BatchNorm2d(width)
        self.bn2   = nn.BatchNorm2d(width)
        self.bn3   = nn.BatchNorm2d(width)
        self.q     = nn.Conv2d(width, 128, 1)
        self.out   = nn.Conv2d(128, out_channels, 1)

    def forward(self, x, return_normalized=False):
        x = F.pad(x, [0, self.padding, 0, self.padding])
        x = self.p(x)
        x = F.gelu(x + self.bn0(self.conv0(x) + self.w0(x)))
        x = F.gelu(x + self.bn1(self.conv1(x) + self.w1(x)))
        x = F.gelu(x + self.bn2(self.conv2(x) + self.w2(x)))
        x = x + self.bn3(self.conv3(x) + self.w3(x))
        x = x[..., :-self.padding, :-self.padding]
        x = F.gelu(self.q(x))
        x = self.out(x)
        if not return_normalized:
            x = x * self.h_std + self.h_mean
        return x


# ==========================================
# 4. 数据集
# ==========================================
class KHDataset(Dataset):
    def __init__(self, k_path, h_path):
        k_mat = sio.loadmat(k_path)
        k_key = [k for k in k_mat if not k.startswith('__')][0]
        self.k_data = k_mat[k_key].astype(np.float32)

        h_mat = sio.loadmat(h_path)
        h_key = [k for k in h_mat if not k.startswith('__')][0]
        self.h_data = h_mat[h_key].astype(np.float32)

        if self.k_data.ndim == 3:
            self.k_data = np.expand_dims(self.k_data, 1)

        self.k_tensor = torch.from_numpy(self.k_data)

        self.h_mean = float(np.mean(self.h_data))
        self.h_std  = float(np.std(self.h_data))
        print(f"K shape: {self.k_data.shape}")
        print(f"H shape: {self.h_data.shape}")
        print(f"H 归一化参数: Mean={self.h_mean:.4f}, Std={self.h_std:.4f}")

        h_norm = (self.h_data - self.h_mean) / (self.h_std + 1e-8)
        self.h_tensor = torch.from_numpy(h_norm)

    def __len__(self):
        return len(self.k_tensor)

    def __getitem__(self, idx):
        return self.k_tensor[idx], self.h_tensor[idx]

    def denormalize_h(self, h_norm):
        return h_norm * self.h_std + self.h_mean


# ==========================================
# 5. 训练主程序
# ==========================================
def main():
    print("=" * 60)
    print("  FNO-1: K → H 训练")
    print(f"  Device: {DEVICE} | Seed: {SEED}")
    print("=" * 60)

    dataset    = KHDataset(PATH_K, PATH_H)
    train_size = int(0.9 * len(dataset))
    val_size   = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # 保存验证集索引
    val_indices = list(val_dataset.indices)
    np.save(os.path.join(SAVE_DIR, 'val_indices.npy'), val_indices)
    print(f"训练集: {train_size} | 验证集: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    model = FNO2d(
        modes1=MODES, modes2=MODES, width=WIDTH,
        in_channels=1, out_channels=OUT_CH,
        h_mean=dataset.h_mean, h_std=dataset.h_std
    ).to(DEVICE)
    print(f"模型归一化参数: h_mean={model.h_mean.item():.4f}, "
          f"h_std={model.h_std.item():.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.5)
    criterion = LpLoss(size_average=False)

    best_val_loss   = float('inf')
    best_model_path = os.path.join(SAVE_DIR, 'best_fno_kh.pth')
    history = {'train_loss': [], 'val_loss': []}
    start   = time.time()

    for epoch in range(EPOCHS):
        # --- 训练 ---
        model.train()
        train_l2 = 0.0
        for k_batch, h_batch in train_loader:
            k_batch, h_batch = k_batch.to(DEVICE), h_batch.to(DEVICE)
            optimizer.zero_grad()
            out  = model(k_batch, return_normalized=True)
            loss = criterion(out, h_batch)
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()
        scheduler.step()
        avg_train = train_l2 / train_size

        # --- 验证 ---
        model.eval()
        val_l2 = 0.0
        with torch.no_grad():
            for k_batch, h_batch in val_loader:
                k_batch, h_batch = k_batch.to(DEVICE), h_batch.to(DEVICE)
                out     = model(k_batch, return_normalized=True)
                val_l2 += criterion(out, h_batch).item()
        avg_val = val_l2 / val_size

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{EPOCHS}: "
                  f"Train={avg_train:.5f} | Val={avg_val:.5f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_model_path)

    elapsed = (time.time() - start) / 60
    print(f"\n训练完成，耗时 {elapsed:.1f} min | 最佳验证损失: {best_val_loss:.5f}")

    # ==========================================
    # 6. 保存中间结果
    # ==========================================
    np.savez(os.path.join(SAVE_DIR, 'history.npz'),
             train_loss=history['train_loss'],
             val_loss=history['val_loss'])

    np.savez(os.path.join(SAVE_DIR, 'normalization_params.npz'),
             h_mean=dataset.h_mean,
             h_std=dataset.h_std)
    print("训练历史与归一化参数已保存")

    # ==========================================
    # 7. 训练损失曲线
    # ==========================================
    epochs_range = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs_range, history['train_loss'], label='Train Rel-L2')
    axes[0].plot(epochs_range, history['val_loss'],   label='Val Rel-L2')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Relative L2 Loss')
    axes[0].set_title('FNO K→H Training Curve')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].semilogy(epochs_range, history['train_loss'], label='Train Rel-L2')
    axes[1].semilogy(epochs_range, history['val_loss'],   label='Val Rel-L2')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Relative L2 Loss (log scale)')
    axes[1].set_title('FNO K→H Training Curve (Log Scale)')
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(f'FNO K→H Training  (Best Val={best_val_loss:.5f})',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png'), dpi=150)
    plt.show()
    print(f"训练曲线已保存至 {SAVE_DIR}/training_curves.png")


if __name__ == "__main__":
    main()
