"""
train_vae.py
VAE 训练脚本
- 修改：去除早停机制，固定 EPOCHS = 100
- 保留：随机种子、最优保存机制、中间结果保存（history、潜变量统计等）
结果保存至 ./vae_results/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import random
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import r2_score

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
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ==========================================
# 1. 参数配置
# ==========================================
DATA_PATH = './data/DataK_6000_51_51.mat'
SAVE_DIR  = './vae_results'
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 32
LR           = 1e-3
EPOCHS       = 100  # <--- 已修改为 100
LATENT_DIM   = 128


# ==========================================
# 2. 数据集
# ==========================================
class KDataset(Dataset):
    def __init__(self, file_path):
        mat_data = sio.loadmat(file_path)
        keys     = [k for k in mat_data.keys() if not k.startswith('__')]
        data     = mat_data[keys[0]].astype(np.float32)
        self.x   = torch.from_numpy(data).unsqueeze(1)
        print(f"数据加载完成，形状: {self.x.shape}, "
              f"范围: [{data.min():.2f}, {data.max():.2f}]")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


# ==========================================
# 3. 模型定义
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(x + self.conv(x))


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2)
        )
        self.enc_res1  = ResidualBlock(64)
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2)
        )
        self.enc_res2  = ResidualBlock(128)
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2)
        )
        self.enc_res3  = ResidualBlock(256)

        self.flatten   = nn.Flatten()
        self.fc_mu     = nn.Linear(256 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(256 * 7 * 7, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256 * 7 * 7)
        self.unflatten      = nn.Unflatten(1, (256, 7, 7))

        self.dec_res3  = ResidualBlock(256)
        self.dec_conv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2)
        )
        self.dec_res2  = ResidualBlock(128)
        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2)
        )
        self.dec_res1  = ResidualBlock(64)
        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.enc_conv1(x)
        x = self.enc_res1(x)
        x = self.enc_conv2(x)
        x = self.enc_res2(x)
        x = self.enc_conv3(x)
        x = self.enc_res3(x)
        x = self.flatten(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = self.unflatten(x)
        x = self.dec_res3(x)
        x = self.dec_conv3(x)
        x = self.dec_res2(x)
        x = self.dec_conv2(x)
        x = self.dec_res1(x)
        x = self.dec_conv1(x)
        return self.refine(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ==========================================
# 4. 损失函数
# ==========================================
def gradient_loss(pred, target):
    def gradient(t):
        dx = t[:, :, :,  1:] - t[:, :, :, :-1]
        dy = t[:, :,  1:, :] - t[:, :, :-1, :]
        return dx, dy
    pdx, pdy = gradient(pred)
    tdx, tdy = gradient(target)
    return F.mse_loss(pdx, tdx) + F.mse_loss(pdy, tdy)


def loss_function(recon_x, x, mu, logvar, beta=0.0001, gamma=0.1):
    mse       = F.mse_loss(recon_x, x, reduction='sum')
    kld       = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    grad      = gradient_loss(recon_x, x) * x.numel()
    total     = mse + beta * kld + gamma * grad
    return total, mse, kld, grad


# ==========================================
# 5. 训练主程序
# ==========================================
def main():
    print("=" * 60)
    print("  VAE V4 训练 (残差连接 + 梯度损失)")
    print(f"  Device: {DEVICE} | Seed: {SEED}")
    print("=" * 60)

    # 数据加载与划分
    dataset    = KDataset(DATA_PATH)
    train_size = int(0.9 * len(dataset))
    val_size   = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # 保存验证集索引，供评估脚本使用
    val_indices = list(val_dataset.indices)
    np.save(os.path.join(SAVE_DIR, 'val_indices.npy'), val_indices)
    print(f"训练集: {train_size} | 验证集: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    model     = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_loss   = float('inf')
    history = {
        'train_mse': [], 'val_mse': [],
        'train_total': [], 'val_r2': []
    }

    best_model_path = os.path.join(SAVE_DIR, 'best_vae.pth')

    for epoch in range(EPOCHS):
        # --- 训练 ---
        model.train()
        train_mse_total   = 0.0
        train_total_total = 0.0

        for batch in train_loader:
            batch = batch.to(DEVICE)
            recon, mu, logvar = model(batch)
            loss, mse, kld, grad = loss_function(recon, batch, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_mse_total   += mse.item()
            train_total_total += loss.item()

        scheduler.step()

        # --- 验证 ---
        model.eval()
        val_mse_sum = 0.0
        all_true, all_pred = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                recon, _, _ = model(batch)
                val_mse_sum += F.mse_loss(recon, batch, reduction='sum').item()
                all_true.append(batch.cpu().numpy())
                all_pred.append(recon.cpu().numpy())

        avg_train_mse   = train_mse_total   / train_size
        avg_train_total = train_total_total / train_size
        avg_val_mse     = val_mse_sum       / val_size

        all_true_np = np.concatenate(all_true)
        all_pred_np = np.concatenate(all_pred)
        val_r2      = r2_score(all_true_np.flatten(), all_pred_np.flatten())

        history['train_mse'].append(avg_train_mse)
        history['train_total'].append(avg_train_total)
        history['val_mse'].append(avg_val_mse)
        history['val_r2'].append(val_r2)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS}: "
                  f"Train MSE={avg_train_mse:.6f} | "
                  f"Val MSE={avg_val_mse:.6f} | "
                  f"Val R²={val_r2:.4f} | "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")

        # 最优保存机制（已去除早停判定）
        if avg_val_mse < best_val_loss:
            best_val_loss = avg_val_mse
            torch.save(model.state_dict(), best_model_path)

    print(f"\n训练完成！最佳验证 MSE: {best_val_loss:.6f}")

    # ==========================================
    # 6. 保存中间结果
    # ==========================================
    np.savez(os.path.join(SAVE_DIR, 'history.npz'),
             train_mse=history['train_mse'],
             train_total=history['train_total'],
             val_mse=history['val_mse'],
             val_r2=history['val_r2'])
    print("训练历史已保存至 history.npz")

    # 计算并保存潜变量统计（供下游任务使用）
    print("计算全量潜变量统计...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    all_z = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=128, shuffle=False):
            batch = batch.to(DEVICE)
            mu, _ = model.encode(batch)
            all_z.append(mu.cpu().numpy())
    all_z = np.concatenate(all_z)
    np.savez(os.path.join(SAVE_DIR, 'latent_statistics.npz'),
             z_mean=np.mean(all_z, axis=0),
             z_std=np.std(all_z,  axis=0))
    print("潜变量统计已保存至 latent_statistics.npz")

    # ==========================================
    # 7. 训练损失曲线
    # ==========================================
    epochs_range = range(1, len(history['train_mse']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].semilogy(epochs_range, history['train_mse'], label='Train MSE')
    axes[0].semilogy(epochs_range, history['val_mse'],   label='Val MSE')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE (log scale)')
    axes[0].set_title('Reconstruction MSE')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].semilogy(epochs_range, history['train_total'], color='purple')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Total Loss (log scale)')
    axes[1].set_title('Total Training Loss')
    axes[1].grid(True)

    axes[2].plot(epochs_range, history['val_r2'], color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('R²')
    axes[2].set_title('Validation R² over Epochs')
    axes[2].grid(True)

    plt.suptitle(f'VAE V4 Training Curves (Best Val MSE={best_val_loss:.6f})',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png'), dpi=150)
    plt.show()
    print(f"训练曲线已保存至 {SAVE_DIR}/training_curves.png")


if __name__ == "__main__":
    main()