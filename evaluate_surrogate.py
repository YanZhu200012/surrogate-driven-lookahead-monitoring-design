"""
evaluate_surrogate.py
使用 100 个测试样本评估三个代理模型：
  - VAE    (K 重建)          → ./vae_results/
  - FNO-1  (K → H 预测)      → ./fno_kh_results/
  - FNO-2  (K → C 预测)      → ./fno_kc_results/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import random
import time
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity as ssim

# ==========================================
# 0. 随机种子
# ==========================================
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed(SEED)

# ==========================================
# 1. 期刊图像全局样式（字号在原基础上 +2）
# ==========================================
JOURNAL_RC = {
    'font.family':        'Arial',
    'font.size':          12,
    'axes.titlesize':     13,
    'axes.labelsize':     12,
    'xtick.labelsize':    10,
    'ytick.labelsize':    11,
    'legend.fontsize':    11,
    'figure.titlesize':   14,
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
}

CBAR_KW = dict(fraction=0.046, pad=0.04)
SAVE_KW = dict(dpi=300, bbox_inches='tight')

def apply_journal_style():
    plt.rcParams.update(JOURNAL_RC)

# ==========================================
# 2. 路径配置
# ==========================================
TEST_PATH_K = './data/test/DataK_test_100_51_51.mat'
TEST_PATH_H = './data/test/DataH_test_100_5_51_51.mat'
TEST_PATH_C = './data/test/DataC_100_5_51_51_Original.mat'

VAE_DIR  = './vae_results'
FNO1_DIR = './fno_kh_results'
FNO2_DIR = './fno_kc_results'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_CH = 5

# ==========================================
# 3. 数据加载
# ==========================================
def load_mat(path):
    mat = sio.loadmat(path)
    key = [k for k in mat if not k.startswith('__')][0]
    return mat[key].astype(np.float32)

def load_test_data():
    k = load_mat(TEST_PATH_K)
    h = load_mat(TEST_PATH_H)
    c = load_mat(TEST_PATH_C)
    if k.ndim == 3:
        k = np.expand_dims(k, 1)
    print("测试数据加载完成:")
    print(f"  K: {k.shape}  H: {h.shape}  C: {c.shape}")
    return k, h, c

# ==========================================
# 4. 模型定义
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(x + self.conv(x))


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.enc_res1  = ResidualBlock(64)
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.enc_res2  = ResidualBlock(128)
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.enc_res3  = ResidualBlock(256)
        self.flatten   = nn.Flatten()
        self.fc_mu     = nn.Linear(256 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(256 * 7 * 7, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 256 * 7 * 7)
        self.unflatten = nn.Unflatten(1, (256, 7, 7))
        self.dec_res3  = ResidualBlock(256)
        self.dec_conv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.dec_res2  = ResidualBlock(128)
        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.dec_res1  = ResidualBlock(64)
        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2))
        self.refine = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid())

    def encode(self, x):
        x = self.enc_res1(self.enc_conv1(x))
        x = self.enc_res2(self.enc_conv2(x))
        x = self.enc_res3(self.enc_conv3(x))
        x = self.flatten(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        x = self.unflatten(self.decoder_input(z))
        x = self.dec_conv3(self.dec_res3(x))
        x = self.dec_conv2(self.dec_res2(x))
        x = self.dec_conv1(self.dec_res1(x))
        return self.refine(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels,
                               modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels,
                               modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, inp, w):
        return torch.einsum("bixy,ioxy->boxy", inp, w)

    def forward(self, x):
        B      = x.shape[0]
        x_ft   = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_channels,
                             x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :,  :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :,  :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


def _fno_body(self, x, return_normalized, norm_std, norm_mean):
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
        x = x * norm_std + norm_mean
    return x


def _build_fno_layers(self, in_channels, out_channels, width, modes1, modes2):
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


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width,
                 in_channels=1, out_channels=5,
                 h_mean=0.0, h_std=1.0):
        super().__init__()
        self.modes1 = modes1; self.modes2 = modes2
        self.width  = width;  self.padding = 9
        self.register_buffer('h_mean', torch.tensor(h_mean, dtype=torch.float32))
        self.register_buffer('h_std',  torch.tensor(h_std,  dtype=torch.float32))
        _build_fno_layers(self, in_channels, out_channels, width, modes1, modes2)

    def forward(self, x, return_normalized=False):
        return _fno_body(self, x, return_normalized, self.h_std, self.h_mean)


class FNO2d_KC(nn.Module):
    def __init__(self, modes1, modes2, width,
                 in_channels=1, out_channels=5,
                 c_mean=0.0, c_std=1.0):
        super().__init__()
        self.modes1 = modes1; self.modes2 = modes2
        self.width  = width;  self.padding = 9
        self.register_buffer('c_mean', torch.tensor(c_mean, dtype=torch.float32))
        self.register_buffer('c_std',  torch.tensor(c_std,  dtype=torch.float32))
        _build_fno_layers(self, in_channels, out_channels, width, modes1, modes2)

    def forward(self, x, return_normalized=False):
        return _fno_body(self, x, return_normalized, self.c_std, self.c_mean)


# ==========================================
# 5. 指标计算
# ==========================================
def compute_metrics(true, pred, n_timesteps=None):
    overall_r2    = r2_score(true.flatten(), pred.flatten())
    overall_rmse  = np.sqrt(np.mean((true - pred) ** 2))
    overall_range = float(true.max() - true.min())
    overall_nrmse = overall_rmse / overall_range if overall_range > 0 else 0.0
    overall_mae   = np.mean(np.abs(true - pred))
    metrics = {
        'overall_r2':    overall_r2,
        'overall_nrmse': overall_nrmse,
        'overall_mae':   overall_mae,
    }
    if n_timesteps is not None:
        per_t_r2, per_t_nrmse = [], []
        for t in range(n_timesteps):
            per_t_r2.append(
                r2_score(true[:, t].flatten(), pred[:, t].flatten()))
            t_rmse  = float(np.sqrt(np.mean((true[:, t] - pred[:, t]) ** 2)))
            t_range = float(true[:, t].max() - true[:, t].min())
            per_t_nrmse.append(t_rmse / t_range if t_range > 0 else 0.0)
        metrics['per_t_r2']    = per_t_r2
        metrics['per_t_nrmse'] = per_t_nrmse
    return metrics


def print_metrics(metrics, label='', n_timesteps=None):
    print(f"\n{'='*52}")
    print(f"  {label} 测试集评估结果")
    print(f"{'='*52}")
    print(f"  整体 R²:    {metrics['overall_r2']:.4f}")
    print(f"  整体 NRMSE: {metrics['overall_nrmse']:.6f}")
    print(f"  整体 MAE:   {metrics['overall_mae']:.6f}")
    if n_timesteps and 'per_t_r2' in metrics:
        print(f"  逐时间步:")
        for t in range(n_timesteps):
            print(f"    t={t+1}: R²={metrics['per_t_r2'][t]:.4f}  "
                  f"NRMSE={metrics['per_t_nrmse'][t]:.6f}")


# ==========================================
# 6. colorbar 工具
# ==========================================
def add_colorbar(fig, im, ax):
    """统一 colorbar，无单位标签"""
    cbar = fig.colorbar(im, ax=ax, **CBAR_KW)
    cbar.ax.tick_params(labelsize=JOURNAL_RC['xtick.labelsize'])
    return cbar


# ==========================================
# 7. 评估 VAE（K 重建）
# ==========================================
def evaluate_vae(k_test):
    print("\n" + "=" * 60)
    print("  评估 VAE (K 重建)")
    print("=" * 60)

    model_path = os.path.join(VAE_DIR, 'best_vae.pth')
    assert os.path.exists(model_path), f"找不到模型权重: {model_path}"

    model = VAE(latent_dim=128).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    k_tensor  = torch.from_numpy(k_test).to(DEVICE)
    all_recon = []
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    t_start = time.time()
    with torch.no_grad():
        for i in range(0, len(k_tensor), 32):
            batch = k_tensor[i:i+32]
            recon, _, _ = model(batch)
            all_recon.append(recon.cpu().numpy())
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    t_end = time.time()
    infer_total = t_end - t_start
    infer_per_sample = infer_total / len(k_tensor)
    print(f"  VAE 推理耗时: 总计 {infer_total:.4f} s | "
          f"单样本 {infer_per_sample*1000:.4f} ms")

    all_recon = np.concatenate(all_recon)
    true      = k_test

    metrics = compute_metrics(true, all_recon)
    metrics['infer_total_s']      = infer_total
    metrics['infer_per_sample_ms'] = infer_per_sample * 1000
    ssim_scores = [
        ssim(true[i, 0], all_recon[i, 0], data_range=1.0)
        for i in range(len(true))
    ]
    metrics['ssim'] = float(np.mean(ssim_scores))
    print_metrics(metrics, label='VAE (K 重建)')
    print(f"  SSIM: {metrics['ssim']:.4f}")

    np.savez(os.path.join(VAE_DIR, 'test_results.npz'),
             k_true=true, k_recon=all_recon,
             r2=metrics['overall_r2'],
             nrmse=metrics['overall_nrmse'],
             ssim=metrics['ssim'])

    _plot_vae_reconstruction(true, all_recon)
    _plot_vae_analysis(true, all_recon, metrics)

    return metrics


def _pick_samples(errors, n=4):
    """按误差从小到大均匀挑选 n 个代表性样本"""
    idx_sorted = np.argsort(errors)
    n_total    = len(idx_sorted)
    picks = [
        idx_sorted[0],
        idx_sorted[n_total // 3],
        idx_sorted[2 * n_total // 3],
        idx_sorted[-1],
    ]
    return picks[:n]


def _plot_vae_reconstruction(true, recon):
    """
    VAE 重建对比图
    布局：3 行 × 4 列
    """
    apply_journal_style()

    per_sample_rmse = np.sqrt(np.mean((true - recon) ** 2, axis=(1, 2, 3)))
    sample_ids      = _pick_samples(per_sample_rmse, n=4)
    n_show          = len(sample_ids)
    print(f"  [VAE] 绘图选中样本 Index -> {sample_ids}")

    fig = plt.figure(figsize=(3.5 * n_show, 9.5))
    gs  = gridspec.GridSpec(3, n_show, figure=fig,
                            hspace=0.12, wspace=0.22)

    row_labels = ['True K', 'Reconstructed K', 'Error']

    for col, idx in enumerate(sample_ids):
        t_data = true[idx, 0]
        r_data = recon[idx, 0]
        diff   = t_data - r_data
        vmin, vmax = t_data.min(), t_data.max()

        ax0 = fig.add_subplot(gs[0, col])
        im0 = ax0.imshow(t_data, cmap='viridis',
                         vmin=vmin, vmax=vmax,
                         origin='lower', aspect='equal')
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_title(f'Sample #{idx}',
                      fontsize=JOURNAL_RC['axes.titlesize'], pad=8)
        if col == 0:
            ax0.set_ylabel(row_labels[0],
                           fontsize=JOURNAL_RC['axes.labelsize'], labelpad=10)
        add_colorbar(fig, im0, ax0)

        ax1 = fig.add_subplot(gs[1, col])
        im1 = ax1.imshow(r_data, cmap='viridis',
                         vmin=vmin, vmax=vmax,
                         origin='lower', aspect='equal')
        ax1.set_xticks([]); ax1.set_yticks([])
        if col == 0:
            ax1.set_ylabel(row_labels[1],
                           fontsize=JOURNAL_RC['axes.labelsize'], labelpad=10)
        add_colorbar(fig, im1, ax1)

        ax2 = fig.add_subplot(gs[2, col])
        vd  = max(float(np.abs(diff).max()), 1e-8)
        im2 = ax2.imshow(diff, cmap='RdBu_r',
                         vmin=-vd, vmax=vd,
                         origin='lower', aspect='equal')
        ax2.set_xticks([]); ax2.set_yticks([])
        if col == 0:
            ax2.set_ylabel(row_labels[2],
                           fontsize=JOURNAL_RC['axes.labelsize'], labelpad=10)
        add_colorbar(fig, im2, ax2)

    fig.subplots_adjust(left=0.08, right=0.92, top=0.93, bottom=0.07)
    plt.savefig(os.path.join(VAE_DIR, 'test_reconstruction.png'), **SAVE_KW)
    plt.close()
    print(f"  → 保存: {VAE_DIR}/test_reconstruction.png")


def _plot_vae_analysis(true, recon, metrics):
    """VAE 分析图：散点图 + SSIM 直方图"""
    apply_journal_style()

    ssim_scores = [
        ssim(true[i, 0], recon[i, 0], data_range=1.0)
        for i in range(len(true))
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    step = max(1, len(true.flatten()) // 5000)
    axes[0].scatter(true.flatten()[::step], recon.flatten()[::step],
                    alpha=0.3, s=3, color='steelblue', label='Samples')
    lim = [min(true.min(), recon.min()), max(true.max(), recon.max())]
    axes[0].plot(lim, lim, 'r--', lw=1.2, label='1:1 line')
    axes[0].set_xlabel('True K')
    axes[0].set_ylabel('Reconstructed K')
    axes[0].legend()
    axes[0].grid(True, lw=0.4, alpha=0.6)

    axes[1].hist(ssim_scores, bins=20, color='steelblue',
                 edgecolor='white', alpha=0.85)
    axes[1].axvline(np.mean(ssim_scores), color='red', lw=1.5,
                    linestyle='--',
                    label=f'Mean = {np.mean(ssim_scores):.4f}')
    axes[1].set_xlabel('SSIM')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    axes[1].grid(True, lw=0.4, alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(VAE_DIR, 'test_analysis.png'), **SAVE_KW)
    plt.close()
    print(f"  → 保存: {VAE_DIR}/test_analysis.png")


# ==========================================
# 8. 评估 FNO-1（K → H）
# ==========================================
def evaluate_fno1(k_test, h_test):
    print("\n" + "=" * 60)
    print("  评估 FNO-1 (K → H)")
    print("=" * 60)

    model_path = os.path.join(FNO1_DIR, 'best_fno_kh.pth')
    norm_path  = os.path.join(FNO1_DIR, 'normalization_params.npz')
    assert os.path.exists(model_path), f"找不到模型权重: {model_path}"
    assert os.path.exists(norm_path),  f"找不到归一化参数: {norm_path}"

    norm   = np.load(norm_path)
    h_mean = float(norm['h_mean'])
    h_std  = float(norm['h_std'])
    print(f"归一化参数: h_mean={h_mean:.4f}, h_std={h_std:.4f}")

    model = FNO2d(modes1=12, modes2=12, width=64,
                  in_channels=1, out_channels=OUT_CH,
                  h_mean=h_mean, h_std=h_std).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    k_tensor = torch.from_numpy(k_test).to(DEVICE)
    all_pred = []
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    t_start = time.time()
    with torch.no_grad():
        for i in range(0, len(k_tensor), 32):
            pred = model(k_tensor[i:i+32], return_normalized=False)
            all_pred.append(pred.cpu().numpy())
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    t_end = time.time()
    infer_total = t_end - t_start
    infer_per_sample = infer_total / len(k_tensor)
    print(f"  FNO-1 推理耗时: 总计 {infer_total:.4f} s | "
          f"单样本 {infer_per_sample*1000:.4f} ms")

    all_pred = np.concatenate(all_pred)
    true     = h_test

    metrics = compute_metrics(true, all_pred, n_timesteps=OUT_CH)
    metrics['infer_total_s']      = infer_total
    metrics['infer_per_sample_ms'] = infer_per_sample * 1000
    print_metrics(metrics, label='FNO-1 (K→H)', n_timesteps=OUT_CH)

    np.savez(os.path.join(FNO1_DIR, 'test_results.npz'),
             h_true=true, h_pred=all_pred,
             r2=metrics['overall_r2'],
             nrmse=metrics['overall_nrmse'],
             per_t_r2=metrics['per_t_r2'],
             per_t_nrmse=metrics['per_t_nrmse'],
             infer_total_s=infer_total,
             infer_per_sample_ms=infer_per_sample * 1000)

    _plot_fno_fields(true, all_pred,
                     save_dir=FNO1_DIR,
                     field_name='H',
                     cmap='turbo')

    _plot_fno_analysis(true, all_pred, metrics,
                       save_dir=FNO1_DIR,
                       field_name='H',
                       scatter_lim=None)

    return metrics


# ==========================================
# 9. 评估 FNO-2（K → C）
# ==========================================
def evaluate_fno2(k_test, c_test):
    print("\n" + "=" * 60)
    print("  评估 FNO-2 (K → C)")
    print("=" * 60)

    model_path = os.path.join(FNO2_DIR, 'best_fno_kc.pth')
    norm_path  = os.path.join(FNO2_DIR, 'normalization_params.npz')
    assert os.path.exists(model_path), f"找不到模型权重: {model_path}"
    assert os.path.exists(norm_path),  f"找不到归一化参数: {norm_path}"

    norm   = np.load(norm_path)
    c_mean = float(norm['c_mean'])
    c_std  = float(norm['c_std'])
    print(f"归一化参数: c_mean={c_mean:.4f}, c_std={c_std:.4f}")

    model = FNO2d_KC(modes1=12, modes2=12, width=64,
                     in_channels=1, out_channels=OUT_CH,
                     c_mean=c_mean, c_std=c_std).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    k_tensor = torch.from_numpy(k_test).to(DEVICE)
    all_pred = []
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    t_start = time.time()
    with torch.no_grad():
        for i in range(0, len(k_tensor), 32):
            pred = model(k_tensor[i:i+32], return_normalized=False)
            all_pred.append(pred.cpu().numpy())
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    t_end = time.time()
    infer_total = t_end - t_start
    infer_per_sample = infer_total / len(k_tensor)
    print(f"  FNO-2 推理耗时: 总计 {infer_total:.4f} s | "
          f"单样本 {infer_per_sample*1000:.4f} ms")

    all_pred = np.concatenate(all_pred)
    true     = c_test

    metrics = compute_metrics(true, all_pred, n_timesteps=OUT_CH)
    metrics['infer_total_s']      = infer_total
    metrics['infer_per_sample_ms'] = infer_per_sample * 1000
    print_metrics(metrics, label='FNO-2 (K→C)', n_timesteps=OUT_CH)

    np.savez(os.path.join(FNO2_DIR, 'test_results.npz'),
             c_true=true, c_pred=all_pred,
             r2=metrics['overall_r2'],
             nrmse=metrics['overall_nrmse'],
             per_t_r2=metrics['per_t_r2'],
             per_t_nrmse=metrics['per_t_nrmse'],
             infer_total_s=infer_total,
             infer_per_sample_ms=infer_per_sample * 1000)

    _plot_fno_fields(true, all_pred,
                     save_dir=FNO2_DIR,
                     field_name='C',
                     cmap='turbo',
                     vrange=(0, 1))

    _plot_fno_analysis(true, all_pred, metrics,
                       save_dir=FNO2_DIR,
                       field_name='C',
                       scatter_lim=(0, 1))

    return metrics


# ==========================================
# 10. FNO 公共绘图函数
# ==========================================
def _plot_fno_fields(true, pred, save_dir, field_name, cmap, vrange=None):
    """
    FNO 场对比图（Low / Medium / High 误差各一张）
    布局：3 行（True / Predicted / Error） × OUT_CH 列（时间步）
    """
    apply_journal_style()

    per_sample_rmse = np.sqrt(np.mean((true - pred) ** 2, axis=(1, 2, 3)))
    idx_sorted      = np.argsort(per_sample_rmse)
    n_total         = len(idx_sorted)
    samples = {
        'low':    idx_sorted[2],
        'medium': idx_sorted[n_total // 2],
        'high':   idx_sorted[-2],
    }
    print(f"  [{field_name}场] 选中样本 -> "
          f"Low: {samples['low']}, "
          f"Medium: {samples['medium']}, "
          f"High: {samples['high']}")

    row_labels = [f'True {field_name}',
                  f'Predicted {field_name}',
                  'Error']

    for group_name, sample_idx in samples.items():
        t_arr = true[sample_idx]
        p_arr = pred[sample_idx]

        fig = plt.figure(figsize=(OUT_CH * 3.2, 8.0))
        gs  = gridspec.GridSpec(3, OUT_CH, figure=fig,
                                hspace=0.12, wspace=0.30)

        for t in range(OUT_CH):
            vmin = t_arr[t].min() if vrange is None else vrange[0]
            vmax = t_arr[t].max() if vrange is None else vrange[1]

            ax0 = fig.add_subplot(gs[0, t])
            im0 = ax0.imshow(t_arr[t], cmap=cmap,
                             vmin=vmin, vmax=vmax,
                             origin='lower', aspect='equal')
            ax0.set_xticks([]); ax0.set_yticks([])
            ax0.set_title(f't = {t + 1}',
                          fontsize=JOURNAL_RC['axes.titlesize'], pad=8)
            if t == 0:
                ax0.set_ylabel(row_labels[0],
                               fontsize=JOURNAL_RC['axes.labelsize'], labelpad=10)
            add_colorbar(fig, im0, ax0)

            ax1 = fig.add_subplot(gs[1, t])
            im1 = ax1.imshow(p_arr[t], cmap=cmap,
                             vmin=vmin, vmax=vmax,
                             origin='lower', aspect='equal')
            ax1.set_xticks([]); ax1.set_yticks([])
            if t == 0:
                ax1.set_ylabel(row_labels[1],
                               fontsize=JOURNAL_RC['axes.labelsize'], labelpad=10)
            add_colorbar(fig, im1, ax1)

            diff = t_arr[t] - p_arr[t]
            vd   = max(float(np.abs(diff).max()), 1e-8)
            ax2  = fig.add_subplot(gs[2, t])
            im2  = ax2.imshow(diff, cmap='RdBu_r',
                              vmin=-vd, vmax=vd,
                              origin='lower', aspect='equal')
            ax2.set_xticks([]); ax2.set_yticks([])
            if t == 0:
                ax2.set_ylabel(row_labels[2],
                               fontsize=JOURNAL_RC['axes.labelsize'], labelpad=10)
            add_colorbar(fig, im2, ax2)

        save_path = os.path.join(save_dir, f'test_prediction_{group_name}.png')
        plt.savefig(save_path, **SAVE_KW)
        plt.close()
        print(f"  → 保存: {save_path}")


def _plot_fno_analysis(true, pred, metrics, save_dir, field_name, scatter_lim):
    """FNO 分析图：2×2 布局"""
    apply_journal_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    step = max(1, true.size // 8000)
    axes[0, 0].scatter(true.flatten()[::step],
                       pred.flatten()[::step],
                       alpha=0.25, s=3, color='steelblue')
    if scatter_lim:
        axes[0, 0].set_xlim(scatter_lim)
        axes[0, 0].set_ylim(scatter_lim)
        axes[0, 0].plot(scatter_lim, scatter_lim, 'r--', lw=1.2)
    else:
        lim = [min(true.min(), pred.min()), max(true.max(), pred.max())]
        axes[0, 0].plot(lim, lim, 'r--', lw=1.2)
    axes[0, 0].set_xlabel(f'True {field_name}')
    axes[0, 0].set_ylabel(f'Predicted {field_name}')
    axes[0, 0].grid(True, lw=0.4, alpha=0.6)

    t_vals  = range(1, OUT_CH + 1)
    r2_vals = metrics['per_t_r2']
    bars    = axes[0, 1].bar(t_vals, r2_vals,
                             color='steelblue', alpha=0.85,
                             edgecolor='white', width=0.6)
    for bar, r2 in zip(bars, r2_vals):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f'{r2:.4f}', ha='center',
                        fontsize=JOURNAL_RC['xtick.labelsize'])
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].set_xticks(list(t_vals))
    y_lo = max(0, min(r2_vals) - 0.02)
    axes[0, 1].set_ylim(y_lo, min(1.0, max(r2_vals) + 0.05))
    axes[0, 1].grid(True, axis='y', lw=0.4, alpha=0.6)

    nrmse_vals = metrics['per_t_nrmse']
    axes[1, 0].plot(list(t_vals), nrmse_vals,
                    marker='o', ms=5, lw=1.5, color='coral')
    axes[1, 0].fill_between(list(t_vals), nrmse_vals,
                             alpha=0.15, color='coral')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('NRMSE')
    axes[1, 0].set_xticks(list(t_vals))
    axes[1, 0].grid(True, lw=0.4, alpha=0.6)

    per_sample = np.sqrt(np.mean((true - pred) ** 2, axis=(1, 2, 3)))
    # 逐样本 NRMSE：除以每个样本的真实值范围
    per_sample_range = true.reshape(true.shape[0], -1).max(axis=1) - \
                       true.reshape(true.shape[0], -1).min(axis=1)
    per_sample_range = np.where(per_sample_range > 0, per_sample_range, 1.0)
    per_sample_nrmse = per_sample / per_sample_range
    axes[1, 1].hist(per_sample_nrmse, bins=20,
                    color='coral', edgecolor='white', alpha=0.85)
    axes[1, 1].axvline(np.mean(per_sample_nrmse), color='red',
                        lw=1.5, linestyle='--',
                        label=f'Mean = {np.mean(per_sample_nrmse):.4f}')
    axes[1, 1].set_xlabel('NRMSE per Sample')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend()
    axes[1, 1].grid(True, lw=0.4, alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'test_analysis.png')
    plt.savefig(save_path, **SAVE_KW)
    plt.close()
    print(f"  → 保存: {save_path}")


# ==========================================
# 11. FNO-1 & FNO-2 联合逐步指标图（新增）
# ==========================================
def plot_fno_combined_metrics(fno1_metrics, fno2_metrics, save_dir='.'):
    """
    一张大图，两个并排 Panel，每个 Panel 对应一个模型，
    内部用双 Y 轴同时展示 NRMSE（柱状图）和 R²（折线图）：

      Panel (a) — FNO-1  K → H  (p = 1 … 5)
        左 Y 轴（钢蓝）：NRMSE 柱状图
        右 Y 轴（珊瑚）：R²  折线图（圆点标记）

      Panel (b) — FNO-2  K → C  (t = 1 … 5)
        左 Y 轴（钢蓝）：NRMSE 柱状图
        右 Y 轴（珊瑚）：R²  折线图（方块标记）

    图像样式完全继承 JOURNAL_RC（Arial, dpi=300）。
    输出路径：{save_dir}/fno_combined_metrics.png
    """
    apply_journal_style()

    steps  = list(range(1, OUT_CH + 1))
    x      = np.array(steps, dtype=float)

    # ── 颜色 ────────────────────────────────────────────────
    COLOR_BAR  = '#2171b5'   # 钢蓝  — RMSE 柱
    COLOR_LINE = '#cb4637'   # 珊瑚  — R²   折线
    ALPHA_BAR  = 0.78
    BAR_W      = 0.50        # 柱宽（单柱居中）

    # ── 画布：1行 2列，宽裕尺寸 ─────────────────────────────
    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 5.2),
        constrained_layout=False
    )
    fig.subplots_adjust(left=0.07, right=0.93, top=0.87,
                        bottom=0.13, wspace=0.58)
    fig.suptitle(
        'FNO-1 (K→H) & FNO-2 (K→C)  —  Per-Step Performance Metrics',
        fontsize=JOURNAL_RC['figure.titlesize'],
        fontweight='bold',
        y=0.97
    )

    panel_cfg = [
        # (ax,       x_label,    nrmse_vals,                 r2_vals,
        #  panel_title,          marker)
        (axes[0],  'p (pumping experiment)',
         fno1_metrics['per_t_nrmse'], fno1_metrics['per_t_r2'],
         '(a)  FNO-1  K → H', 'o'),
        (axes[1],  't (timestep)',
         fno2_metrics['per_t_nrmse'], fno2_metrics['per_t_r2'],
         '(b)  FNO-2  K → C', 's'),
    ]

    for ax_l, xlabel, nrmse_vals, r2_vals, title, marker in panel_cfg:

        # ── 右 Y 轴（R²） ────────────────────────────────────
        ax_r = ax_l.twinx()

        # 柱状图：NRMSE（左轴，钢蓝）
        bars = ax_l.bar(
            x, nrmse_vals,
            width=BAR_W, color=COLOR_BAR, alpha=ALPHA_BAR,
            label='NRMSE', zorder=3
        )
        _bar_labels(ax_l, bars,
                    fmt='{:.4f}',
                    fontsize=JOURNAL_RC['xtick.labelsize'] - 1,
                    color=COLOR_BAR)

        # 折线图：R²（右轴，珊瑚）
        line, = ax_r.plot(
            steps, r2_vals,
            color=COLOR_LINE, marker=marker, ms=7,
            lw=2.0, linestyle='-',
            label='R²  (right axis)', zorder=5
        )
        # ax_r.fill_between(steps, r2_vals,
        #                   alpha=0.10, color=COLOR_LINE, zorder=2)
        # _line_labels(ax_r, steps, r2_vals,
        #              fmt='{:.4f}',
        #              fontsize=JOURNAL_RC['xtick.labelsize'] - 1,
        #              color=COLOR_LINE,
        #              dy=_r2_label_offset(r2_vals))

        # ── 左轴格式（NRMSE） ────────────────────────────────
        ax_l.set_xticks(steps)
        ax_l.set_xticklabels([str(s) for s in steps],
                             fontsize=JOURNAL_RC['xtick.labelsize'])
        ax_l.set_xlabel(xlabel, fontsize=JOURNAL_RC['axes.labelsize'])
        ax_l.set_ylabel('NRMSE',
                        color=COLOR_BAR,
                        fontsize=JOURNAL_RC['axes.labelsize'])
        ax_l.tick_params(axis='y', labelcolor=COLOR_BAR,
                         labelsize=JOURNAL_RC['ytick.labelsize'])
        ax_l.yaxis.grid(True, linestyle='--', alpha=0.35, zorder=0)
        ax_l.set_axisbelow(True)
        ax_l.spines['left'].set_edgecolor(COLOR_BAR)
        _expand_ylim(ax_l, nrmse_vals, margin=0.28, lower_zero=True)

        # ── 右轴格式（R²） ───────────────────────────────────
        ax_r.set_ylabel('R²',
                        color=COLOR_LINE,
                        fontsize=JOURNAL_RC['axes.labelsize'])
        ax_r.tick_params(axis='y', labelcolor=COLOR_LINE,
                         labelsize=JOURNAL_RC['ytick.labelsize'])
        ax_r.spines['right'].set_edgecolor(COLOR_LINE)
        _expand_ylim(ax_r, r2_vals, margin=0.10, lower_zero=False)

        # ── 合并图例 ─────────────────────────────────────────
        h_bar, l_bar = ax_l.get_legend_handles_labels()
        h_line       = [line]
        # l_line       = ['R²  (right axis)']
        l_line       = ['R²']
        ax_l.legend(
            h_bar + h_line, l_bar + l_line,
            loc='lower right',
            fontsize=JOURNAL_RC['legend.fontsize'],
            framealpha=0.88,
            edgecolor='#cccccc'
        )

        ax_l.set_title(title,
                       fontsize=JOURNAL_RC['axes.titlesize'],
                       pad=8)

    # ── 保存 ────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'fno_combined_metrics.png')
    plt.savefig(save_path, **SAVE_KW)
    plt.close()
    print(f"  → 保存: {save_path}")


# ── 内部辅助：柱顶标注 ──────────────────────────────────────
def _bar_labels(ax, bars, fmt='{:.4f}', fontsize=9, color='black'):
    """在每根柱子顶部添加数值文字。"""
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            fmt.format(h),
            ha='center', va='bottom',
            fontsize=fontsize,
            color=color,
            fontweight='normal'
        )


# ── 内部辅助：折线数值标注 ──────────────────────────────────
def _line_labels(ax, xs, ys, fmt='{:.4f}', fontsize=9,
                 color='black', dy=0.004):
    """在每个折线数据点旁标注数值，dy 为纵向偏移量。"""
    for xi, yi in zip(xs, ys):
        ax.text(
            xi, yi + dy,
            fmt.format(yi),
            ha='center', va='bottom',
            fontsize=fontsize,
            color=color
        )


def _r2_label_offset(r2_vals):
    """
    根据 R² 值域自动计算标注偏移量，
    确保标注不超出右轴上限。
    """
    span = max(r2_vals) - min(r2_vals)
    return max(span * 0.06, 5e-4)


# ── 内部辅助：自动扩展 Y 轴范围 ─────────────────────────────
def _expand_ylim(ax, values, margin=0.18, lower_zero=True):
    """
    将 Y 轴上限扩展，为柱顶 / 折点数值标注留出空间。
    lower_zero=True 时下限强制为 0（适用于 RMSE）。
    """
    vmax = max(values)
    vmin = min(values)
    span = vmax - vmin if vmax != vmin else abs(vmax) * 0.1 + 1e-8
    if lower_zero:
        ax.set_ylim(bottom=0, top=vmax + span * margin)
    else:
        pad = span * margin
        ax.set_ylim(bottom=vmin - pad * 0.3,
                    top=vmax   + pad)


# ==========================================
# 12. 汇总对比
# ==========================================
def print_summary(vae_metrics, fno1_metrics, fno2_metrics):
    print("\n" + "=" * 60)
    print("  三模型汇总对比 (100 样本测试集)")
    print("=" * 60)
    print(f"{'模型':<14} {'R²':>8} {'NRMSE':>12} {'MAE':>12}")
    print("-" * 50)
    rows = [
        ('VAE  (K→K)',   vae_metrics),
        ('FNO-1 (K→H)', fno1_metrics),
        ('FNO-2 (K→C)', fno2_metrics),
    ]
    for name, m in rows:
        print(f"{name:<14} "
              f"{m['overall_r2']:>8.4f} "
              f"{m['overall_nrmse']:>12.6f} "
              f"{m['overall_mae']:>12.6f}")
    if 'ssim' in vae_metrics:
        print(f"\n  VAE SSIM: {vae_metrics['ssim']:.4f}")
    # 打印推理时间
    for name, m in rows:
        if 'infer_total_s' in m:
            print(f"  {name} 推理: 总计 {m['infer_total_s']:.4f} s | "
                  f"单样本 {m['infer_per_sample_ms']:.4f} ms")
    print("=" * 60)


# ==========================================
# 13. 主程序
# ==========================================
def main():
    print("=" * 60)
    print("  代理模型测试集评估")
    print(f"  Device: {DEVICE} | Seed: {SEED}")
    print("=" * 60)

    k_test, h_test, c_test = load_test_data()

    vae_metrics  = evaluate_vae(k_test)
    fno1_metrics = evaluate_fno1(k_test, h_test)
    fno2_metrics = evaluate_fno2(k_test, c_test)

    print_summary(vae_metrics, fno1_metrics, fno2_metrics)

    print("\n[绘图] FNO 联合逐步指标图 ...")
    plot_fno_combined_metrics(
        fno1_metrics, fno2_metrics,
        save_dir='.'       # 输出到当前目录；如需放入子目录可修改
    )


if __name__ == "__main__":
    main()