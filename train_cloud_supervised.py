import os
import math
import h5py
import tifffile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import matplotlib

# 使用弹窗显示
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
BASE_H5_DIR = r"E:\001\CMA\CMA\data\result\2025"
BASE_TIF_DIR = r"E:\001\CMA\CMA\data\mask\2025"
CHANNELS = ['CH1064', 'PDR532', 'CH532']
USE_LOG10 = True
CLIP_MIN = 1e-6
NORM_PER_CHANNEL = True

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4  # 稍微增加批次大小
EPOCHS = 20  # 增加训练轮数
LEARNING_RATE = 1e-4  # 降低学习率
WEIGHT_DECAY = 1e-5
POS_WEIGHT = 10.0  # 调整正样本权重

OUTPUT_DIR = os.path.join(os.getcwd(), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_channels(channel_arrays: List[np.ndarray]) -> np.ndarray:
    safe_arrays = []
    for arr in channel_arrays:
        arr_safe = np.where(arr > 0, arr, CLIP_MIN)
        if USE_LOG10:
            arr_safe = np.log10(arr_safe)
        safe_arrays.append(arr_safe)
    x = np.stack(safe_arrays, axis=0)  # (C, T, H)
    if NORM_PER_CHANNEL:
        for c in range(x.shape[0]):
            mean = np.mean(x[c])
            std = np.std(x[c]) + 1e-8
            x[c] = (x[c] - mean) / std
    return x.astype(np.float32)


# -----------------------------
# Dataset with Data Augmentation
# -----------------------------
class LidarCloudDataset(Dataset):
    def __init__(self, day_list: List[str], preload: bool = True, augment: bool = False):
        self.samples: List[Tuple[str, str]] = []
        for day_folder in day_list:
            h5_day_path = os.path.join(BASE_H5_DIR, day_folder)
            tif_day_path = os.path.join(BASE_TIF_DIR, day_folder)
            if not os.path.isdir(h5_day_path) or not os.path.isdir(tif_day_path):
                continue
            for file in sorted(os.listdir(h5_day_path)):
                if file.endswith('.h5'):
                    h5_path = os.path.join(h5_day_path, file)
                    tif_path = os.path.join(tif_day_path, file.replace('.h5', '.tif'))
                    if os.path.exists(tif_path):
                        self.samples.append((h5_path, tif_path))

        self.preload = preload
        self.augment = augment
        if preload:
            print(f"Preloading {len(self.samples)} samples into memory...")
            self.data_cache = []
            for h5_path, tif_path in tqdm(self.samples, desc="Loading data"):
                x, label = self._load_sample(h5_path, tif_path)
                self.data_cache.append((x, label))
            print("Data preloading finished!")

    def _load_sample(self, h5_path, tif_path):
        with h5py.File(h5_path, 'r') as f:
            data = [f[ch][:] for ch in CHANNELS if ch in f]
        x = preprocess_channels(data)
        x = np.transpose(x, (0, 2, 1))  # (C, H, W)

        # 读取TIF并进行上下翻转
        label = tifffile.imread(tif_path).astype(np.float32)
        label = np.flipud(label).copy()  # 上下翻转
        return x, label

    def _augment(self, x, y):
        # 随机水平翻转
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=2).copy()
            y = np.flip(y, axis=1).copy()

        # 随机垂直翻转
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=1).copy()
            y = np.flip(y, axis=0).copy()

        # 随机亮度调整
        if np.random.rand() > 0.5:
            factor = 0.8 + 0.4 * np.random.rand()
            x = x * factor

        return x, y

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.preload and hasattr(self, 'data_cache'):
            x, y = self.data_cache[idx]
        else:
            h5_path, tif_path = self.samples[idx]
            x, y = self._load_sample(h5_path, tif_path)

        # 数据增强
        if self.augment:
            x, y = self._augment(x, y)

        return x, y


# -----------------------------
# collate_fn
# -----------------------------
def collate_fn(batch):
    max_w = max(x.shape[2] for x, _ in batch)
    batch_x, batch_y = [], []
    for x, y in batch:
        pad_w = max_w - x.shape[2]
        if pad_w > 0:
            x = np.pad(x, ((0, 0), (0, 0), (0, pad_w)), mode='constant')
            y = np.pad(y, ((0, 0), (0, pad_w)), mode='constant')
        batch_x.append(torch.tensor(x, dtype=torch.float32))
        batch_y.append(torch.tensor(y.copy(), dtype=torch.float32))
    return torch.stack(batch_x), torch.stack(batch_y)


# -----------------------------
# Enhanced Model
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.dconv_down1 = DoubleConv(in_ch, 32)  # 增加通道数
        self.pool1 = nn.MaxPool2d(2)
        self.dconv_down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.dconv_down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.dconv_down4 = DoubleConv(128, 256)  # 增加一层
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dconv_up4 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dconv_up3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dconv_up2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dconv_up1 = DoubleConv(64, 32)
        self.out_conv = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(self.pool1(conv1))
        conv3 = self.dconv_down3(self.pool2(conv2))
        conv4 = self.dconv_down4(self.pool3(conv3))
        conv5 = self.bottleneck(self.pool4(conv4))

        x = self.up4(conv5)
        x = F.interpolate(x, size=conv4.shape[2:], mode='bilinear', align_corners=False)
        x = self.dconv_up4(torch.cat([x, conv4], 1))

        x = self.up3(x)
        x = F.interpolate(x, size=conv3.shape[2:], mode='bilinear', align_corners=False)
        x = self.dconv_up3(torch.cat([x, conv3], 1))

        x = self.up2(x)
        x = F.interpolate(x, size=conv2.shape[2:], mode='bilinear', align_corners=False)
        x = self.dconv_up2(torch.cat([x, conv2], 1))

        x = self.up1(x)
        x = F.interpolate(x, size=conv1.shape[2:], mode='bilinear', align_corners=False)
        x = self.dconv_up1(torch.cat([x, conv1], 1))

        return self.out_conv(x)


# -----------------------------
# Enhanced Loss Functions
# -----------------------------
def dice_loss(pred, target, smooth=1e-6):
    pred_prob = torch.sigmoid(pred)
    intersection = (pred_prob * target).sum()
    return 1 - (2 * intersection + smooth) / (pred_prob.sum() + target.sum() + smooth)


def focal_loss(pred, target, alpha=0.25, gamma=2):
    pred_prob = torch.sigmoid(pred)
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal_loss = alpha * (1 - pt) ** gamma * bce
    return focal_loss.mean()


def combined_loss(pred, target):
    pos_w = torch.tensor([POS_WEIGHT], device=pred.device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)(pred, target)
    dice = dice_loss(pred, target)
    focal = focal_loss(pred, target)
    return bce + dice + 0.5 * focal  # 组合多种损失


# -----------------------------
# Training with Validation
# -----------------------------
def train_one_epoch(model, train_loader, optimizer):
    model.train()
    total_loss = 0.0
    for X, y in tqdm(train_loader, desc="Training"):
        X, y = X.to(DEVICE), y.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(X)
        loss = combined_loss(logits, y)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(train_loader))


@torch.no_grad()
def validate(model, val_loader):
    model.eval()
    total_loss = 0.0
    for X, y in val_loader:
        X, y = X.to(DEVICE), y.to(DEVICE).unsqueeze(1)
        logits = model(X)
        loss = combined_loss(logits, y)
        total_loss += loss.item()
    return total_loss / max(1, len(val_loader))


# -----------------------------
# Enhanced Visualization
# -----------------------------
@torch.no_grad()
def visualize_and_save(model, dataset, epoch=0):
    if len(dataset) == 0:
        return

    model.eval()
    for idx in range(min(5, len(dataset))):  # 只可视化前5个样本
        X, y = dataset[idx]
        X_tensor = torch.tensor(X).unsqueeze(0).to(DEVICE)
        logits = model(X_tensor)
        pred_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        pred = (pred_prob > 0.5).astype(np.uint8)
        gt = y.astype(np.uint8)

        # 计算指标
        intersection = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()
        iou = intersection / (union + 1e-8) if union > 0 else 0

        # 保存预测结果
        out_path = os.path.join(OUTPUT_DIR, f"pred_epoch{epoch:03d}_{idx:03d}.tif")
        tifffile.imwrite(out_path, pred)

        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Ground Truth
        axes[0, 0].imshow(gt, cmap='gray', origin='lower', aspect='auto')
        axes[0, 0].set_title("Ground Truth")

        # Prediction
        axes[0, 1].imshow(pred, cmap='gray', origin='lower', aspect='auto')
        axes[0, 1].set_title("Prediction")

        # Probability map
        im = axes[1, 0].imshow(pred_prob, cmap='jet', origin='lower', aspect='auto', vmin=0, vmax=1)
        axes[1, 0].set_title("Probability Map")
        plt.colorbar(im, ax=axes[1, 0])

        # Overlay
        overlay = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.float32)
        overlay[gt == 1] = [0, 1, 0]  # 绿色表示真实标签
        overlay[pred == 1] = [1, 0, 0]  # 红色表示预测
        overlay[(gt == 1) & (pred == 1)] = [1, 1, 0]  # 黄色表示重叠部分
        axes[1, 1].imshow(overlay, origin='lower', aspect='auto')
        axes[1, 1].set_title(f"Overlay (IoU: {iou:.3f})")

        plt.suptitle(f"Epoch {epoch}, Sample {idx}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"vis_epoch{epoch:03d}_{idx:03d}.png"))
        plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    train_days = [f"{i:04d}" for i in range(101, 125)]
    val_days = [f"{i:04d}" for i in range(126, 127)]

    print("Creating dataset...")
    train_dataset = LidarCloudDataset(train_days, preload=True, augment=True)  # 启用数据增强
    val_dataset = LidarCloudDataset(val_days, preload=False, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn)

    print(f"Device: {DEVICE} | Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    model = UNet(in_ch=len(CHANNELS)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)

    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE / 10
    )

    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)

        scheduler.step()

        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f"New best model saved with val_loss={val_loss:.4f}")

        # 每5个epoch可视化一次
        if epoch % 5 == 0 or epoch == EPOCHS:
            print("Generating predictions...")
            visualize_and_save(model, val_dataset, epoch)

    print("Training finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Predictions saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()