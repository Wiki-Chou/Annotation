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
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# -----------------------------
# Config（双GPU+核心参数保留原脚本逻辑）
# -----------------------------
BASE_H5_DIR = r"/kaggle/input/h5-and-tif/h5"  # 可根据本地路径修改
BASE_TIF_DIR = r"/kaggle/input/h5-and-tif/tif"
CHANNELS = ['CH1064', 'PDR532', 'CH532', 'PDR355', 'CH355']  # 按你的数据通道调整
USE_LOG10 = True
CLIP_MIN = 1e-6
NORM_PER_CHANNEL = True

# 双GPU配置：自动检测并使用2张GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU_NUM = torch.cuda.device_count()
print(f"Detected {GPU_NUM} GPU(s). Using {min(GPU_NUM, 2)} GPU(s) for training.")

# 调整训练超参数
BATCH_SIZE = 8  # 增大批次大小以提高训练稳定性
EPOCHS = 15    # 增加训练轮数，让模型有更充分的训练时间
LEARNING_RATE = 5e-4  # 稍微增大学习率
WEIGHT_DECAY = 1e-4   # 增大权重衰减以增加正则化
POS_WEIGHT = 10.0

OUTPUT_DIR = os.path.join(os.getcwd(), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Preprocessing（完全保留原脚本逻辑）
# -----------------------------
def preprocess_channels(channel_arrays: List[np.ndarray]) -> np.ndarray:
    safe_arrays = []
    for arr in channel_arrays:
        arr_safe = np.where(arr > 0, arr, CLIP_MIN)
        if USE_LOG10:
            arr_safe = np.log10(arr_safe)
        # 0-1归一化，每通道独立
        arr_min = np.min(arr_safe)
        arr_max = np.max(arr_safe)
        arr_norm = (arr_safe - arr_min) / (arr_max - arr_min + 1e-8)
        safe_arrays.append(arr_norm)
    x = np.stack(safe_arrays, axis=0)  # (C, T, H)
    return x.astype(np.float32)


# -----------------------------
# Dataset（核心优化：读取TIF时直接匹配输入尺寸）
# -----------------------------
class LidarCloudDataset(Dataset):
    def __init__(self, day_list: List[str], preload: bool = True, augment: bool = False):
        self.samples: List[Tuple[str, str]] = []
        # 保留原脚本样本筛选逻辑
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
        # 1. 加载H5输入并预处理（完全保留原脚本）
        with h5py.File(h5_path, 'r') as f:
            data = [f[ch][:] for ch in CHANNELS if ch in f]
        x = preprocess_channels(data)
        x = np.transpose(x, (0, 2, 1))  # 原脚本：(C, T, H) → (C, H, W)，记输入尺寸为(H, W)

        # 2. 读取TIF并直接匹配输入尺寸（核心优化）
        label = tifffile.imread(tif_path).astype(np.float32)
        # 关键：转置标签（原标签可能是(W, H)），确保与输入的(H, W)一致
        if label.shape[0] == x.shape[2] and label.shape[1] == x.shape[1]:
            label = np.transpose(label)  # 当标签是(W, H)时，转置为(H, W)
        # 保留原脚本的上下翻转逻辑
        label = np.flipud(label).copy()
        # 最终校验：若仍不匹配，用原脚本逻辑resize（避免极端情况）
        if label.shape != x.shape[1:]:
            from skimage.transform import resize
            label = resize(label, x.shape[1:], order=0, preserve_range=True)
        
        return x, label

    def _augment(self, x, y):
        # 完全保留原脚本数据增强逻辑
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=2).copy()
            y = np.flip(y, axis=1).copy()
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=1).copy()
            y = np.flip(y, axis=0).copy()
        if np.random.rand() > 0.5:
            factor = 0.8 + 0.4 * np.random.rand()
            x = x * factor
        return x, y

    def __len__(self):
        return len(self.data_cache) if self.preload else len(self.samples)

    def __getitem__(self, idx: int):
        if self.preload and hasattr(self, 'data_cache'):
            x, y = self.data_cache[idx]
        else:
            h5_path, tif_path = self.samples[idx]
            x, y = self._load_sample(h5_path, tif_path)
        if self.augment:
            x, y = self._augment(x, y)
        return x, y


# -----------------------------
# collate_fn（优化：同时处理H和W维度，适配所有样本尺寸）
# -----------------------------
def collate_fn(batch):
    # 原脚本只处理W，新增H维度处理，确保批次内所有样本尺寸一致
    max_h = max(x.shape[1] for x, _ in batch)  # 输入x形状：(C, H, W)，H在索引1
    max_w = max(x.shape[2] for x, _ in batch)  # W在索引2
    batch_x, batch_y = [], []
    for x, y in batch:
        # 计算需要补充的高度和宽度
        pad_h = max_h - x.shape[1]
        pad_w = max_w - x.shape[2]
        # 对x和y同时padding，确保尺寸匹配
        if pad_h > 0 or pad_w > 0:
            x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
            y = np.pad(y, ((0, pad_h), (0, pad_w)), mode='constant')
        batch_x.append(torch.tensor(x, dtype=torch.float32))
        batch_y.append(torch.tensor(y.copy(), dtype=torch.float32))
    return torch.stack(batch_x), torch.stack(batch_y)


# -----------------------------
# Model（完全保留原脚本的UNet结构）
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
        self.dconv_down1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.dconv_down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.dconv_down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.dconv_down4 = DoubleConv(128, 256)
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
# Loss Functions（完全保留原脚本逻辑）
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
    return bce + dice + 0.5 * focal


# -----------------------------
# Training with Validation（适配双GPU）
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

        # 保留原脚本的梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(train_loader))


@torch.no_grad()
def validate(model, val_loader):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for X, y in val_loader:
        X, y = X.to(DEVICE), y.to(DEVICE).unsqueeze(1)
        logits = model(X)
        loss = combined_loss(logits, y)
        total_loss += loss.item()
        
        # 收集预测和真实标签用于计算指标
        pred_prob = torch.sigmoid(logits)
        preds = (pred_prob > 0.5).cpu().numpy().reshape(-1)
        targets = y.cpu().numpy().reshape(-1)
        all_preds.extend(preds)
        all_targets.extend(targets)
    
    # 计算评估指标
    precision = precision_score(all_targets, all_preds, average='micro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='micro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='micro', zero_division=0)
    accuracy = accuracy_score(all_targets, all_preds)
    
    return total_loss / max(1, len(val_loader)), precision, recall, f1, accuracy


# -----------------------------
# Visualization（保留原脚本逻辑，适配双GPU模型）
# -----------------------------
@torch.no_grad()
def visualize_and_save(model, dataset, epoch=0):
    if len(dataset) == 0:
        return

    model.eval()
    # 双GPU适配：获取原始模型（DataParallel包装后需用.module访问）
    raw_model = model.module if hasattr(model, 'module') else model
    
    for idx in range(min(5, len(dataset))):
        X, y = dataset[idx]
        X_tensor = torch.tensor(X).unsqueeze(0).to(DEVICE)
        logits = raw_model(X_tensor)
        pred_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        pred = (pred_prob > 0.5).astype(np.uint8)
        gt = y.astype(np.uint8)

        # 可视化：只输出PNG，不输出tif
        plt.figure(figsize=(12, 6))
        ch532_idx = CHANNELS.index('CH532')
        ch532_data = X[ch532_idx]
        plt.imshow(ch532_data, cmap='jet', origin='lower', aspect='auto')
        plt.colorbar(label='CH532 (normalized)')
        # 预测云层边界（轮廓）
        from skimage import measure
        contours = measure.find_contours(pred, 0.5)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2, label='Predicted Cloud')
        # 真实云层边界（可选）
        # gt_contours = measure.find_contours(gt, 0.5)
        # for contour in gt_contours:
        #     plt.plot(contour[:, 1], contour[:, 0], color='lime', linewidth=2, linestyle='--', label='GT Cloud')
        plt.xlabel('Time')
        plt.ylabel('Height')
        plt.title(f'Cloud Detection Result (Sample {idx}) Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"cloud_pred_epoch{epoch:03d}_{idx:03d}.png"))
        plt.close()


# -----------------------------
# Main（保留原脚本逻辑，适配双GPU）
# -----------------------------
def main():
    # 保留原脚本的训练/验证集划分逻辑
    train_days = [f"{i:04d}" for i in range(101, 126)]
    val_days = [f"{i:04d}" for i in range(128, 131)]

    print("Creating dataset...")
    train_dataset = LidarCloudDataset(train_days, preload=True, augment=True)
    val_dataset = LidarCloudDataset(val_days, preload=False, augment=False)

    # 适配双GPU的数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn, num_workers=2)

    print(f"Device: {DEVICE} | Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 双GPU模型包装
    model = UNet(in_ch=len(CHANNELS))
    if GPU_NUM >= 2:
        model = nn.DataParallel(model, device_ids=[0, 1])  # 使用前2张GPU
    model = model.to(DEVICE)

    # 保留原脚本的优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    # 使用更平缓的学习率衰减策略
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
        min_lr=1e-6, verbose=True
    )

    best_val_loss = float('inf')
    best_metrics = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, precision, recall, f1, accuracy = validate(model, val_loader)

        # 根据验证损失调整学习率
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={current_lr:.6f}")
        print(f"Metrics: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Accuracy={accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            }
            save_model = model.module if hasattr(model, 'module') else model
            torch.save(save_model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f"New best model saved with val_loss={val_loss:.4f}")

        # 每5个epoch和最后一个epoch可视化
        if epoch % 5 == 0 or epoch == EPOCHS:
            print(f"Generating predictions at epoch {epoch}...")
            visualize_and_save(model, val_dataset, epoch)

    # 保存最佳模型的评估指标到txt文件
    metrics_file = os.path.join(OUTPUT_DIR, 'best_model_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Best Model Evaluation Metrics:\n")
        f.write(f"Precision: {best_metrics['precision']:.4f}\n")
        f.write(f"Recall: {best_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {best_metrics['f1']:.4f}\n")
        f.write(f"Accuracy: {best_metrics['accuracy']:.4f}\n")

    print("Training finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final metrics saved to: {metrics_file}")
    print(f"Predictions saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
