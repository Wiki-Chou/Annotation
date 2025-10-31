import os
# 核心显存优化：解决碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 明确使用双GPU

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
# Config（调整学习率和训练轮数）
# -----------------------------
BASE_H5_DIR = r"/kaggle/input/dataset/h5"  # 可根据本地路径修改
BASE_TIF_DIR = r"/kaggle/input/dataset/tif"
CHANNELS = ['CH1064', 'PDR532', 'CH532', 'PDR355', 'CH355']  # 按你的数据通道调整
USE_LOG10 = True
CLIP_MIN = 1e-6
NORM_PER_CHANNEL = True

# 双GPU配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU_NUM = torch.cuda.device_count()
print(f"Detected {GPU_NUM} GPU(s). Using {min(GPU_NUM, 2)} GPU(s) for training.")

# 调整训练超参数
BATCH_SIZE = 4  
ACCUMULATION_STEPS = 4  # 启用梯度累积
EPOCHS = 50  # 增加训练轮数至50
LEARNING_RATE = 3e-4  # 降低学习率至3e-4，更稳定
WEIGHT_DECAY = 1e-4
POS_WEIGHT = 10.0
EARLY_STOPPING_PATIENCE = 10  # 早停：验证loss连续10个epoch未改善则停止训练

OUTPUT_DIR = os.path.join(os.getcwd(), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Preprocessing（保留原逻辑）
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
# Dataset（确保标签只包含0和1）
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
        # 1. 加载H5输入并预处理
        with h5py.File(h5_path, 'r') as f:
            data = [f[ch][:] for ch in CHANNELS if ch in f]
        x = preprocess_channels(data)
        x = np.transpose(x, (0, 2, 1))  # (C, T, H) → (C, H, W)

        # 2. 读取TIF并确保标签只包含0和1（核心修改）
        label = tifffile.imread(tif_path).astype(np.float32)
        
        # 关键：强制标签二值化，只保留0和1
        label = np.where(label > 0.5, 1.0, 0.0)  # 大于0.5的为1，否则为0
        
        # 转置标签确保与输入尺寸一致
        if label.shape[0] == x.shape[2] and label.shape[1] == x.shape[1]:
            label = np.transpose(label)
        label = np.flipud(label).copy()
        
        # 尺寸不匹配时调整，并再次确保二值化
        if label.shape != x.shape[1:]:
            from skimage.transform import resize
            label = resize(label, x.shape[1:], order=0, preserve_range=True)
            label = np.where(label > 0.5, 1.0, 0.0)  # 调整后再次二值化
        
        # 最终校验：确保没有其他值
        unique_vals = np.unique(label)
        if not np.array_equal(unique_vals, [0.0]) and not np.array_equal(unique_vals, [1.0]) and not np.array_equal(np.sort(unique_vals), [0.0, 1.0]):
            print(f"Warning: Unexpected values in label {tif_path}: {unique_vals}")
            label = np.where(label > 0.5, 1.0, 0.0)  # 强制清理
        
        return x, label

    def _augment(self, x, y):
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
# collate_fn（保持原逻辑）
# -----------------------------
def collate_fn(batch):
    max_h = max(x.shape[1] for x, _ in batch)
    max_w = max(x.shape[2] for x, _ in batch)
    batch_x, batch_y = [], []
    for x, y in batch:
        pad_h = max_h - x.shape[1]
        pad_w = max_w - x.shape[2]
        if pad_h > 0 or pad_w > 0:
            x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
            y = np.pad(y, ((0, pad_h), (0, pad_w)), mode='constant')
        batch_x.append(torch.tensor(x, dtype=torch.float32))
        batch_y.append(torch.tensor(y.copy(), dtype=torch.float32))
    return torch.stack(batch_x), torch.stack(batch_y)


# -----------------------------
# Model（完全保留原结构）
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
# Loss Functions（保留原逻辑）
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
# Training（启用梯度累积）
# -----------------------------
def train_one_epoch(model, train_loader, optimizer):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()  # 初始清零梯度

    for idx, (X, y) in enumerate(tqdm(train_loader, desc="Training")):
        X, y = X.to(DEVICE), y.to(DEVICE).unsqueeze(1)
        logits = model(X)
        loss = combined_loss(logits, y)
        
        # 应用梯度累积（核心修改）
        loss = loss / ACCUMULATION_STEPS  # 损失归一化
        loss.backward()  # 累积梯度
        
        # 每累积指定批次后更新参数
        if (idx + 1) % ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()  # 重置梯度
        
        total_loss += loss.item() * ACCUMULATION_STEPS  # 还原真实损失值
    return total_loss / max(1, len(train_loader))


# -----------------------------
# Validation（增加分布统计）
# -----------------------------
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
        
        pred_prob = torch.sigmoid(logits)
        preds = (pred_prob > 0.5).cpu().numpy().reshape(-1)
        targets = y.cpu().numpy().reshape(-1)
        all_preds.extend(preds)
        all_targets.extend(targets)
    
    # 统计标签和预测分布，帮助调试指标问题
    target_pos = np.sum(all_targets)
    target_neg = len(all_targets) - target_pos
    pred_pos = np.sum(all_preds)
    pred_neg = len(all_preds) - pred_pos
    print(f"\nValidation distribution:")
    print(f"Labels - 1s: {target_pos}, 0s: {target_neg} (positive ratio: {target_pos/len(all_targets):.3f})")
    print(f"Preds  - 1s: {pred_pos}, 0s: {pred_neg} (positive ratio: {pred_pos/len(all_preds):.3f})")
    
    # 计算评估指标
    precision = precision_score(all_targets, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='binary', zero_division=0)
    accuracy = accuracy_score(all_targets, all_preds)
    
    return total_loss / max(1, len(val_loader)), precision, recall, f1, accuracy


# -----------------------------
# Visualization（保留原逻辑）
# -----------------------------
@torch.no_grad()
def visualize_and_save(model, dataset, epoch=0, prefix=''):
    if len(dataset) == 0:
        return

    model.eval()
    raw_model = model.module if hasattr(model, 'module') else model
    
    for idx in range(min(5, len(dataset))):
        X, y = dataset[idx]
        X_tensor = torch.tensor(X).unsqueeze(0).to(DEVICE)
        logits = raw_model(X_tensor)
        pred_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        pred = (pred_prob > 0.5).astype(np.uint8)
        gt = y.astype(np.uint8)

        plt.figure(figsize=(12, 6))
        # 使用CH1064作为背景
        ch1064_idx = CHANNELS.index('CH1064')
        ch1064_data = X[ch1064_idx]
        plt.imshow(ch1064_data, cmap='jet', origin='lower', aspect='auto')
        plt.colorbar(label='CH1064 (normalized)')
        from skimage import measure
        
        # 绘制真值轮廓（显眼的绿色实线）
        gt_contours = measure.find_contours(gt, 0.5)
        for contour in gt_contours:
            plt.plot(contour[:, 1], contour[:, 0], color='lime', linewidth=3, 
                    linestyle='-', label='Ground Truth')
        
        # 绘制预测轮廓（品红色虚线，不在jet colorbar中）
        pred_contours = measure.find_contours(pred, 0.5)
        for contour in pred_contours:
            plt.plot(contour[:, 1], contour[:, 0], color='magenta', linewidth=2.5, 
                    linestyle='--', label='Prediction')
        
        # 去重图例（只显示一次）
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.xlabel('Time')
        plt.ylabel('Height')
        plt.title(f'Cloud Detection Result (Sample {idx}) Epoch {epoch}')
        plt.tight_layout()
        
        # 根据prefix保存文件
        if prefix:
            filename = f"{prefix}_cloud_pred_epoch{epoch:03d}_{idx:03d}.png"
        else:
            filename = f"cloud_pred_epoch{epoch:03d}_{idx:03d}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()


# -----------------------------
# Main（主函数）
# -----------------------------
def main():
    val_days = ['0101', '0121', '0131']
    train_days = [f"{i:04d}" for i in range(101, 132) if f"{i:04d}" not in val_days]

    print("Creating dataset...")
    train_dataset = LidarCloudDataset(train_days, preload=True, augment=True)
    val_dataset = LidarCloudDataset(val_days, preload=False, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn, num_workers=2)

    print(f"Device: {DEVICE} | Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 双GPU模型包装
    model = UNet(in_ch=len(CHANNELS))
    if GPU_NUM >= 2:
        model = nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
        min_lr=1e-6, verbose=True
    )

    best_val_loss = float('inf')
    best_metrics = None
    best_epoch = 0  # 记录最佳epoch
    patience_counter = 0  # 早停计数器
    
    # 记录训练和验证loss用于绘制曲线
    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, precision, recall, f1, accuracy = validate(model, val_loader)
        
        # 记录loss历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={current_lr:.6f}")
        print(f"Metrics: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Accuracy={accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch  # 记录最佳epoch
            patience_counter = 0  # 重置早停计数器
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            }
            save_model = model.module if hasattr(model, 'module') else model
            torch.save(save_model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f"New best model saved with val_loss={val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s). Best val_loss: {best_val_loss:.4f} at epoch {best_epoch}")
            
            # 早停检查
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered! No improvement for {EARLY_STOPPING_PATIENCE} epochs.")
                print(f"Best model was at epoch {best_epoch} with val_loss={best_val_loss:.4f}")
                break

        if epoch % 10 == 0 or epoch == EPOCHS:
            print(f"Generating predictions at epoch {epoch}...")
            visualize_and_save(model, val_dataset, epoch)

    # 保存最佳指标
    metrics_file = os.path.join(OUTPUT_DIR, 'best_model_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Best Model Evaluation Metrics:\n")
        f.write(f"Precision: {best_metrics['precision']:.4f}\n")
        f.write(f"Recall: {best_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {best_metrics['f1']:.4f}\n")
        f.write(f"Accuracy: {best_metrics['accuracy']:.4f}\n")

    # 绘制训练和验证loss曲线
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, EPOCHS + 1)
    plt.plot(epochs_range, train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.plot(epochs_range, val_losses, 'r-', linewidth=2, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Curves', fontsize=14)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_curve_path = os.path.join(OUTPUT_DIR, 'loss_curves.png')
    plt.savefig(loss_curve_path, dpi=150)
    plt.close()
    print(f"Loss curves saved to: {loss_curve_path}")
    
    print("\n" + "="*60)
    print("Training finished.")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Final metrics saved to: {metrics_file}")
    print(f"Best epoch predictions saved with prefix 'best_'")
    print("="*60)


if __name__ == '__main__':
    main()
