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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# -----------------------------
# Config（双GPU+核心参数保留原脚本逻辑）
# -----------------------------
BASE_H5_DIR = r"/kaggle/input/dataset/h5"  # 可根据本地路径修改
BASE_TIF_DIR = r"/kaggle/input/dataset/tif"
CHANNELS = ['CH1064', 'PDR532', 'CH532', 'PDR355', 'CH355']  # 按你的数据通道调整
USE_LOG10 = True
CLIP_MIN = 1e-6
NORM_PER_CHANNEL = True

# 双GPU配置：自动检测并使用2张GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU_NUM = torch.cuda.device_count()
print(f"Detected {GPU_NUM} GPU(s). Using {min(GPU_NUM, 2)} GPU(s) for training.")

# 调整训练超参数
BATCH_SIZE = 4     # 保持小批次以提高稳定性
EPOCHS = 50        # 保持训练轮数不变
LEARNING_RATE = 1e-3  # 保持学习率不变
WEIGHT_DECAY = 5e-3   # 增加weight decay以加强正则化
POS_WEIGHT = 2.0      # 降低正样本权重以提高precision

# 梯度累积步数（相当于模拟更大的批次）
GRADIENT_ACCUMULATION_STEPS = 4  # 累积4步，相当于之前的batch_size=16的效果

# 混合精度训练设置
USE_AMP = True  # 启用自动混合精度训练

# warmup和学习率调度参数
WARMUP_EPOCHS = 5  # warmup阶段的epoch数
MIN_LR = 1e-6      # 最小学习率
T_MAX = 45         # 余弦退火周期（EPOCHS - WARMUP_EPOCHS）

# Focal Loss参数
FOCAL_GAMMA = 5.0  # 增加gamma以更强调困难样本
FOCAL_ALPHA = 0.10 # 进一步降低alpha以减少正样本偏好

OUTPUT_DIR = os.path.join(os.getcwd(), 'outputs')1
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Preprocessing（完全保留原脚本逻辑）
# -----------------------------
def preprocess_channels(channel_arrays: List[np.ndarray]) -> np.ndarray:
    safe_arrays = []
    for arr in channel_arrays:
        # 先把非有限值（NaN/Inf）替换为一个小正数，避免后续比较/取 log 出现警告或错误
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=CLIP_MIN, posinf=CLIP_MIN, neginf=CLIP_MIN)

        # 将小于等于0的值替换为 CLIP_MIN，保证 log10 和归一化安全
        arr_safe = np.where(arr > 0, arr, CLIP_MIN)

        if USE_LOG10:
            # log10 需要正值；此处 arr_safe 已确保为正
            arr_safe = np.log10(arr_safe)

        # 0-1归一化，每通道独立（防止除以0，加小常数）
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
            total_pos = 0
            total_pix = 0
            for h5_path, tif_path in tqdm(self.samples, desc="Loading data"):
                x, label = self._load_sample(h5_path, tif_path)
                self.data_cache.append((x, label))
                total_pos += int(label.sum())
                total_pix += int(label.size)
            print("Data preloading finished!")
            # 打印总体的正样本比例，帮助诊断类别不平衡
            if total_pix > 0:
                pos_ratio = total_pos / total_pix
                print(f"Preload summary: total_positive_pixels={total_pos}, total_pixels={total_pix}, positive_ratio={pos_ratio:.6f}")

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

        # 把异常值（NaN/Inf）设为0，且将标签二值化：任何大于0的像素视为云（1），否则为0
        # 这样可以避免多类标签（例如出现2）干扰二分类评估
        label = np.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0)
        label_bin = (label > 0).astype(np.uint8)

        return x, label_bin

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
        # 可学习的通道缩放器：使用 depthwise 1x1 conv 为每个输入通道学习独立的 scale 和 bias
        # 这比单纯的 nn.Parameter 更灵活（包含偏置），且在 DataParallel 下行为一致
        self.channel_scaler = nn.Conv2d(in_ch, in_ch, kernel_size=1, groups=in_ch, bias=True)
        # 初始化为恒等（scale=1, bias=0）
        with torch.no_grad():
            self.channel_scaler.weight.fill_(1.0)
            if self.channel_scaler.bias is not None:
                self.channel_scaler.bias.fill_(0.0)
        # 减小特征通道数以降低显存占用
        self.dconv_down1 = DoubleConv(in_ch, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.dconv_down2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.dconv_down3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        self.dconv_down4 = DoubleConv(64, 128)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)

        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dconv_up4 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dconv_up3 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dconv_up2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dconv_up1 = DoubleConv(32, 16)
        self.out_conv = nn.Conv2d(16, out_ch, 1)

    def forward(self, x):
        # 在进入 UNet 主体之前，先对每个通道做可学习缩放（scale + bias）
        x = self.channel_scaler(x)
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
# Visualization Helpers
# -----------------------------
class TrainingHistory:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.metrics = {
            'precision': [], 'recall': [], 
            'f1': [], 'accuracy': []
        }
        self.learning_rates = []
        
    def update(self, train_loss, val_loss, metrics, lr):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        for k, v in metrics.items():
            self.metrics[k].append(v)
        self.learning_rates.append(lr)
    
    def plot_training_curves(self, output_dir):
        """绘制训练过程曲线"""
        epochs = range(1, len(self.train_losses) + 1)
        
        # 创建2x2的子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Loss曲线
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Precision & Recall
        ax2.plot(epochs, self.metrics['precision'], 'g-', label='Precision')
        ax2.plot(epochs, self.metrics['recall'], 'b-', label='Recall')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_title('Precision and Recall')
        ax2.legend()
        ax2.grid(True)
        
        # 3. F1 & Accuracy
        ax3.plot(epochs, self.metrics['f1'], 'r-', label='F1')
        ax3.plot(epochs, self.metrics['accuracy'], 'y-', label='Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.set_title('F1 and Accuracy')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Learning Rate
        ax4.plot(epochs, self.learning_rates, 'k-')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.grid(True)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'))
        plt.close()

def plot_channel_importance(model, channels, output_dir):
    """绘制输入通道重要性分析图"""
    if hasattr(model, 'module'):
        model = model.module
    
    # 获取通道缩放器的权重和偏置
    with torch.no_grad():
        weights = model.channel_scaler.weight.squeeze().cpu().numpy()
        if model.channel_scaler.bias is not None:
            biases = model.channel_scaler.bias.cpu().numpy()
        else:
            biases = np.zeros_like(weights)
    
    # 计算每个通道的综合重要性（权重的绝对值 + 偏置的绝对值）
    importance = np.abs(weights) + np.abs(biases)
    
    # 绘制条形图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(channels, importance)
    plt.xlabel('Input Channels')
    plt.ylabel('Channel Importance')
    plt.title('Input Channel Importance Analysis')
    plt.xticks(rotation=45)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'channel_importance.png'))
    plt.close()

def plot_height_distribution(dataset, output_dir, title_prefix=''):
    """统计并绘制不同高度层的云层分布"""
    height_counts = np.zeros(dataset[0][1].shape[0])
    total_samples = len(dataset)
    
    for _, label in dataset:
        height_counts += np.mean(label > 0, axis=1)
    
    height_freq = height_counts / total_samples
    
    plt.figure(figsize=(8, 10))
    plt.barh(np.arange(len(height_freq)), height_freq)
    plt.ylabel('Height Level')
    plt.xlabel('Cloud Occurrence Frequency')
    plt.title(f'{title_prefix} Cloud Height Distribution')
    plt.grid(True, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title_prefix.lower()}_height_distribution.png'))
    plt.close()

def analyze_difficult_samples(model, dataset, output_dir, n_samples=5):
    """分析并展示最容易和最难预测的样本"""
    model.eval()
    sample_metrics = []
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            X, y = dataset[idx]
            X_tensor = torch.tensor(X).unsqueeze(0).to(DEVICE)
            logits = model(X_tensor)
            pred_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred = (pred_prob > 0.5).astype(np.uint8)
            
            # 计算此样本的准确率
            accuracy = np.mean(pred == y)
            sample_metrics.append((idx, accuracy))
    
    # 按准确率排序
    sample_metrics.sort(key=lambda x: x[1])
    
    # 创建一个包含最难和最易样本的图
    fig, axes = plt.subplots(2, n_samples, figsize=(20, 8))
    
    # 处理最难的样本
    for i, (idx, acc) in enumerate(sample_metrics[:n_samples]):
        X, y = dataset[idx]
        X_tensor = torch.tensor(X).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(X_tensor)
            pred_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred = (pred_prob > 0.5).astype(np.uint8)
        
        # 显示原始数据
        ch532_idx = CHANNELS.index('CH532')
        axes[0, i].imshow(X[ch532_idx], cmap='jet')
        axes[0, i].imshow(np.ma.masked_where(y == 0, y), 
                         cmap=matplotlib.colors.ListedColormap(['green']), alpha=0.3)
        axes[0, i].imshow(np.ma.masked_where(pred == 0, pred), 
                         cmap=matplotlib.colors.ListedColormap(['red']), alpha=0.3)
        axes[0, i].set_title(f'Hardest {i+1}\nAcc: {acc:.3f}')
        axes[0, i].axis('off')
    
    # 处理最容易的样本
    for i, (idx, acc) in enumerate(reversed(sample_metrics[-n_samples:])):
        X, y = dataset[idx]
        X_tensor = torch.tensor(X).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(X_tensor)
            pred_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred = (pred_prob > 0.5).astype(np.uint8)
        
        # 显示原始数据
        ch532_idx = CHANNELS.index('CH532')
        axes[1, i].imshow(X[ch532_idx], cmap='jet')
        axes[1, i].imshow(np.ma.masked_where(y == 0, y), 
                         cmap=matplotlib.colors.ListedColormap(['green']), alpha=0.3)
        axes[1, i].imshow(np.ma.masked_where(pred == 0, pred), 
                         cmap=matplotlib.colors.ListedColormap(['red']), alpha=0.3)
        axes[1, i].set_title(f'Easiest {i+1}\nAcc: {acc:.3f}')
        axes[1, i].axis('off')
    
    plt.suptitle('Analysis of Difficult vs Easy Samples\nGreen: Ground Truth, Red: Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_difficulty_analysis.png'))
    plt.close()

# -----------------------------
# Loss Functions（完全保留原脚本逻辑）
# -----------------------------
def dice_loss(pred, target, smooth=1e-6):
    pred_prob = torch.sigmoid(pred)
    intersection = (pred_prob * target).sum()
    return 1 - (2 * intersection + smooth) / (pred_prob.sum() + target.sum() + smooth)


def focal_loss(pred, target, alpha=None, gamma=None):
    # 使用全局设定的focal loss参数
    alpha = FOCAL_ALPHA if alpha is None else alpha
    gamma = FOCAL_GAMMA if gamma is None else gamma
    
    pred_prob = torch.sigmoid(pred)
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    
    # 对于负样本，使用1-alpha
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal_loss = alpha_t * (1 - pt) ** gamma * bce
    
    return focal_loss.mean()


def combined_loss(pred, target):
    pos_w = torch.tensor([POS_WEIGHT], device=pred.device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)(pred, target)
    dice = dice_loss(pred, target)
    focal = focal_loss(pred, target)
    # 增加BCE权重，减少dice权重以提高precision
    return 0.5 * bce + 0.2 * dice + 0.3 * focal


# -----------------------------
# Training with Validation（适配双GPU）
# -----------------------------
def train_one_epoch(model, train_loader, optimizer):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()  # 在epoch开始时清零梯度
    
    # 创建GradScaler用于AMP
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    for i, (X, y) in enumerate(tqdm(train_loader, desc="Training")):
        X, y = X.to(DEVICE), y.to(DEVICE).unsqueeze(1)
        
        # 使用AMP自动混合精度训练
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(X)
            loss = combined_loss(logits, y)
            # 根据梯度累积步数缩放损失
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            
        # 使用scaler处理反向传播
        scaler.scale(loss).backward()

        # 累积指定步数的梯度后更新参数
        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            # 保留原脚本的梯度裁剪（注意scaler）
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 使用scaler执行优化器步骤
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS  # 还原真实loss
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
        preds = (pred_prob > 0.7).cpu().numpy().reshape(-1)  # 提高决策阈值到0.7
        targets = y.cpu().numpy().reshape(-1)
        all_preds.extend(preds)
        all_targets.extend(targets)

    # 转为 numpy array 并强制为整型
    all_preds = np.asarray(all_preds).astype(np.int32)
    all_targets = np.asarray(all_targets).astype(np.int32)

    # 诊断性计数：正样本与预测正样本的数量，帮助判断模型是否预测任何正类
    total_pixels = all_targets.size
    pos_targets_count = int(np.sum(all_targets > 0))
    pos_preds_count = int(np.sum(all_preds > 0))
    print(f"[validate] counts: positives_in_targets={pos_targets_count}, positives_in_preds={pos_preds_count}, total_pixels={total_pixels}")

    # 诊断信息：检查唯一值分布，帮助判断是否为二分类或异常编码
    unique_targets = np.unique(all_targets)
    unique_preds = np.unique(all_preds)
    print(f"[validate] unique_targets={unique_targets}, unique_preds={unique_preds}")

    # 如标签不是严格的 0/1，则尝试按阈值二值化并再次打印
    if not set(unique_targets).issubset({0, 1}):
        print(f"[validate] Non-binary target values detected, binarizing with threshold 0.5. Sample uniques before: {unique_targets}")
        all_targets = (all_targets > 0.5).astype(np.int32)
        unique_targets = np.unique(all_targets)
        print(f"[validate] uniques after bin: {unique_targets}")

    if not set(unique_preds).issubset({0, 1}):
        print(f"[validate] Non-binary pred values detected, binarizing with threshold 0.5. Sample uniques before: {unique_preds}")
        all_preds = (all_preds > 0.5).astype(np.int32)
        unique_preds = np.unique(all_preds)
        print(f"[validate] pred uniques after bin: {unique_preds}")

    # 输出混淆矩阵以便进一步诊断（针对二分类）
    try:
        cm = confusion_matrix(all_targets, all_preds)
        print(f"[validate] Confusion matrix:\n{cm}")
    except Exception as e:
        print(f"[validate] Could not compute confusion matrix: {e}")

    # 根据标签的唯一值选择合适的平均方式：二分类使用 'binary'，否则使用 'micro'
    if set(np.unique(all_targets)).issubset({0, 1}):
        average_mode = 'binary'
    else:
        average_mode = 'micro'

    precision = precision_score(all_targets, all_preds, average=average_mode, zero_division=0)
    recall = recall_score(all_targets, all_preds, average=average_mode, zero_division=0)
    f1 = f1_score(all_targets, all_preds, average=average_mode, zero_division=0)
    accuracy = accuracy_score(all_targets, all_preds)

    print(f"[validate] metrics choice: average_mode={average_mode}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, acc={accuracy:.4f}")

    return total_loss / max(1, len(val_loader)), precision, recall, f1, accuracy


# -----------------------------
# Visualization（保留原脚本逻辑，适配双GPU模型）
# -----------------------------
@torch.no_grad()
def visualize_and_save(model, dataset, epoch=0, max_samples=5):
    if len(dataset) == 0:
        return

    model.eval()
    # 双GPU适配：获取原始模型（DataParallel包装后需用.module访问）
    raw_model = model.module if hasattr(model, 'module') else model
    
    # 打印数据集大小和可用样本数
    print(f"[visualize] Total validation samples: {len(dataset)}")
    print(f"[visualize] Will visualize {min(max_samples, len(dataset))} samples")
    
    # 获取所有样本的文件名，便于选择
    all_samples = [os.path.basename(sample[0]).replace('.h5', '') for sample in dataset.samples]
    print(f"[visualize] Available samples: {all_samples}")
    
    # 确定可用于可视化的样本索引：优先使用 dataset.samples（如果存在），并从中等间距抽取 max_samples 个样本
    if hasattr(dataset, 'samples') and isinstance(dataset.samples, list) and len(dataset.samples) > 0:
        total_samples = len(dataset.samples)
    else:
        total_samples = len(dataset)

    n_to_show = min(max_samples, total_samples)
    if n_to_show <= 0:
        return

    # 等间距选取索引，避免只看前几个样本
    indices = np.linspace(0, total_samples - 1, n_to_show, dtype=int)

    for idx in indices:
        X, y = dataset[idx]
        # 提取样本ID（优先从 dataset.samples 获取文件路径，否则使用索引命名）
        if hasattr(dataset, 'samples') and idx < len(dataset.samples):
            sample_id = os.path.basename(dataset.samples[idx][0]).replace('.h5', '')
        else:
            sample_id = f'sample_{idx}'

        X_tensor = torch.tensor(X).unsqueeze(0).to(DEVICE)
        logits = raw_model(X_tensor)
        pred_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        pred = (pred_prob > 0.7).astype(np.uint8)  # 提高决策阈值到0.7
        gt = y.astype(np.uint8)

        # 创建第一个图：显示CH532数据和云层标记
        fig1 = plt.figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        fig1.suptitle(f'Cloud Detection Result ({sample_id}) Epoch {epoch}')

        ch532_idx = CHANNELS.index('CH532')
        ch532_data = X[ch532_idx]
        # 使用 nearest 插值来避免颜色平滑导致掩模边缘模糊
        ax1.imshow(ch532_data, cmap='jet', origin='lower', aspect='auto', interpolation='nearest')

        # 更清晰地绘制真值：使用边界线替代半透明填充，避免模糊
        from scipy import ndimage
        kernel = np.ones((3, 3))
        gt_dil = ndimage.binary_dilation(gt, kernel)
        gt_ero = ndimage.binary_erosion(gt, kernel)
        gt_boundary = gt_dil & ~gt_ero

        # 显示真值边界（绿色），使用 imshow + masked array 保持像素对齐
        gt_edge_mask = np.ma.masked_where(~gt_boundary, np.ones_like(gt_boundary))
        ax1.imshow(gt_edge_mask, cmap=matplotlib.colors.ListedColormap(['green']),
                  alpha=1.0, origin='lower', aspect='auto', interpolation='nearest')

        # 显示预测结果（红色轮廓线），同样使用 nearest，保持线条清晰
        pred_dil = ndimage.binary_dilation(pred, kernel)
        pred_ero = ndimage.binary_erosion(pred, kernel)
        pred_boundary = pred_dil & ~pred_ero
        pred_edge_mask = np.ma.masked_where(~pred_boundary, np.ones_like(pred_boundary))
        ax1.imshow(pred_edge_mask, cmap=matplotlib.colors.ListedColormap(['red']),
                  alpha=1.0, origin='lower', aspect='auto', interpolation='nearest')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Height')

        # 调整布局并保存第一个图
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"cloud_pred_epoch{epoch:03d}_{sample_id}_detection.png"))
        plt.close()

        # 创建第二个图：显示廓线分布
        fig2 = plt.figure(figsize=(6, 8))
        ax2 = fig2.add_subplot(111)
        fig2.suptitle(f'Cloud Profile ({sample_id}) Epoch {epoch}')
        
        height = ch532_data.shape[0]
        
        # 计算每个高度层的云层出现比例
        gt_profile = np.mean(gt > 0, axis=1)
        pred_profile = np.mean(pred > 0, axis=1)
        
        # 绘制廓线
        heights = np.arange(height)
        ax2.plot(gt_profile, heights, 'g-', label='Ground Truth', linewidth=2)
        ax2.plot(pred_profile, heights, 'r-', label='Prediction', linewidth=2)
        
        # 设置图的显示范围和标签
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, height)
        ax2.set_xlabel('Cloud Fraction')
        ax2.set_ylabel('Height')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()
        
        # 调整布局并保存第二个图
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"cloud_pred_epoch{epoch:03d}_{sample_id}_profile.png"))
        plt.close()


# -----------------------------
# Main（保留原脚本逻辑，适配双GPU）
# -----------------------------
def main():
    # 创建训练历史记录器
    history = TrainingHistory()
    
    # 保留原脚本的训练/验证集划分逻辑
    # 修改为：0101-0126 训练，0127-0131 验证
    train_days = [f"{i:04d}" for i in range(101, 127)]  # 101..126
    val_days = [f"{i:04d}" for i in range(127, 132)]    # 127..131

    # --- 新增：逐天统计并打印每个 day 的可用样本数（存在 h5 且对应 tif 存在）
    def _count_samples_for_days(day_list):
        counts = []
        for d in day_list:
            h5_day_path = os.path.join(BASE_H5_DIR, d)
            tif_day_path = os.path.join(BASE_TIF_DIR, d)
            n = 0
            if os.path.isdir(h5_day_path):
                for fn in os.listdir(h5_day_path):
                    if fn.endswith('.h5'):
                        tif_fn = fn.replace('.h5', '.tif')
                        if os.path.exists(os.path.join(tif_day_path, tif_fn)):
                            n += 1
            counts.append((d, n))
        return counts

    train_counts = _count_samples_for_days(train_days)
    val_counts = _count_samples_for_days(val_days)
    print(f"Per-day sample counts (train): {train_counts}")
    print(f"Per-day sample counts (val):   {val_counts}")
    # --- end 新增

    print("Creating dataset...")
    train_dataset = LidarCloudDataset(train_days, preload=True, augment=True)
    val_dataset = LidarCloudDataset(val_days, preload=False, augment=False)

    # 适配双GPU的数据加载器，减少工作进程数以降低内存占用
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn, num_workers=1,
                              pin_memory=True)  # 启用pin_memory以加速数据传输
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn, num_workers=1,
                            pin_memory=True)

    print(f"Device: {DEVICE} | Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 双GPU模型包装
    model = UNet(in_ch=len(CHANNELS))
    if GPU_NUM >= 2:
        model = nn.DataParallel(model, device_ids=[0, 1])  # 使用前2张GPU
    model = model.to(DEVICE)

    # 使用AdamW优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                weight_decay=WEIGHT_DECAY)

    # 创建带warmup的余弦退火调度器
    def get_lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            # warmup阶段线性增加学习率
            return epoch / WARMUP_EPOCHS
        else:
            # warmup后使用余弦退火
            epoch_adj = epoch - WARMUP_EPOCHS
            return MIN_LR / LEARNING_RATE + 0.5 * (1 - MIN_LR / LEARNING_RATE) * \
                   (1 + math.cos(math.pi * epoch_adj / T_MAX))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)

    best_val_loss = float('inf')
    best_metrics = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, precision, recall, f1, accuracy = validate(model, val_loader)

        # 更新学习率调度器
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={current_lr:.6f}")
        print(f"Metrics: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Accuracy={accuracy:.4f}")

        # 更新训练历史
        metrics_dict = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }
        history.update(train_loss, val_loss, metrics_dict, current_lr)
        
        # 每5个epoch绘制训练曲线
        if epoch % 5 == 0 or epoch == EPOCHS:
            history.plot_training_curves(OUTPUT_DIR)
            
            # 绘制通道重要性分析
            plot_channel_importance(model, CHANNELS, OUTPUT_DIR)
            
            # 绘制高度分布统计
            plot_height_distribution(train_dataset, OUTPUT_DIR, 'Training Set')
            plot_height_distribution(val_dataset, OUTPUT_DIR, 'Validation Set')
            
            # 分析难易样本
            analyze_difficult_samples(model, val_dataset, OUTPUT_DIR)

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
            visualize_and_save(model, val_dataset, epoch, max_samples=10)  # 显示10个样本

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
