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
try:
    from tqdm import tqdm
except Exception:
    # 环境缺少tqdm时的降级：使用普通可迭代对象
    def tqdm(iterable, **kwargs):
        return iterable
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
BATCH_SIZE = 16  
ACCUMULATION_STEPS = 2  # 启用梯度累积（有效batch=16*2=32，需确保显存足够）
EPOCHS = 200  # 训练轮数调整为200
LEARNING_RATE = 3e-4  # 降低学习率至3e-4，更稳定
WEIGHT_DECAY = 1e-4
POS_WEIGHT = 10.0
EARLY_STOPPING_PATIENCE = 20  # 早停：验证loss连续20个epoch未改善则停止训练

OUTPUT_DIR = os.path.join(os.getcwd(), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 固定宽度窗口配置（沿时间/宽度维度切片）
WINDOW_W = 60                 # 每次仅送入60条廓线（列）
WINDOW_STRIDE_TRAIN = 60      # 训练集窗口步长（可设小于60形成重叠）
WINDOW_STRIDE_VAL = 60        # 验证集窗口步长
PAD_SHORT_WINDOW = True       # 当样本宽度小于窗口宽度时，是否用0右侧padding以凑满窗口
DROP_LAST_TAIL = False        # 宽度不能整除步长时，是否丢弃尾部不足窗口的部分（False时会补上覆盖到末尾的最后一个窗口）

# 轻量数据增强配置（稳妥）：对全部5个通道添加轻微高斯噪声
GAUSS_NOISE_P = 0.5           # 增强概率（每个样本/窗口）
GAUSS_NOISE_SIGMA = 0.01      # 高斯噪声标准差（数据已归一化到[0,1]，取小值更稳妥）

# 仪器增益/校准相关增强（贴近物理）
# CH通道（后向散射）做“组增益”+“通道微调”+“小偏置”；PDR通道做更小幅度的增益/偏置；可选轻度gamma压扩
CH_GAIN_P = 0.5               # 施加CH组增益与微调的概率
CH_GROUP_GAIN_RANGE = (0.95, 1.05)   # CH整体增益范围（所有CH通道共用一个因子）
CH_PER_GAIN_RANGE = (0.98, 1.02)     # CH每通道细微增益范围
CH_OFFSET_RANGE = (-0.01, 0.01)      # CH每通道偏置范围

PDR_GAIN_P = 0.4              # 施加PDR增益/偏置的概率（PDR更保守）
PDR_GAIN_RANGE = (0.98, 1.02)
PDR_OFFSET_RANGE = (-0.005, 0.005)

CH_GAMMA_P = 0.3              # 对CH通道做轻度gamma压扩的概率
CH_GAMMA_RANGE = (0.9, 1.1)   # 0.9<gamma<1.1，靠近1，避免过强非线性


# -----------------------------
# Preprocessing（保留原逻辑）
# -----------------------------
def preprocess_channels(channel_arrays: List[np.ndarray], norm_stats: dict | None = None) -> np.ndarray:
    """按通道预处理。
    当提供 norm_stats 时，使用全数据集的每通道全局 min/max 做归一化；
    否则退化为每样本每通道的 min-max 归一化。
    """
    safe_arrays = []
    for i, arr in enumerate(channel_arrays):
        arr_safe = np.where(arr > 0, arr, CLIP_MIN)
        if USE_LOG10:
            arr_safe = np.log10(arr_safe)

        if norm_stats is not None:
            gmin = float(norm_stats['min'][i])
            gmax = float(norm_stats['max'][i])
        else:
            gmin = float(np.min(arr_safe))
            gmax = float(np.max(arr_safe))

        arr_norm = (arr_safe - gmin) / (gmax - gmin + 1e-8)
        safe_arrays.append(arr_norm)
    x = np.stack(safe_arrays, axis=0)  # (C, T, H)
    return x.astype(np.float32)


def compute_global_channel_stats(day_list: List[str]) -> dict:
    """遍历给定日期列表，计算每通道在全数据集上的全局 min/max（在log与裁剪之后）。"""
    mins = [np.inf] * len(CHANNELS)
    maxs = [-np.inf] * len(CHANNELS)
    file_count = 0

    for day_folder in day_list:
        h5_day_path = os.path.join(BASE_H5_DIR, day_folder)
        if not os.path.isdir(h5_day_path):
            continue
        for file in sorted(os.listdir(h5_day_path)):
            if not file.endswith('.h5'):
                continue
            h5_path = os.path.join(h5_day_path, file)
            try:
                with h5py.File(h5_path, 'r') as f:
                    for ci, ch in enumerate(CHANNELS):
                        if ch not in f:
                            continue
                        arr = f[ch][:]
                        arr_safe = np.where(arr > 0, arr, CLIP_MIN)
                        if USE_LOG10:
                            arr_safe = np.log10(arr_safe)
                        cmin = float(np.min(arr_safe))
                        cmax = float(np.max(arr_safe))
                        if cmin < mins[ci]:
                            mins[ci] = cmin
                        if cmax > maxs[ci]:
                            maxs[ci] = cmax
                file_count += 1
            except Exception:
                continue

    mins = np.array([0.0 if not np.isfinite(v) else v for v in mins], dtype=np.float32)
    maxs = np.array([1.0 if (not np.isfinite(v)) else v for v in maxs], dtype=np.float32)
    for i in range(len(CHANNELS)):
        if maxs[i] - mins[i] < 1e-8:
            maxs[i] = mins[i] + 1.0

    print(f"Global channel stats computed on {file_count} files.")
    return {'min': mins, 'max': maxs}


# -----------------------------
# Dataset（确保标签只包含0和1）
# -----------------------------
class LidarCloudDataset(Dataset):
    def __init__(self, day_list: List[str], preload: bool = True, augment: bool = False, norm_stats: dict | None = None):
        self.samples: List[Tuple[str, str]] = []
        self.norm_stats = norm_stats
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
        x = preprocess_channels(data, norm_stats=self.norm_stats)
        x = np.transpose(x, (0, 2, 1))  # (C, T, H) → (C, H, W)

        # 2. 读取TIF并确保标签只包含0和1（核心修改）
        label = tifffile.imread(tif_path).astype(np.float32)
        
        # 关键：强制标签二值化（仅保留等于1的像素为1，其余全为0）
        label = np.where(label == 1.0, 1.0, 0.0)
        
        # 转置标签确保与输入尺寸一致
        if label.shape[0] == x.shape[2] and label.shape[1] == x.shape[1]:
            label = np.transpose(label)
        label = np.flipud(label).copy()
        
        # 尺寸不匹配时采用最近邻方式调整，并再次确保二值化（不依赖第三方库）
        if label.shape != x.shape[1:]:
            def _nearest_resize_label(arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
                th, tw = target_hw
                sh, sw = arr.shape
                # 通过线性映射索引实现最近邻
                y_idx = (np.linspace(0, max(1, sh) - 1, th)).astype(int)
                x_idx = (np.linspace(0, max(1, sw) - 1, tw)).astype(int)
                return arr[np.ix_(y_idx, x_idx)]
            label = _nearest_resize_label(label, x.shape[1:])
            # 调整后再次二值化：仅1为1，其余为0
            label = np.where(label == 1.0, 1.0, 0.0)
        
        # 最终校验：确保没有其他值
        unique_vals = np.unique(label)
        if not np.array_equal(unique_vals, [0.0]) and not np.array_equal(unique_vals, [1.0]) and not np.array_equal(np.sort(unique_vals), [0.0, 1.0]):
            print(f"Warning: Unexpected values in label {tif_path}: {unique_vals}")
            # 强制清理为严格二值（仅1视为正类）
            label = np.where(label == 1.0, 1.0, 0.0)
        
        return x, label

    def _augment(self, x, y):
        # 水平翻转（时间轴）保留；垂直翻转（高度轴）移除以保持物理意义
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=2).copy()  # W/Time 轴
            y = np.flip(y, axis=1).copy()  # 与x保持一致
        # 垂直翻转会颠倒高度，违背物理意义，已禁用
        # if np.random.rand() > 0.5:
        #     x = np.flip(x, axis=1).copy()  # H 轴
        #     y = np.flip(y, axis=0).copy()
        if np.random.rand() > 0.5:
            factor = 0.8 + 0.4 * np.random.rand()
            x = x * factor
        # 轻量高斯噪声（对全部通道）：稳妥设置，按概率添加小幅噪声，并裁剪回[0,1]
        if np.random.rand() < GAUSS_NOISE_P:
            noise = np.random.normal(loc=0.0, scale=GAUSS_NOISE_SIGMA, size=x.shape).astype(np.float32)
            x = x + noise
            x = np.clip(x, 0.0, 1.0)

        # -------------------------------
        # 仪器增益/校准相关增强（贴近物理）
        # -------------------------------
        # 识别通道分组
        ch_indices = [i for i, name in enumerate(CHANNELS) if name.startswith('CH')]
        pdr_indices = [i for i, name in enumerate(CHANNELS) if name.startswith('PDR')]

        # CH组增益 + 每通道微调 + 小偏置
        if len(ch_indices) > 0 and np.random.rand() < CH_GAIN_P:
            g_min, g_max = CH_GROUP_GAIN_RANGE
            pg_min, pg_max = CH_PER_GAIN_RANGE
            b_min, b_max = CH_OFFSET_RANGE

            group_gain = np.float32(np.random.uniform(g_min, g_max))
            per_gain = np.random.uniform(pg_min, pg_max, size=len(ch_indices)).astype(np.float32)
            offsets = np.random.uniform(b_min, b_max, size=len(ch_indices)).astype(np.float32)

            for idx_c, ch in enumerate(ch_indices):
                x[ch] = x[ch] * (group_gain * per_gain[idx_c]) + offsets[idx_c]
            x = np.clip(x, 0.0, 1.0)

        # PDR增益/偏置（更保守）
        if len(pdr_indices) > 0 and np.random.rand() < PDR_GAIN_P:
            g_min, g_max = PDR_GAIN_RANGE
            b_min, b_max = PDR_OFFSET_RANGE
            per_gain = np.random.uniform(g_min, g_max, size=len(pdr_indices)).astype(np.float32)
            offsets = np.random.uniform(b_min, b_max, size=len(pdr_indices)).astype(np.float32)
            for idx_p, pdri in enumerate(pdr_indices):
                x[pdri] = x[pdri] * per_gain[idx_p] + offsets[idx_p]
            x = np.clip(x, 0.0, 1.0)

        # 轻度gamma压扩（仅CH通道）
        if len(ch_indices) > 0 and np.random.rand() < CH_GAMMA_P:
            gmin, gmax = CH_GAMMA_RANGE
            gamma = float(np.random.uniform(gmin, gmax))
            # 避免负值；对[0,1]做幂运算
            for ch in ch_indices:
                x[ch] = np.power(np.clip(x[ch], 0.0, 1.0), gamma, dtype=np.float32)
            x = np.clip(x, 0.0, 1.0)
        return x, y


class LidarCloudWindowedDataset(Dataset):
    """将基础数据集沿宽度（时间）维切成定宽窗口（如60列）以形成固定尺寸输入。
    - 若样本宽度W < window_w 且 pad_short=True，则右侧0填充到window_w；否则可选择跳过。
    - 若步长不能整除（W - window_w）且 drop_last=False，则补一个覆盖到末尾的窗口。
    """
    def __init__(self, base_ds: LidarCloudDataset, window_w: int, stride: int,
                 pad_short: bool = True, drop_last: bool = False):
        self.base = base_ds
        self.window_w = window_w
        self.stride = stride
        self.pad_short = pad_short
        self.drop_last = drop_last
        # 预构建索引：(sample_idx, start_col)
        self.win_index: List[Tuple[int, int]] = []
        for si, (h5_path, _tif_path) in enumerate(self.base.samples):
            # 快速读任一通道的形状以获取时间长度T（对应宽度W）
            w_len = None
            try:
                with h5py.File(h5_path, 'r') as f:
                    for ch in CHANNELS:
                        if ch in f:
                            # 原数据形状通常为 (T, H)
                            shape = f[ch].shape
                            if len(shape) >= 2:
                                w_len = int(shape[0])  # 宽度=时间维T
                                break
            except Exception:
                pass
            if w_len is None:
                continue

            if w_len < self.window_w:
                if self.pad_short:
                    self.win_index.append((si, 0))
                # 否则跳过该样本
                continue

            # 正常切窗
            start = 0
            last_start = max(0, w_len - self.window_w)
            while start <= w_len - self.window_w:
                self.win_index.append((si, start))
                start += self.stride
            if (not self.drop_last) and (len(self.win_index) > 0):
                # 若最后一个窗口未覆盖至末尾，则补一个以对齐末端
                if self.win_index[-1][0] == si and self.win_index[-1][1] != last_start:
                    self.win_index.append((si, last_start))

    def __len__(self):
        return len(self.win_index)

    def __getitem__(self, idx: int):
        si, start = self.win_index[idx]
        # 取出样本
        if self.base.preload and hasattr(self.base, 'data_cache'):
            x, y = self.base.data_cache[si]
        else:
            h5_path, tif_path = self.base.samples[si]
            x, y = self.base._load_sample(h5_path, tif_path)
        # 切窗/填充
        H, W = x.shape[1], x.shape[2]
        if W >= self.window_w:
            end = min(start + self.window_w, W)
            x_win = x[:, :, start:end]
            y_win = y[:, start:end]
            # 末尾不足时补齐（理论上start的选择已避免）
            if x_win.shape[2] < self.window_w and self.pad_short:
                pad_w = self.window_w - x_win.shape[2]
                x_win = np.pad(x_win, ((0, 0), (0, 0), (0, pad_w)), mode='constant')
                y_win = np.pad(y_win, ((0, 0), (0, pad_w)), mode='constant')
        else:
            # 整个样本宽度不足窗口宽度
            if self.pad_short:
                pad_w = self.window_w - W
                x_win = np.pad(x, ((0, 0), (0, 0), (0, pad_w)), mode='constant')
                y_win = np.pad(y, ((0, 0), (0, pad_w)), mode='constant')
            else:
                # 回退为返回原始（不建议）
                x_win, y_win = x, y
        # 仅在训练集（base.augment=True）时进行增强；增强在切窗与padding之后进行
        if getattr(self.base, 'augment', False):
            x_win, y_win = self.base._augment(x_win, y_win)
        return x_win, y_win


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
# Visualization（修改为支持完整图像拼接）
# -----------------------------
@torch.no_grad()
def predict_full_image_with_stitching(model, x_full, window_w, device):
    """
    新函数：对单个完整的、预处理后的图像(x_full)进行滑动窗口预测。
    通过拼接重叠窗口的预测结果，生成一张完整的预测图。
    - model: 训练好的模型。
    - x_full: 形状为 (C, H, W_full) 的完整输入张量。
    - window_w: 滑动窗口的宽度，应与训练时一致。
    - device: 计算设备 (cuda/cpu)。
    返回: 形状为 (H, W_full) 的完整概率图。
    """
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.eval()

    C, H, W_full = x_full.shape
    
    # 1. 创建一个空的概率图和权重图，用于平滑拼接
    full_pred_prob = torch.zeros((H, W_full), device=device)
    sum_weights = torch.zeros((H, W_full), device=device)

    # 2. 定义滑动步长和重叠区域
    stride = window_w // 2
    
    # 3. 定义一个三角权重窗口，用于平滑拼接
    window_weights = torch.bartlett_window(window_w, periodic=False).to(device)
    window_weights = window_weights.view(1, -1) # (1, window_w)

    # 4. 滑动窗口预测
    for start in range(0, W_full, stride):
        end = min(start + window_w, W_full)
        
        win_data = x_full[:, :, start:end]
        
        pad_w = window_w - win_data.shape[2]
        if pad_w > 0:
            win_data_tensor = torch.from_numpy(win_data).to(device)
            win_data_padded = F.pad(win_data_tensor, (0, pad_w), 'constant', 0)
        else:
            win_data_padded = torch.from_numpy(win_data).to(device)

        logits = raw_model(win_data_padded.unsqueeze(0))
        pred_prob_win = torch.sigmoid(logits).squeeze(0).squeeze(0) # (H, window_w)

        # 5. 将预测结果加权累加到完整概率图中
        effective_len = end - start
        full_pred_prob[:, start:end] += pred_prob_win[:, :effective_len] * window_weights[:, :effective_len]
        sum_weights[:, start:end] += window_weights[:, :effective_len]

    # 6. 对累加后的概率图进行平均
    sum_weights[sum_weights == 0] = 1.0
    final_pred_prob = full_pred_prob / sum_weights
    
    return final_pred_prob.cpu().numpy()


@torch.no_grad()
def visualize_and_save(model, base_dataset, epoch=0, prefix=''):
    """
    可视化函数（已修改）：
    - 从 base_dataset (包含完整样本) 中取样。
    - 调用 predict_full_image_with_stitching 生成完整预测。
    - 绘制完整的背景、真值和预测图。
    """
    if len(base_dataset.samples) == 0:
        return

    model.eval()
    
    indices = np.random.choice(len(base_dataset.samples), min(5, len(base_dataset.samples)), replace=False)

    for idx in indices:
        if base_dataset.preload:
            X_full, y_full = base_dataset.data_cache[idx]
        else:
            h5_path, tif_path = base_dataset.samples[idx]
            X_full, y_full = base_dataset._load_sample(h5_path, tif_path)
        
        pred_prob_full = predict_full_image_with_stitching(model, X_full, WINDOW_W, DEVICE)
        
        pred_full = (pred_prob_full > 0.5).astype(np.uint8)
        gt_full = y_full.astype(np.uint8)

        plt.figure(figsize=(18, 6))
        
        ch1064_idx = CHANNELS.index('CH1064')
        ch1064_data = X_full[ch1064_idx]
        plt.imshow(ch1064_data, cmap='jet', origin='lower', aspect='auto')
        plt.colorbar(label='CH1064 (normalized)')
        
        try:
            from skimage import measure
        except ImportError:
            print("Warning: scikit-image not found. Contours will not be drawn.")
            measure = None

        if measure:
            gt_contours = measure.find_contours(gt_full, 0.5)
            for contour in gt_contours:
                plt.plot(contour[:, 1], contour[:, 0], color='white', linewidth=2, linestyle='-')
            
            pred_contours = measure.find_contours(pred_full, 0.5)
            for contour in pred_contours:
                plt.plot(contour[:, 1], contour[:, 0], color='cyan', linewidth=2, linestyle='--')

        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='white', lw=2, label='Ground Truth'),
                           Line2D([0], [0], color='cyan', lw=2, linestyle='--', label='Prediction')]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.xlabel('Time (Full Profile)')
        plt.ylabel('Height')
        plt.title(f'Full Prediction vs GT (Sample {idx}) Epoch {epoch}')
        plt.tight_layout()
        
        original_h5_path = base_dataset.samples[idx][0]
        base_filename = os.path.basename(original_h5_path).replace('.h5', '')
        filename = f"{prefix}_full_pred_epoch{epoch:03d}_{base_filename}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()


# -----------------------------
# Main（主函数）
# -----------------------------
def main():
    # 验证集：每5天取一天，共7天；其余24天为训练集
    val_days = ['0101', '0106', '0111', '0116', '0121', '0126', '0131']
    train_days = [f"{i:04d}" for i in range(101, 132) if f"{i:04d}" not in val_days]

    print("Computing global per-channel normalization stats (train+val days)...")
    norm_stats = compute_global_channel_stats(sorted(set(train_days + val_days)))
    print("Creating dataset...")
    base_train = LidarCloudDataset(train_days, preload=True, augment=True, norm_stats=norm_stats)
    # 预加载验证集，减少验证时CPU/HDF5 I/O压力
    base_val = LidarCloudDataset(val_days, preload=True, augment=False, norm_stats=norm_stats)

    # 使用定宽窗口包装数据集
    train_dataset = LidarCloudWindowedDataset(
        base_train, window_w=WINDOW_W, stride=WINDOW_STRIDE_TRAIN,
        pad_short=PAD_SHORT_WINDOW, drop_last=DROP_LAST_TAIL
    )
    val_dataset = LidarCloudWindowedDataset(
        base_val, window_w=WINDOW_W, stride=WINDOW_STRIDE_VAL,
        pad_short=PAD_SHORT_WINDOW, drop_last=DROP_LAST_TAIL
    )

    # 预加载后CPU负担主要来自拼接与拷贝，降低workers并开启pin_memory以减轻CPU占用
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    print(f"Device: {DEVICE} | Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 双GPU模型包装
    model = UNet(in_ch=len(CHANNELS))
    if GPU_NUM >= 2:
        model = nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8,
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

        # 每20个epoch可视化一次（不在“最佳epoch”单独可视化）
        if epoch % 20 == 0 or epoch == EPOCHS:
            print(f"Generating full predictions at epoch {epoch}...")
            # 注意：这里传入 base_val，它包含完整的、未切片的样本
            visualize_and_save(model, base_val, epoch)

    # 保存最佳指标和归一化统计数据
    metrics_file = os.path.join(OUTPUT_DIR, 'best_model_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Best Model Evaluation Metrics at Epoch {best_epoch}:\n")
        f.write(f"Validation Loss: {best_val_loss:.4f}\n")
        f.write(f"Precision: {best_metrics['precision']:.4f}\n")
        f.write(f"Recall: {best_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {best_metrics['f1']:.4f}\n")
        f.write(f"Accuracy: {best_metrics['accuracy']:.4f}\n")

    # 保存归一化统计数据，供预测时使用
    norm_stats_path = os.path.join(OUTPUT_DIR, 'norm_stats.pt')
    torch.save(norm_stats, norm_stats_path)
    print(f"Normalization stats saved to: {norm_stats_path}")

    # 绘制训练和验证loss曲线
    plt.figure(figsize=(10, 6))
    # 注意：若早停提前结束，loss 列表长度 < EPOCHS，需按实际轮数绘图
    epochs_done = len(train_losses)
    epochs_range = range(1, epochs_done + 1)
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
    print(f"Predictions images are saved periodically (every 20 epochs).")
    print("="*60)


if __name__ == '__main__':
    main()
