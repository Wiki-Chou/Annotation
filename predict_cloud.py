import os
import h5py
import tifffile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List
from tqdm import tqdm

# -----------------------------
# Config（与训练脚本保持一致）
# -----------------------------
CHANNELS = ['CH1064', 'PDR532', 'CH532', 'PDR355', 'CH355']
USE_LOG10 = True
CLIP_MIN = 1e-6

# 模型和数据路径配置
MODEL_PATH = os.path.join(os.getcwd(), '1031', 'best_model_1031.pth')  # 最优模型路径
INPUT_H5_FILE = r"E:/001/CMA/CMA/data/result/2025/0131/51628.h5"  # 要预测的H5文件
INPUT_TIF_FILE = r"E:/001/CMA/CMA/data/mask/2025/0131/51628.tif"  # 对应的TIF真值文件（可选）
OUTPUT_DIR = os.path.join(os.getcwd(), '1031/prediction')  # 预测结果输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# -----------------------------
# Preprocessing（与训练脚本完全一致）
# -----------------------------
def preprocess_channels(channel_arrays: List[np.ndarray]) -> np.ndarray:
    """预处理通道数据，与训练时保持一致"""
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
# Model（与训练脚本完全一致的UNet结构）
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
# Load Model
# -----------------------------
def load_model(model_path: str) -> nn.Module:
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = UNet(in_ch=len(CHANNELS))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded from: {model_path}")
    return model


# -----------------------------
# Predict Single File
# -----------------------------
@torch.no_grad()
def predict_h5_file(model: nn.Module, h5_path: str, tif_path: str = None) -> np.ndarray:
    """
    预测单个H5文件

    Args:
        model: 训练好的模型
        h5_path: H5文件路径
        tif_path: 可选，对应的TIF真值文件路径

    Returns:
        pred_mask: 预测的二值掩码 (H, W)
    """
    # 1. 读取并预处理数据
    with h5py.File(h5_path, 'r') as f:
        data = [f[ch][:] for ch in CHANNELS if ch in f]

    x = preprocess_channels(data)
    x = np.transpose(x, (0, 2, 1))  # (C, T, H) → (C, H, W)

    # 2. 读取ground truth（如果提供）
    gt = None
    if tif_path and os.path.exists(tif_path):
        gt = tifffile.imread(tif_path).astype(np.float32)
        # 与训练脚本保持一致的处理
        gt = np.where(gt > 0.5, 1.0, 0.0)
        if gt.shape[0] == x.shape[2] and gt.shape[1] == x.shape[1]:
            gt = np.transpose(gt)
        gt = np.flipud(gt).copy()
        if gt.shape != x.shape[1:]:
            from skimage.transform import resize
            gt = resize(gt, x.shape[1:], order=0, preserve_range=True)
            gt = np.where(gt > 0.5, 1.0, 0.0)
        gt = gt.astype(np.uint8)

    # 3. 模型推理
    x_tensor = torch.tensor(x).unsqueeze(0).to(DEVICE)  # (1, C, H, W)
    logits = model(x_tensor)
    pred_prob = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H, W)
    pred_mask = (pred_prob > 0.5).astype(np.uint8)

    # 4. 保存PNG可视化结果
    parent_folder = os.path.basename(os.path.dirname(h5_path))
    base_name = os.path.splitext(os.path.basename(h5_path))[0]
    output_name = f"{parent_folder}_{base_name}_pred"
    png_path = os.path.join(OUTPUT_DIR, f"{output_name}.png")
    visualize_prediction(x, pred_mask, gt, png_path)
    print(f"Prediction saved to: {png_path}")

    return pred_mask


# -----------------------------
# Visualization
# -----------------------------
def visualize_prediction(x: np.ndarray, pred_mask: np.ndarray, gt: np.ndarray = None, save_path: str = None):
    """可视化预测结果（与训练脚本风格完全一致）"""
    plt.figure(figsize=(12, 6))

    # 使用CH1064作为背景
    ch1064_idx = CHANNELS.index('CH1064')
    ch1064_data = x[ch1064_idx]
    plt.imshow(ch1064_data, cmap='jet', origin='lower', aspect='auto')
    plt.colorbar(label='CH1064 (normalized)')

    from skimage import measure

    # 绘制真值轮廓（白色实线，在红黄背景上最清晰）- 如果提供了ground truth
    if gt is not None:
        gt_contours = measure.find_contours(gt, 0.5)
        for contour in gt_contours:
            plt.plot(contour[:, 1], contour[:, 0], color='white', linewidth=3,
                     linestyle='-', label='Ground Truth')

    # 绘制预测轮廓（深蓝色虚线，与暖色背景形成对比）
    pred_contours = measure.find_contours(pred_mask, 0.5)
    for contour in pred_contours:
        plt.plot(contour[:, 1], contour[:, 0], color='blue', linewidth=2.5,
                 linestyle='--', label='Prediction')

    # 去重图例（只显示一次）
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.xlabel('Time')
    plt.ylabel('Height')
    plt.title(f'Cloud Detection Prediction')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    """
    主函数：预测指定的H5文件

    使用方法：
    修改脚本开头的 INPUT_H5_FILE 路径，然后运行脚本即可
    """
    # 检查文件是否存在
    if not os.path.exists(INPUT_H5_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_H5_FILE}")

    print("\n" + "=" * 60)
    print(f"Predicting file: {INPUT_H5_FILE}")
    print("=" * 60)

    # 加载模型
    model = load_model(MODEL_PATH)

    # 预测单个文件
    pred_mask = predict_h5_file(model, INPUT_H5_FILE, INPUT_TIF_FILE)
   # pred_mask = predict_h5_file(model, INPUT_H5_FILE,)
    print(f"\nPrediction completed!")
    print(f"Prediction shape: {pred_mask.shape}")
    print(
        f"Positive pixels (cloud): {np.sum(pred_mask)} / {pred_mask.size} ({100 * np.sum(pred_mask) / pred_mask.size:.2f}%)")


if __name__ == '__main__':
    main()
