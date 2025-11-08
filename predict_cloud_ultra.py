import os
import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List
from matplotlib.lines import Line2D

# --- 脚本配置 ---
# 重要：这些配置必须与训练脚本 train_cloud_ultra.py 保持完全一致
CHANNELS = ['CH1064', 'PDR532', 'CH532', 'PDR355', 'CH355']
USE_LOG10 = True
CLIP_MIN = 1e-6
WINDOW_W = 60  # 滑动窗口宽度

# --- 路径配置 ---
# 模型和归一化数据所在的目录
OUTPUT_DIR = os.path.join(os.getcwd(), 'outputs')
# 预测结果图保存的目录
OUTPUT_PRED_DIR = os.path.join(os.getcwd(), 'outputs_prediction')
os.makedirs(OUTPUT_PRED_DIR, exist_ok=True)

# --- 设备配置 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 模型定义 (必须与训练时完全一致) ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
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

# --- 数据预处理函数 (必须与训练时完全一致) ---
def preprocess_channels(channel_arrays: List[np.ndarray], norm_stats: dict) -> np.ndarray:
    safe_arrays = []
    for i, arr in enumerate(channel_arrays):
        arr_safe = np.where(arr > 0, arr, CLIP_MIN)
        if USE_LOG10:
            arr_safe = np.log10(arr_safe)
        gmin, gmax = float(norm_stats['min'][i]), float(norm_stats['max'][i])
        arr_norm = (arr_safe - gmin) / (gmax - gmin + 1e-8)
        safe_arrays.append(arr_norm)
    return np.stack(safe_arrays, axis=0).astype(np.float32)

# --- 核心函数：滑动窗口预测与拼接 ---
@torch.no_grad()
def predict_full_image_with_stitching(model, x_full, window_w, device):
    """
    对单个完整的、预处理后的图像(x_full)进行滑动窗口预测。
    通过拼接重叠窗口的预测结果，生成一张完整的预测图。
    """
    model.eval()
    C, H, W_full = x_full.shape
    
    full_pred_prob = torch.zeros((H, W_full), device=device)
    sum_weights = torch.zeros((H, W_full), device=device)

    stride = window_w // 2
    window_weights = torch.bartlett_window(window_w, periodic=False).to(device).view(1, -1)

    for start in range(0, W_full, stride):
        end = min(start + window_w, W_full)
        win_data = x_full[:, :, start:end]
        
        pad_w = window_w - win_data.shape[2]
        if pad_w > 0:
            win_data = np.pad(win_data, ((0,0), (0,0), (0, pad_w)), 'constant')

        win_tensor = torch.from_numpy(win_data).unsqueeze(0).to(device)
        logits = model(win_tensor)
        pred_prob_win = torch.sigmoid(logits).squeeze()

        effective_len = end - start
        full_pred_prob[:, start:end] += pred_prob_win[:, :effective_len] * window_weights[:, :effective_len]
        sum_weights[:, start:end] += window_weights[:, :effective_len]

    sum_weights[sum_weights == 0] = 1.0
    final_pred_prob = full_pred_prob / sum_weights
    
    return final_pred_prob.cpu().numpy()

# --- 主函数 ---
def main():
    # --- 1. 加载模型和归一化统计数据 ---
    print("Loading model and normalization stats...")
    model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
    norm_stats_path = os.path.join(OUTPUT_DIR, 'norm_stats.pt')

    if not os.path.exists(model_path) or not os.path.exists(norm_stats_path):
        print(f"Error: Cannot find model '{model_path}' or stats '{norm_stats_path}'.")
        print("Please run train_cloud_ultra.py first to generate these files.")
        return

    model = UNet(in_ch=len(CHANNELS)).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    norm_stats = torch.load(norm_stats_path)
    print("Model and stats loaded successfully.")

    # --- 2. 指定要预测的H5文件 ---
    # !!! 重要：请在这里修改为您要预测的H5文件路径 !!!
    # 这是一个示例路径，请务必修改为您自己的文件路径
    h5_file_path = r"E:\003Study\Demo_910\demo\OriginalData\0101\58847.h5" 
    
    if not os.path.exists(h5_file_path):
        print(f"Error: Input file not found at '{h5_file_path}'")
        print("Please modify the 'h5_file_path' variable in the script.")
        return

    # --- 3. 加载并预处理完整数据 ---
    print(f"Loading and preprocessing: {h5_file_path}")
    with h5py.File(h5_file_path, 'r') as f:
        channel_arrays = [f[ch][:] for ch in CHANNELS if ch in f]
    
    x_full = preprocess_channels(channel_arrays, norm_stats)
    x_full = np.transpose(x_full, (0, 2, 1))  # (C, T, H) -> (C, H, W)

    # --- 4. 执行完整图像预测 ---
    print("Performing full image prediction with stitching...")
    pred_prob_full = predict_full_image_with_stitching(model, x_full, WINDOW_W, DEVICE)
    pred_mask_full = (pred_prob_full > 0.5).astype(np.uint8)

    # --- 5. 可视化并保存结果 ---
    print("Visualizing and saving the result...")
    plt.figure(figsize=(18, 6))

    ch1064_idx = CHANNELS.index('CH1064')
    background = x_full[ch1064_idx]
    plt.imshow(background, cmap='jet', origin='lower', aspect='auto')
    plt.colorbar(label='CH1064 (Normalized)')

    try:
        from skimage import measure
        pred_contours = measure.find_contours(pred_mask_full, 0.5)
        for contour in pred_contours:
            plt.plot(contour[:, 1], contour[:, 0], color='cyan', linewidth=2, linestyle='--')
    except ImportError:
        print("Warning: scikit-image not found. Prediction contours will not be drawn.")

    legend_elements = [Line2D([0], [0], color='cyan', lw=2, linestyle='--', label='Prediction')]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.xlabel("Time (Full Profile)")
    plt.ylabel("Height")
    plt.title(f"Full Prediction for {os.path.basename(h5_file_path)}")
    plt.tight_layout()

    output_filename = f"pred_{os.path.basename(h5_file_path).replace('.h5', '.png')}"
    save_path = os.path.join(OUTPUT_PRED_DIR, output_filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Prediction saved to: {save_path}")

if __name__ == '__main__':
    main()
