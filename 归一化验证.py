# plot_profiles_3files.py
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt

# ----------------------------
# 站点-通道归一化上限（与你现用一致）
# ----------------------------
'''NORMALIZE_RANGE = {
    51628: {"CH355": 3.948e5, "CH532": 5.169e5, "CH1064": 3.283e5},
    53845: {"CH355": 4.158e5, "CH532": 4.750e5, "CH1064": 2.294e6},
    58847: {"CH355": 4.547e5, "CH532": 3.911e5, "CH1064":1.321e5},
}'''
NORMALIZE_RANGE = {
    51628: {"CH355": 3.948e5, "CH532": 3.948e5, "CH1064": 3.948e5},
    53845: {"CH355": 3.948e5, "CH532": 3.948e5, "CH1064": 3.948e5},
    58847: {"CH355": 3.948e5, "CH532": 3.948e5, "CH1064": 3.948e5},
}
CHANNEL_KEYS = ['CH355', 'CH532', 'CH1064', 'PDR355', 'PDR532']


def normalize_to_unit_range(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    norm = (data - min_val) / (max_val - min_val)
    return norm


def _extract_station_id(h5_path: Path) -> int:
    try:
        return int(h5_path.stem)
    except Exception as e:
        raise ValueError(f"无法从文件名解析站点编号：{h5_path.name}（需要像 58847.h5 这样的命名）") from e


def _load_channel_and_height(h5_path: Path, channel: str):
    if not h5_path.exists():
        raise FileNotFoundError(f"文件不存在：{h5_path}")

    with h5py.File(h5_path, 'r') as f:
        if 'height' not in f:
            raise KeyError(f"{h5_path} 中缺少 'height' 数据集")
        if channel not in f:
            raise KeyError(f"{h5_path} 中缺少通道 '{channel}' 数据集")

        height = f['height'][:]  # shape: (H,)
        arr = f[channel][:]      # shape: (T, H) 或 (H, T)

    H = len(height)
    if arr.ndim != 2:
        raise ValueError(f"{h5_path} 的通道 {channel} 不是二维数组。当前形状: {arr.shape}")
    # 统一为 (T, H)
    if arr.shape[1] == H:
        pass  # already (T, H)
    elif arr.shape[0] == H:
        arr = arr.T  # (H, T) -> (T, H)
    else:
        raise ValueError(f"{h5_path} 的 {channel} 形状 {arr.shape} 与 height 长度 {H} 不匹配。")
    return arr, height


def _range_correct(arr_Th: np.ndarray, height: np.ndarray, channel: str) -> np.ndarray:
    """ 对 CH* 通道做 × height^2 的距离校正；PDR* 不校正 """
    if channel.upper().startswith('CH'):
        return arr_Th * (height[None, :] ** 2)
    return arr_Th


def _get_norm_max(station_id: int, channel: str) -> float:
    if channel.upper().startswith('PDR'):
        return 1.0  # PDR 默认 0-1
    if station_id not in NORMALIZE_RANGE:
        raise KeyError(f"站点 {station_id} 未在 NORMALIZE_RANGE 中定义。")
    chmap = NORMALIZE_RANGE[station_id]
    if channel not in chmap:
        raise KeyError(f"站点 {station_id} 未定义通道 {channel} 的归一化上限。")
    return chmap[channel]


def load_and_make_profile(h5_path: Path, channel: str, t_idx: int):
    station_id = _extract_station_id(h5_path)
    arr, height = _load_channel_and_height(h5_path, channel)
    T = arr.shape[0]
    if not (0 <= t_idx < T):
        raise IndexError(f"{h5_path} 的时间索引超界：给定 {t_idx}，有效范围 0~{T-1}")

    # 距离校正
    arr_rc = _range_correct(arr, height, channel)

    # 取单条廓线 (H,)
    prof = arr_rc[t_idx, :].astype(np.float64)
    prof = np.nan_to_num(prof, nan=0.0, posinf=0.0, neginf=0.0)

    # 归一化到 [0,1]
    max_val = _get_norm_max(station_id, channel)
    prof_norm = normalize_to_unit_range(prof, 0.0, max_val)

    # 高度单位：默认按米转 km
    height_km = height 

    return {
        "station_id": station_id,
        "t_idx": t_idx,
        "height_km": height_km,
        "profile": prof_norm
    }


def plot_three_profiles(h5_files, channel: str, t_indices, title_extra: str = None, save: Path | None = None):
    """h5_files: [p1, p2, p3]; t_indices: [t1, t2, t3]"""
    if len(h5_files) != 3 or len(t_indices) != 3:
        raise ValueError("需要恰好三个 h5 文件与三个 time index。")

    traces = []
    for p, t_idx in zip(h5_files, t_indices):
        traces.append(load_and_make_profile(Path(p), channel, int(t_idx)))

    plt.figure(figsize=(6.2, 8))
    for tr in traces:
        label = f"{tr['station_id']} @ {tr['t_idx']}"
        plt.plot(tr["profile"], tr["height_km"], label=label)

    plt.xlabel(f"{channel}（距离校正后并归一化至 0–1）")
    #X scale log
    plt.xscale('log')
    plt.ylabel("Height (km)")
    ttl = f"{channel} profiles @ time indices {t_indices}"
    if title_extra:
        ttl += f" — {title_extra}"
    plt.title(ttl)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Station @ t")
    plt.tight_layout()

    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, dpi=200)
        print(f"✅ Saved figure to: {save}")
        plt.close()
    else:
        plt.show()


def main():
    # ====== 直接在这里填写你的三个文件与各自的时间索引 ======
    H5_FILE_1 = r"F:\Workspace\Projects\LidarCloud\CMA\2025\2025\0108\58847.h5"
    H5_FILE_2 = r"F:\Workspace\Projects\LidarCloud\CMA\2025\2025\0108\53845.h5"
    H5_FILE_3 = r"F:\Workspace\Projects\LidarCloud\CMA\2025\2025\0108\51628.h5"

    # 为三个文件分别指定 time index（0~1439）
    T1 = 120
    T2 = 170
    T3 = 140

    CHANNEL = "CH1064"  # 例如：'CH1064' / 'CH532' / 'CH355' / 'PDR532' / 'PDR355'
    SAVE_PATH = None     # 例如 r"F:\figs\CH1064_t194_compare.png"；为 None 时直接弹窗显示

    plot_three_profiles(
        [H5_FILE_1, H5_FILE_2, H5_FILE_3],
        channel=CHANNEL,
        t_indices=[T1, T2, T3],
        title_extra=None,
        save=SAVE_PATH
    )


if __name__ == "__main__":
    main()
