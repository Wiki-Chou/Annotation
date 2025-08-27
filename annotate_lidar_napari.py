import h5py
import numpy as np
from pathlib import Path
import napari
import tifffile
from pathlib import Path
import matplotlib.pyplot  as plt#194 

# ----------------------------
# 通道映射 + 归一化标准
# ----------------------------
normalize_range={
    51628: {"CH355": 3.948e+05, "CH532": 5.169e+05 , "CH1064": 2.283e+05
    },
    53845: {"CH355": 4.158e+05, "CH532": 4.750e+05, "CH1064": 2.294e+05
    },
    58847: {"CH355": 4.547e+05, "CH532": 3.911e+05, "CH1064": 2.321e+05
    }
}

channel_keys = ['CH355', 'CH532', 'CH1064', 'PDR355', 'PDR532']


def load_lidar_channels(h5_path, selected_keys=None):
    selected_keys = selected_keys or channel_keys
    data_dict = {}
    with h5py.File(h5_path, 'r') as f:
        for key in selected_keys:
            if key in f:
                data_dict[key] = f[key][:]
            else:
                print(f"⚠️ Warning: {key} not found in file.")
    return data_dict


def normalize_to_unit_range(data, min_val, max_val):
    norm = (data - min_val) / (max_val - min_val)
    return np.clip(norm, 0, 1)


def start_annotation(h5_path):
    #识别是哪个站点
    file_stem = Path(h5_path).stem
    file_stem = int(file_stem)
    CH355max = normalize_range[file_stem]["CH355"]
    CH532max = normalize_range[file_stem]["CH532"]
    CH1064max = normalize_range[file_stem]["CH1064"]
    channel_norm_ranges = {
    'CH355': (0, CH355max),
    'CH532': (0, CH532max),
    'CH1064': (0, CH1064max),
    'PDR355': (0, 1),
    'PDR532': (0, 1)
    }
    # 1. 加载数据和 mask
    data_dict = load_lidar_channels(h5_path)
    
    lidar_cloud = None
    with h5py.File(Path(h5_path), 'r') as f:
        if 'lidar_cloud' in f:
            lidar_cloud = f['lidar_cloud'][:]
            height = f['height'][:]
            print("✅ Loaded lidar_cloud mask.")
        else:
            print("⚠️ No 'lidar_cloud' dataset found, will initialize blank mask.")

    # 2. 创建 Napari Viewer
    viewer = napari.Viewer()

    # 3. 加载图像通道
    for name, data in data_dict.items():
        '''if name == "CH532":
            profile= data[194*5]
            #可视化，y log
            plt.plot(profile)
            plt.yscale('log')
            plt.title('CH532 Profile')
            plt.xlabel('Range Index')
            plt.ylabel('Intensity')
            plt.show()'''
        
        if name[:2] == 'CH':
            data = data*height**2
        if data.ndim == 2:
            data = np.transpose(data, (1, 0))[::-1,:]
        
        min_val, max_val = channel_norm_ranges[name]
        data = np.nan_to_num(data, nan=-1)
        norm_data = normalize_to_unit_range(data, min_val, max_val)
        
        viewer.add_image(
            norm_data,
            name=name,
            colormap='gray',
            contrast_limits=(0, 1),
            blending='additive',
            visible=True
        )
    #额外可视化 CR1064_532
    '''if 'CH1064' in data_dict and 'CH532' in data_dict:
        ch1064 = data_dict['CH1064']
        ch532 = data_dict['CH532']
        if ch1064.ndim == 2:
            ch1064 = np.transpose(ch1064, (1, 0))[::-1,:]
        if ch532.ndim == 2:
            ch532 = np.transpose(ch532, (1, 0))[::-1,:]
        cr1064_532 = ch1064 / (ch532 + 1e-6)  # 避免除以0
        viewer.add_image(
            cr1064_532,
            name='CR1064_532',
            colormap='gray',
            contrast_limits=(0, 1),
            blending='additive',
            visible=True
        )'''


    # 4. 加载原始 mask（或初始化为0）
    if lidar_cloud is not None:
        lidar_cloud = np.transpose(lidar_cloud, (1, 0))[::-1,:]  # T-H -> H-T for napari
    else:
        shape = list(data_dict.values())[0].shape
        lidar_cloud = np.zeros(shape, dtype=np.uint8)
    # >0.3
    lidar_cloud[lidar_cloud > 0.3] = 1
    lidar_cloud = lidar_cloud.astype(np.uint8)
    viewer.add_labels(
        data=lidar_cloud,
        name="mask"
    )
    # 5. 保存键绑定
    
    from magicgui import magicgui
    from scipy.ndimage import gaussian_filter

    from scipy.ndimage import generic_filter
    from magicgui import magicgui

    @magicgui(
        auto_call=True,
        window_size={"label": "窗口大小", "min": 1, "max": 21, "step": 2},
        count_threshold={"label": "噪声像素个数阈值", "min": 1, "max": 100, "step": 1},
        intensity_threshold={"label": "强度阈值", "min": 0.0, "max": 1.0, "step": 0.01}
    )
    def noise_mask_widget(window_size=5, count_threshold=5, intensity_threshold=0.15):
        selected_layer = viewer.layers.selection.active
        if selected_layer is None or selected_layer.name == "mask":
            print("⚠️ 请选择图像通道进行噪声屏蔽")
            return
        
        image = selected_layer.data
        original = selected_layer.metadata.get('original_data')
        if original is None:
            selected_layer.metadata['original_data'] = image.copy()
            original = image

        # 定义滑窗函数
        def count_low_vals(values):
            return np.sum(values < intensity_threshold)

        # 应用 generic_filter（在每个窗口上运行 count_low_vals）
        footprint = np.ones((window_size, window_size))
        low_val_count = generic_filter(
            original, function=count_low_vals, size=window_size, mode='nearest'
        )

        # 判断噪声（low count >= n）
        noise_mask = (low_val_count >= count_threshold).astype(np.uint8)

        # 添加或更新图层
        if 'noise_mask' in viewer.layers:
            viewer.layers['noise_mask'].data = noise_mask
        else:
            viewer.add_labels(
                data=noise_mask,
                name='noise_mask',
                opacity=0.4,
                color={1: 'red'}
            )

    # 添加到右侧控件栏
    viewer.window.add_dock_widget(noise_mask_widget, area='right')

    # 假设 h5_path 是你读取的原始 HDF5 文件路径（字符串）

    @viewer.bind_key('s')
    def save_mask(viewer):
        mask_layer = viewer.layers['mask'] if 'mask' in viewer.layers else None
        if mask_layer is not None:
            # 构造基础输出路径
            input_path = Path(h5_path)
            base_output = Path(h5_path.replace('CMA', 'mask')).with_suffix('.tif')

            # 检查是否存在，自动生成不重复路径
            output_path = base_output
            counter = 1
            while output_path.exists():
                output_path = base_output.with_stem(base_output.stem + f"_v{counter}")
                counter += 1

            # 创建父目录
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存 mask 数据
            save_mask = mask_layer.data.astype(np.uint8)
            tifffile.imwrite(output_path, save_mask)
            print(f"✅ Mask saved to: {output_path}")
        else:
            print("⚠️ No mask layer named 'mask' found.")


    napari.run()

import sys
if __name__ == '__main__':
    '''if len(sys.argv) != 2:
        print("用法: python annotate_lidar.py <路径/文件名.h5>")
        sys.exit(1)

    h5_file = sys.argv[1]
    if not Path(h5_file).exists():
        print(f"❌ 指定的文件不存在: {h5_file}")
        sys.exit(1)'''
    
    h5_file=r"F:\Workspace\Projects\LidarCloud\CMA\2025\2025\0115\58847.h5"
    #h5_file=r"E:\Project\Lidar-cloud-CMA\result\2025\0101\51628.h5"
    start_annotation(h5_file)