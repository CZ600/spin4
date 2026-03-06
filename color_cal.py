import numpy as np
from PIL import Image


def analyze_pixel_distribution(image_path):
    """
    分析图像的像素值分布比例
    支持灰度图和彩色图（RGB模式）
    返回格式：
        - 灰度图返回 { "Gray": { 像素值: 百分比 } }
        - 彩色图返回 { "Red": { ... }, "Green": { ... }, "Blue": { ... } }
    """
    # 读取图像并自动处理模式
    img = Image.open(image_path)

    # 处理不同图像模式
    if img.mode == 'L':
        # 灰度图像处理
        img_array = np.array(img)
        pixel_counts = np.bincount(img_array.flatten(), minlength=256)
    elif img.mode in ['RGB', 'RGBA']:
        # 彩色图像处理（自动转换为RGB三通道）
        img = img.convert('RGB')
        img_array = np.array(img)
        pixel_counts = [
            np.bincount(img_array[:, :, i].flatten(), minlength=256)
            for i in range(3)
        ]
    else:
        raise ValueError("Unsupported image mode. Please convert to RGB or Grayscale.")

    # 计算百分比
    total_pixels = img_array.shape[0] * img_array.shape[1]
    results = {}

    if img.mode == 'L':
        percentages = (pixel_counts / total_pixels * 100).round(4)
        results["Gray"] = {
            val: percent
            for val, percent in enumerate(percentages)
            if percent > 0  # 过滤零值
        }
    else:
        channels = ['Red', 'Green', 'Blue']
        for i in range(3):
            percentages = (pixel_counts[i] / total_pixels * 100).round(4)
            results[channels[i]] = {
                val: percent
                for val, percent in enumerate(percentages)
                if percent > 0  # 过滤零值
            }

    return results


# 使用示例
if __name__ == "__main__":
    distribution = analyze_pixel_distribution(r"D:\project\pythonProject\Road_Identification\spin4\output\00037_pred.png")

    for channel, values in distribution.items():
        print(f"=== {channel} Channel ===")
        print(f"Unique values: {len(values)}")
        print("Top 5 frequent values:")
        for val, percent in sorted(values.items(), key=lambda x: -x[1])[:5]:
            print(f"  {val}: {percent}%")
        print()
