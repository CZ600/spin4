import os
import cv2
import numpy as np
import torch
import json
from road_dataset import MyRoadData
from model.models import MODELS
from torch.utils.data import DataLoader
import tqdm
import shutil

# #################### 配置区域 ####################
WEIGHT_PATH = r"D:\project\pythonProject\Road_Identification\spin4\myroadExp\expriment\checkpoint_epoch_060.pth.tar"
INPUT_DIR = r"dataset/test/data/"
OUTPUT_DIR = "output/"
MODEL_NAME = "StackHourglassNetMTL"
MASK_DIR = r"dataset/test/seg/"

# 滑动窗口配置
WINDOW_SIZE = 512
STRIDE = 256
SIGMA = 0.3  # 高斯权重衰减系数（越小边缘衰减越快）
THRESHOLD = 0.5  # 概率阈值


# #################################################

def load_model():
    """加载模型和配置"""
    if not os.path.exists(WEIGHT_PATH):
        raise FileNotFoundError(f"权重文件 {WEIGHT_PATH} 不存在")

    checkpoint = torch.load(WEIGHT_PATH)
    config = checkpoint.get('config', None) or json.load(open("config.json"))

    model = MODELS[MODEL_NAME](
        config["task1_classes"],
        config["task2_classes"],
        **config.get("model_kwargs", {})
    )

    state_dict = checkpoint['state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    return config, model


class PredictDataset:
    """预测数据集（返回原始尺寸图像）"""

    def __init__(self, config):
        ds_cfg = config["val_dataset"]["myroad"]
        self.img_dir = INPUT_DIR
        self.mean = np.array(eval(config["val_dataset"]["mean"]))
        self.std = np.array(eval(config["val_dataset"]["std"]))
        self.normalize_type = config["val_dataset"]["normalize_type"]
        self.image_suffix = ds_cfg["image_suffix"]

        self.images = [
            f for f in os.listdir(self.img_dir)
            if f.endswith(self.image_suffix)
        ]
        if not self.images:
            raise FileNotFoundError(f"{self.img_dir} 中没有找到{self.image_suffix}文件")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = cv2.imread(img_path).astype(np.float32)

        # 保持原始尺寸
        if self.normalize_type == "Std":
            image = (image - self.mean) / (3 * self.std)
        elif self.normalize_type == "Mean":
            image -= self.mean
        else:
            image = (image / 255.0) * 2 - 1

        image = image.transpose(2, 0, 1)
        return torch.FloatTensor(image), self.images[idx]


def binarize_array(arr):
    binary_arr = np.full_like(arr, 255, dtype=np.uint8)
    binary_arr[arr == 36] = 0
    return binary_arr


def create_gaussian_weight(size, sigma=1.0):
    """创建高斯权重矩阵"""
    center = size // 2
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    weight = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * (sigma * size) ** 2))
    weight /= weight.max()  # 归一化到0-1
    return weight


def sliding_window_prediction(model, image_tensor, image_size=1024):
    """改进的滑动窗口预测函数"""
    window_size = WINDOW_SIZE
    stride = STRIDE
    height, width = image_size, image_size

    y_starts = list(range(0, height - window_size + 1, stride))
    x_starts = list(range(0, width - window_size + 1, stride))

    # 处理不能整除的情况
    if (height - window_size) % stride != 0:
        y_starts.append(height - window_size)
    if (width - window_size) % stride != 0:
        x_starts.append(width - window_size)

    # 创建高斯权重矩阵
    weight_matrix = create_gaussian_weight(window_size, sigma=SIGMA)

    # 初始化全局概率图和权重图
    global_prob = np.zeros((height, width), dtype=np.float32)
    global_weight = np.zeros((height, width), dtype=np.float32)

    with torch.no_grad():
        for y in y_starts:
            for x in x_starts:
                # 截取窗口
                window = image_tensor[:, y:y + window_size, x:x + window_size].unsqueeze(0).cuda()

                # 模型预测
                outputs = model(window)

                # 处理多任务输出结构
                if isinstance(outputs, (list, tuple)):
                    seg_output = outputs[-1] if len(outputs) == 2 else outputs[0]
                    while isinstance(seg_output, (list, tuple)):
                        seg_output = seg_output[-1]
                else:
                    seg_output = outputs

                # 获取类别36的概率图
                prob = torch.softmax(seg_output, dim=1)
                class36_prob = prob[:, 36, :, :].cpu().numpy().squeeze()  # (512, 512)

                # 应用权重矩阵
                weighted_prob = class36_prob * weight_matrix

                # 累加到全局
                global_prob[y:y + window_size, x:x + window_size] += weighted_prob
                global_weight[y:y + window_size, x:x + window_size] += weight_matrix

        # 计算加权平均概率
        global_prob /= global_weight

        # 生成最终mask（0: 道路，255: 背景）
        final_mask = np.full((height, width), 255, dtype=np.uint8)
        final_mask[global_prob > THRESHOLD] = 0

    return final_mask


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config, model = load_model()
    dataset = PredictDataset(config)
    loader = DataLoader(dataset, batch_size=18, shuffle=False)

    print(f"开始预测，共发现 {len(dataset)} 张图片")
    for images, filenames in tqdm.tqdm(loader, desc='预测进度'):
        # 获取原始尺寸图像
        image_tensor = images[0]  # shape: [3, H, W]
        filename = filenames[0]

        # 滑动窗口预测
        final_mask = sliding_window_prediction(model, image_tensor)

        # 保存结果
        original_name = os.path.splitext(filename)[0]
        filename_base = original_name.split("_")[0]

        # 保存预测mask
        save_path = os.path.join(OUTPUT_DIR, f"{filename_base}_pred.png")
        cv2.imwrite(save_path, final_mask)

        # 复制原图
        src_img = os.path.join(INPUT_DIR, filename)
        dst_img = os.path.join(OUTPUT_DIR, f"{filename_base}_sat.jpg")
        if not os.path.exists(dst_img):
            shutil.copy(src_img, dst_img)

        # 复制标注文件（如果存在）
        mask_src = os.path.join(MASK_DIR, f"{filename_base}_mask.png")
        mask_dst = os.path.join(OUTPUT_DIR, f"{filename_base}_mask.png")
        if os.path.exists(mask_src):
            shutil.copy(mask_src, mask_dst)

    print(f"预测完成，结果保存在: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        raise NotADirectoryError(f"输入目录 {INPUT_DIR} 不存在")
    main()