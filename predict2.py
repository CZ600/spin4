import os
import cv2
import numpy as np
import torch
import json
from road_dataset import MyRoadData
from model.models import MODELS
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.loss import CrossEntropyLoss2d, mIoULoss
from utils import util
from utils import viz_util

# 配置区域
WEIGHT_PATH = r"D:\project\pythonProject\Road_Identification\spin4\myroadExp\expriment\checkpoint_epoch_060.pth.tar"
INPUT_DIR = r"dataset/test_min/"
OUTPUT_DIR = "output/"
MODEL_NAME = "StackHourglassNetMTL"
WINDOW_SIZE = 512
STRIDE = 256


def sliding_windows(image_size):
    """生成滑动窗口坐标"""
    positions = []
    for y in range(0, image_size, STRIDE):
        for x in range(0, image_size, STRIDE):
            x_end = min(x + WINDOW_SIZE, image_size)
            y_end = min(y + WINDOW_SIZE, image_size)
            positions.append((x, y, x_end, y_end))
    return positions


def merge_predictions(full_mask, window_mask, coords, overlap_logic='or'):
    """合并预测结果"""
    x1, y1, x2, y2 = coords
    window_area = full_mask[y1:y2, x1:x2]

    if overlap_logic == 'or':
        merged = np.logical_or(window_area, window_mask)
    else:  # 其他合并逻辑可根据需要扩展
        merged = np.maximum(window_area, window_mask)

    full_mask[y1:y2, x1:x2] = merged
    return full_mask


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
    """预测数据集（支持滑动窗口）"""

    def __init__(self, config):
        ds_cfg = config["val_dataset"]["myroad"]
        self.img_dir = INPUT_DIR
        self.image_suffix = ds_cfg["image_suffix"]
        self.mean = np.array(eval(config["val_dataset"]["mean"]))
        self.std = np.array(eval(config["val_dataset"]["std"]))
        self.normalize_type = config["val_dataset"]["normalize_type"]

        self.images = [f for f in os.listdir(self.img_dir)
                       if f.endswith(self.image_suffix)]
        if not self.images:
            raise FileNotFoundError(f"{self.img_dir} 中没有找到{self.image_suffix}文件")

    def __len__(self):
        return len(self.images)

    def preprocess(self, image):
        """图像预处理"""
        if self.normalize_type == "Std":
            image = (image - self.mean) / (3 * self.std)
        elif self.normalize_type == "Mean":
            image -= self.mean
        else:
            image = (image / 255.0) * 2 - 1
        return image.transpose(2, 0, 1)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = cv2.imread(img_path).astype(np.float32)

        # 生成滑动窗口
        h, w = image.shape[:2]
        windows = sliding_windows(h)
        patches = []
        coords = []

        for x1, y1, x2, y2 in windows:
            patch = image[y1:y2, x1:x2]
            patch = cv2.resize(patch, (WINDOW_SIZE, WINDOW_SIZE))
            patch = self.preprocess(patch)
            patches.append(patch)
            coords.append((x1, y1, x2, y2))

        return (
            torch.FloatTensor(np.array(patches)),
            self.images[idx],
            np.array(coords),
            image.astype(np.uint8)  # 原始图像用于拼接
        )


# 修改predict.py中的结果保存部分
def savePredicted(real_img, gt_img, pred_mask, pred_affinity, save_path, norm_type="Mean"):
    """
    四宫格拼接函数
    参数均为numpy数组:
    real_img: HxWx3
    gt_img: HxW
    pred_mask: HxW
    pred_affinity: HxW
    """
    # 颜色转换
    mean_bgr = np.array([70.95016901, 71.16398124, 71.30953645])
    std_bgr = np.array([34.00087859, 35.18201658, 36.40463264])

    # 反归一化原图
    if norm_type == "Mean":
        real_img = real_img.astype(float) + mean_bgr
    elif norm_type == "Std":
        real_img = (real_img.astype(float) * std_bgr) + mean_bgr
    real_img = np.clip(real_img, 0, 255).astype(np.uint8)

    # 处理标注图
    gt_viz = cv2.cvtColor(gt_img.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)

    # 处理预测结果
    pred_viz = cv2.cvtColor(pred_mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)

    # 处理方向预测图
    affinity_viz = cv2.applyColorMap(pred_affinity.astype(np.uint8), cv2.COLORMAP_HSV)

    # 水平拼接
    top_row = np.concatenate([real_img, gt_viz], axis=1)
    bottom_row = np.concatenate([pred_viz, affinity_viz], axis=1)

    # 垂直拼接
    final_img = np.concatenate([top_row, bottom_row], axis=0)
    cv2.imwrite(save_path, final_img)





def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config, model = load_model()
    dataset = PredictDataset(config)

    print(f"开始预测，共发现 {len(dataset)} 张图片")
    #
    with torch.no_grad():
        for patches, filename, coords, orig_image in tqdm(dataset, desc="处理图像"):
            # 初始化全尺寸mask
            full_mask = np.zeros((1024, 1024), dtype=np.uint8)
            full_affinity = np.zeros((1024, 1024), dtype=np.uint8)

            # 批量预测
            batch_size = 8
            for i in range(0, len(patches), batch_size):
                batch = patches[i:i + batch_size].cuda()

                # 模型预测
                outputs = model(batch)

                # 解析多任务输出
                if isinstance(outputs, (list, tuple)):
                    seg_output = outputs[0][-1] if len(outputs[0]) > 1 else outputs[0]
                    affinity_output = outputs[1][-1] if len(outputs[1]) > 1 else outputs[1]
                else:
                    seg_output = outputs
                    affinity_output = None

                # 处理分割结果
                seg_preds = torch.argmax(seg_output, dim=1).cpu().numpy().astype(np.uint8)

                # 处理方向预测
                if affinity_output is not None:
                    affinity_preds = torch.argmax(affinity_output, dim=1).cpu().numpy().astype(np.uint8)
                else:
                    affinity_preds = [None] * len(seg_preds)

                # 合并结果
                for j in range(len(seg_preds)):
                    idx = i + j
                    if idx >= len(coords):
                        break

                    x1, y1, x2, y2 = coords[idx]

                    # 调整预测结果尺寸
                    mask = cv2.resize(seg_preds[j], (x2 - x1, y2 - y1),
                                      interpolation=cv2.INTER_NEAREST)
                    full_mask = merge_predictions(full_mask, mask, coords[idx])

                    if affinity_preds[j] is not None:
                        aff = cv2.resize(affinity_preds[j], (x2 - x1, y2 - y1),
                                         interpolation=cv2.INTER_NEAREST)
                        full_affinity = merge_predictions(full_affinity, aff, coords[idx])

            # 生成拼接图像
            base_name = os.path.splitext(filename)[0]
            save_path = os.path.join(OUTPUT_DIR, f"{base_name}_pred.png")

            # 假设的标注图路径（根据实际情况修改）
            gt_path = os.path.join("dataset/val/seg/", f"{base_name}_mask.png")
            gt_image = cv2.imread(gt_path, 0) if os.path.exists(gt_path) else np.zeros_like(full_mask)

            # 生成可视化结果
            # viz_image = util.savePredictedProb(
            #     orig_image,  # 原图
            #     gt_image,  # 标注图
            #     full_mask,  # 预测结果
            #     full_affinity,  # 方向预测
            #     image_name=save_path,
            #     norm_type=config["val_dataset"]["normalize_type"]
            # )
            # 在main函数中调用
            savePredicted(
                orig_image,  # 原图 numpy数组 [H,W,3]
                gt_image,  # 标注图 numpy数组 [H,W]
                full_mask,  # 预测结果 numpy数组 [H,W]
                full_affinity,  # 方向预测 numpy数组 [H,W]
                save_path=save_path,
                norm_type=config["val_dataset"]["normalize_type"]
            )


if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        raise NotADirectoryError(f"输入目录 {INPUT_DIR} 不存在")
    main()
