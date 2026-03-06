import os
import cv2
import torch
import json
import shutil
from tqdm import tqdm
from road_dataset import MyRoadData
from model.models import MODELS
from torch.utils.data import DataLoader
import numpy as np

# 配置区域（根据实际情况修改）
WEIGHT_PATH = "myroadExp/expriment/checkpoint_epoch_060.pth.tar"
INPUT_DIR = "dataset/test_min/"
OUTPUT_DIR = "output/"
MODEL_NAME = "StackHourglassNetMTL"
MASK_DIR = "dataset/test/seg/"


def load_model():
    """加载训练好的模型"""
    checkpoint = torch.load(WEIGHT_PATH)
    config = checkpoint.get('config') or json.load(open("config.json"))

    model = MODELS[MODEL_NAME](
        config["task1_classes"],
        config["task2_classes"],
        **config.get("model_kwargs", {})
    )

    state_dict = checkpoint['state_dict']
    # 处理多GPU训练保存的模型
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    return config, model


class PredictDataset:
    """自定义预测数据集"""

    def __init__(self, config):
        ds_cfg = config["val_dataset"]["myroad"]
        self.img_dir = INPUT_DIR
        self.image_suffix = ds_cfg["image_suffix"]
        self.images = [f for f in os.listdir(INPUT_DIR) if f.endswith(self.image_suffix)]

        # 新增配置参数
        self.normalize_type = config["val_dataset"]["normalize_type"]
        self.mean = np.array(eval(config["val_dataset"]["mean"]))
        self.std = np.array(eval(config["val_dataset"]["std"]))

        if not self.images:
            raise FileNotFoundError(f"未找到{self.image_suffix}文件")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = cv2.imread(img_path).astype(np.float32)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)

        # 修改后的标准化处理
        if self.normalize_type == "Std":
            image = (image - self.mean) / self.std
        elif self.normalize_type == "Mean":
            image -= self.mean
        else:
            image = image / 255.0 * 2 - 1  # 默认归一化到[-1,1]

        image = image.transpose(2, 0, 1)
        return torch.FloatTensor(image), self.images[idx]


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载模型和配置
    config, model = load_model()
    dataset = PredictDataset(config)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    print(f"🔍 发现 {len(dataset)} 张待预测图像")

    # 预测流程
    with torch.no_grad():
        for images, filenames in tqdm(loader, desc="🚀 预测进度", unit="batch"):
            outputs = model(images.cuda())

            # 调试输出（需要处理元组类型）
            print("\nModel output type:", type(outputs))
            print("Output tuple length:", len(outputs))

            # 处理多任务输出（假设第一个元素是分割输出）
            if isinstance(outputs, tuple):
                print("检测到多任务输出，解析结构：")
                # 假设分割输出是第一个元素，且是张量
                seg_output = outputs[0]
                # 打印分割输出的统计信息
                print("Segmentation output shape:", seg_output.shape)
                print("Min:", seg_output.min().item())
                print("Max:", seg_output.max().item())
                print("Mean:", seg_output.mean().item())

                # 如果有其他输出（如角度预测）
                if len(outputs) > 1:
                    angle_output = outputs[1]
                    print("Angle output shape:", angle_output.shape)
            else:
                seg_output = outputs
                print("单一输出模式")
                print("Min:", seg_output.min().item())
                print("Max:", seg_output.max().item())
                print("Mean:", seg_output.mean().item())
            # 修改后的输出处理逻辑
            if isinstance(outputs, (list, tuple)):
                # 假设分割输出是最后一个元素（多尺度预测常见模式）
                seg_output = outputs[-1]
                while isinstance(seg_output, (list, tuple)):
                    seg_output = seg_output[-1]
            else:
                seg_output = outputs
            # 确保最终得到的是张量
            assert isinstance(seg_output, torch.Tensor), "输出解析错误，最终结果不是张量"

            # 后续处理保持不变...
            preds = torch.argmax(seg_output, dim=1).cpu().numpy().astype(np.uint8)


            # 保存结果
            for i in range(preds.shape[0]):
                base_name = filenames[i].replace("_sat.jpg", "")

                # 保存预测结果
                cv2.imwrite(
                    os.path.join(OUTPUT_DIR, f"{base_name}_predict.png"),
                    preds[i] * 255
                )

                # 复制原图
                src_img = os.path.join(INPUT_DIR, filenames[i])
                dst_img = os.path.join(OUTPUT_DIR, f"{base_name}_sat.jpg")
                if not os.path.exists(dst_img):
                    shutil.copy(src_img, dst_img)

                # 复制标注文件（如果存在）
                mask_src = os.path.join(MASK_DIR, f"{base_name}_mask.png")
                mask_dst = os.path.join(OUTPUT_DIR, f"{base_name}_mask.png")
                if os.path.exists(mask_src):
                    shutil.copy(mask_src, mask_dst)

    print(f"✅ 预测完成！结果保存在：{os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        raise NotADirectoryError(f"输入目录 {INPUT_DIR} 不存在")
    main()
