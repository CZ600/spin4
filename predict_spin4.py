#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPIN4模型预测脚本
参考DinoUnet预测格式，生成predict和visual两类图像
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, List, Tuple
import time
import json

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, '/root/autodl-tmp/spin4')

from road_dataset import MyRoadData, resolve_path
from model.models import MODELS


class SPIN4Predictor:
    """SPIN4模型预测器"""

    def __init__(self,
                 model_path: str,
                 config_path: str = 'config.json',
                 device: str = 'auto',
                 confidence_threshold: float = 0.5):
        """
        Args:
            model_path: 模型检查点路径
            config_path: 配置文件路径
            device: 推理设备
            confidence_threshold: 置信度阈值
        """
        self.model_path = model_path
        self.config_path = config_path
        self.confidence_threshold = confidence_threshold

        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 加载配置和模型
        self.config = self._load_config()
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ 模型加载成功: {model_path}")
        print(f"📱 设备: {self.device}")

    def _load_config(self) -> Dict:
        """加载配置文件"""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        return config

    def _load_model(self):
        """加载训练好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"权重文件 {self.model_path} 不存在")

        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        # 获取模型配置
        config = checkpoint.get('config', None) or self.config
        model_name = config.get('model_name', 'StackHourglassNetMTL')
        
        # 创建模型
        model = MODELS[model_name](
            config["task1_classes"],
            config["task2_classes"],
            **config.get("model_kwargs", {})
        )
        
        # 加载权重
        state_dict = checkpoint['state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        
        return model

    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """预处理图像"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        orig_h, orig_w = image.shape[:2]
        
        # 转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 归一化
        mean = np.array([70.95016901, 71.16398124, 71.30953645], dtype=np.float32)
        image -= mean
        
        # 转换为tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return image, (orig_h, orig_w)

    def predict_single_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """对单张图像进行预测"""
        # 预处理
        input_tensor, (orig_h, orig_w) = self.preprocess_image(image_path)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # 处理多任务输出
            if isinstance(outputs, (list, tuple)):
                seg_output = outputs[-1] if len(outputs) == 2 else outputs[0]
                while isinstance(seg_output, (list, tuple)):
                    seg_output = seg_output[-1]
            else:
                seg_output = outputs
            
            # 获取道路类别的概率
            prob = torch.softmax(seg_output, dim=1)
            road_prob = prob[0, 1].cpu().numpy()  # 假设类别1是道路
            
            # 调整大小到原始尺寸
            road_prob = cv2.resize(road_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # 转换为0-255范围
            prob_map_uint8 = (road_prob * 255).astype(np.uint8)
            
            # 二值化
            binary_map = (prob_map_uint8 > self.confidence_threshold * 255).astype(np.uint8) * 255
            
        return prob_map_uint8, binary_map

    def create_visualization(self,
                            original_image: np.ndarray,
                            prob_map: np.ndarray,
                            binary_map: np.ndarray,
                            mask_image: np.ndarray = None) -> np.ndarray:
        """创建可视化结果 - 参考DinoUnet格式，五图拼接"""
        # 调整图像大小一致
        target_size = (512, 512)
        
        # 处理原图
        if len(original_image.shape) == 3:
            original_resized = cv2.resize(original_image, target_size)
        else:
            original_resized = cv2.cvtColor(cv2.resize(original_image, target_size), cv2.COLOR_GRAY2RGB)
        
        # 处理概率图 - 应用colormap
        prob_colored = cv2.applyColorMap(cv2.resize(prob_map, target_size), cv2.COLORMAP_JET)
        
        # 处理二值图
        binary_resized = cv2.resize(binary_map, target_size)
        binary_colored = cv2.cvtColor(binary_resized, cv2.COLOR_GRAY2RGB)
        
        # 创建概率图叠加
        alpha = 0.6
        prob_overlay = cv2.addWeighted(original_resized, 1-alpha, prob_colored, alpha, 0)
        
        # 处理标签图
        if mask_image is not None:
            if len(mask_image.shape) == 3:
                mask_resized = cv2.resize(mask_image, target_size)
            else:
                mask_resized = cv2.cvtColor(cv2.resize(mask_image, target_size), cv2.COLOR_GRAY2RGB)
            images = [original_resized, prob_colored, binary_colored, prob_overlay, mask_resized]
            titles = ['Original', 'Probability Map', 'Binary Map', 'Probability Overlay', 'Ground Truth']
        else:
            images = [original_resized, prob_colored, binary_colored, prob_overlay]
            titles = ['Original', 'Probability Map', 'Binary Map', 'Probability Overlay']
        
        # 创建水平拼接图
        h_space = 20  # 图像间距
        total_width = len(images) * target_size[0] + (len(images) - 1) * h_space
        total_height = target_size[1] + 60  # 额外空间放标题
        
        # 创建白色背景
        result = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
        
        # 拼接每张图
        for i, (img, title) in enumerate(zip(images, titles)):
            x_offset = i * (target_size[0] + h_space)
            
            # 放置图像
            result[40:40+target_size[1], x_offset:x_offset+target_size[0]] = img
            
            # 添加标题（使用简单的文字）
            cv2.putText(result, title, (x_offset + target_size[0]//2 - len(title)*10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return result

    def predict_batch(self,
                     input_dir: str,
                     output_dir: str,
                     mask_dir: str = None,
                     create_visuals: bool = True) -> None:
        """批量预测"""
        
        # 创建输出目录
        predict_dir = os.path.join(output_dir, 'predict')
        visual_dir = os.path.join(output_dir, 'visual')
        
        os.makedirs(predict_dir, exist_ok=True)
        os.makedirs(visual_dir, exist_ok=True)
        
        # 获取所有测试图像
        image_files = [f for f in os.listdir(input_dir) if f.endswith('_sat.jpg')]
        image_files.sort()
        
        print(f"📁 找到 {len(image_files)} 张测试图像")
        
        # 处理每张图像
        for i, image_file in enumerate(tqdm(image_files, desc="预测进度")):
            try:
                # 构建路径
                image_path = os.path.join(input_dir, image_file)
                
                # 生成基础文件名
                base_name = image_file.replace('_sat.jpg', '')
                
                # 生成输出文件名
                prob_filename = f"{base_name}_prob.png"
                pred_filename = f"{base_name}_pred.png"
                visual_filename = f"{base_name}_visual.png"
                
                # 预测
                prob_map, binary_map = self.predict_single_image(image_path)
                
                # 保存概率图
                prob_path = os.path.join(predict_dir, prob_filename)
                cv2.imwrite(prob_path, prob_map)
                
                # 保存预测结果图（pred）
                # 注意：二值图中255表示道路，0表示背景
                pred_path = os.path.join(predict_dir, pred_filename)
                cv2.imwrite(pred_path, binary_map)
                
                # 创建可视化
                if create_visuals:
                    # 读取原图
                    original_image = cv2.imread(image_path)
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                    
                    # 尝试读取标签图
                    mask_image = None
                    if mask_dir:
                        mask_path = os.path.join(mask_dir, f"{base_name}_mask.png")
                        if os.path.exists(mask_path):
                            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2RGB)
                    
                    # 创建可视化图像
                    visual_result = self.create_visualization(
                        original_image, prob_map, binary_map, mask_image
                    )
                    
                    # 保存可视化结果
                    visual_path = os.path.join(visual_dir, visual_filename)
                    visual_bgr = cv2.cvtColor(visual_result, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(visual_path, visual_bgr)
                
                if (i + 1) % 100 == 0:
                    print(f"✅ 已完成 {i + 1}/{len(image_files)} 张图像")
                
            except Exception as e:
                print(f"❌ 处理 {image_file} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"🎉 预测完成！结果保存在 {output_dir}")
        print(f"📊 预测结果: {predict_dir}")
        if create_visuals:
            print(f"🎨 可视化结果: {visual_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SPIN4模型预测脚本')
    parser.add_argument('--model_path', type=str,
                       default='/root/autodl-tmp/spin4/checkpoints/best.pth.tar',
                       help='模型权重路径')
    parser.add_argument('--config_path', type=str,
                       default='config.json',
                       help='配置文件路径')
    parser.add_argument('--input_dir', type=str,
                       default='dataset/test/data',
                       help='输入图像目录')
    parser.add_argument('--mask_dir', type=str,
                       default='dataset/test/seg',
                       help='标签目录')
    parser.add_argument('--output_dir', type=str,
                       default='output',
                       help='输出结果目录')
    parser.add_argument('--device', type=str, default='auto',
                       help='推理设备 (auto/cpu/cuda)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='置信度阈值')
    parser.add_argument('--no_visual', action='store_true',
                       help='不创建可视化结果')

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("🚀 开始SPIN4模型预测")
    print(f"📂 输入目录: {args.input_dir}")
    print(f"📂 标签目录: {args.mask_dir}")
    print(f"💾 输出目录: {args.output_dir}")
    print(f"🎯 模型: {args.model_path}")
    print(f"📊 阈值: {args.confidence_threshold}")
    print("=" * 50)

    # 创建预测器
    predictor = SPIN4Predictor(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device,
        confidence_threshold=args.confidence_threshold
    )

    # 开始预测
    start_time = time.time()
    predictor.predict_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mask_dir=args.mask_dir,
        create_visuals=not args.no_visual
    )
    end_time = time.time()

    print(f"⏱️ 总耗时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
