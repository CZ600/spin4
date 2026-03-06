#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPIN4模型预测脚本 - 修正版
使用任务1的输出（2类分割）而不是任务2的输出
"""

import os
import sys
import json
import torch
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, '/root/autodl-tmp/spin4')

from model.models import MODELS

# 配置
WEIGHT_PATH = '/root/autodl-tmp/spin4/checkpoints/best.pth.tar'
INPUT_DIR = '/root/autodl-fs/Deepglobe/deepglobe/test/data'
MASK_DIR = '/root/autodl-fs/Deepglobe/deepglobe/test/seg'
OUTPUT_DIR = '/root/autodl-tmp/spin4/output'
CONFIG_PATH = '/root/autodl-tmp/spin4/config.json'

CONFIDENCE_THRESHOLD = 0.5

def load_model():
    """加载模型"""
    print(f"📂 加载模型: {WEIGHT_PATH}")
    
    checkpoint = torch.load(WEIGHT_PATH, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', None) or json.load(open(CONFIG_PATH))
    model_name = config.get('model_name', 'StackHourglassNetMTL')
    
    model = MODELS[model_name](
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
    
    print(f"✅ 模型加载成功")
    return model, config

def preprocess_image(image_path, config):
    """预处理图像"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    orig_h, orig_w = image.shape[:2]
    
    # 归一化
    mean = np.array([70.95016901, 71.16398124, 71.30953645], dtype=np.float32)
    image = image.astype(np.float32) - mean
    
    # 转换为tensor
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    
    return image, (orig_h, orig_w)

def predict_image(model, image_path, config):
    """预测单张图像 - 使用任务1的输出"""
    image, (orig_h, orig_w) = preprocess_image(image_path, config)
    image = image.cuda()
    
    with torch.no_grad():
        outputs = model(image)
        
        # 使用任务1的输出（道路分割，2类）
        # outputs[0] 是任务1的输出，是一个列表
        # outputs[0][3] 是1024x1024分辨率的输出，形状是 [1, 2, 1024, 1024]
        seg_output = outputs[0][3]  # [1, 2, 1024, 1024]
        
        # 获取道路类别的概率（类别1是道路）
        prob = torch.softmax(seg_output, dim=1)
        road_prob = prob[0, 1].cpu().numpy()  # 道路类别的概率
        
        # 调整大小到原始尺寸
        road_prob = cv2.resize(road_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
        # 转换为0-255范围
        prob_map_uint8 = (road_prob * 255).astype(np.uint8)
        
        # 二值化 - 255表示道路，0表示背景
        binary_map = (prob_map_uint8 > CONFIDENCE_THRESHOLD * 255).astype(np.uint8) * 255
        
    return prob_map_uint8, binary_map

def create_visualization(original_image, prob_map, binary_map, mask_image=None):
    """创建五图拼接可视化"""
    target_size = (512, 512)
    
    # 处理原图
    if len(original_image.shape) == 3:
        original_resized = cv2.resize(original_image, target_size)
    else:
        original_resized = cv2.cvtColor(cv2.resize(original_image, target_size), cv2.COLOR_GRAY2RGB)
    
    # 处理概率图
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
        titles = ['Original', 'Probability', 'Prediction', 'Overlay', 'Ground Truth']
    else:
        images = [original_resized, prob_colored, binary_colored, prob_overlay]
        titles = ['Original', 'Probability', 'Prediction', 'Overlay']
    
    # 创建水平拼接图
    h_space = 20
    total_width = len(images) * target_size[0] + (len(images) - 1) * h_space
    total_height = target_size[1] + 60
    
    result = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    
    for i, (img, title) in enumerate(zip(images, titles)):
        x_offset = i * (target_size[0] + h_space)
        result[40:40+target_size[1], x_offset:x_offset+target_size[0]] = img
        cv2.putText(result, title, (x_offset + target_size[0]//2 - len(title)*8, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return result

def main():
    print("🚀 开始SPIN4模型预测 - 修正版")
    print(f"📂 输入目录: {INPUT_DIR}")
    print(f"📂 标签目录: {MASK_DIR}")
    print(f"💾 输出目录: {OUTPUT_DIR}")
    print(f"🎯 模型: {WEIGHT_PATH}")
    print("=" * 50)
    
    # 加载模型
    model, config = load_model()
    
    # 创建输出目录
    predict_dir = os.path.join(OUTPUT_DIR, 'predict')
    visual_dir = os.path.join(OUTPUT_DIR, 'visual')
    os.makedirs(predict_dir, exist_ok=True)
    os.makedirs(visual_dir, exist_ok=True)
    
    # 获取所有测试图像
    image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('_sat.jpg')]
    image_files.sort()
    
    print(f"📁 找到 {len(image_files)} 张测试图像")
    
    # 处理每张图像
    for i, image_file in enumerate(tqdm(image_files, desc="预测进度")):
        try:
            image_path = os.path.join(INPUT_DIR, image_file)
            base_name = image_file.replace('_sat.jpg', '')
            
            # 预测
            prob_map, binary_map = predict_image(model, image_path, config)
            
            # 保存prob和pred
            cv2.imwrite(os.path.join(predict_dir, f"{base_name}_prob.png"), prob_map)
            cv2.imwrite(os.path.join(predict_dir, f"{base_name}_pred.png"), binary_map)
            
            # 创建可视化
            original = cv2.imread(image_path)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            
            mask = None
            mask_path = os.path.join(MASK_DIR, f"{base_name}_mask.png")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            
            visual = create_visualization(original, prob_map, binary_map, mask)
            cv2.imwrite(os.path.join(visual_dir, f"{base_name}_visual.png"), 
                       cv2.cvtColor(visual, cv2.COLOR_RGB2BGR))
            
            if (i + 1) % 50 == 0:
                print(f"✅ 已完成 {i + 1}/{len(image_files)} 张图像")
                
        except Exception as e:
            print(f"❌ 处理 {image_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n🎉 预测完成！")
    print(f"📊 预测结果: {predict_dir}")
    print(f"🎨 可视化结果: {visual_dir}")

if __name__ == "__main__":
    main()
