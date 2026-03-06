[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spin-road-mapper-extracting-roads-from-aerial/road-segementation-on-deepglobe)](https://paperswithcode.com/sota/road-segementation-on-deepglobe?p=spin-road-mapper-extracting-roads-from-aerial)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spin-road-mapper-extracting-roads-from-aerial/road-segementation-on-massachusetts-roads)](https://paperswithcode.com/sota/road-segementation-on-massachusetts-roads?p=spin-road-mapper-extracting-roads-from-aerial)

## SPIN Road Mapper: Extracting Roads from Aerial Images via Spatial and Interaction Space Graph Reasoning for Autonomous Driving (ICRA'22)
[Wele Gedara Chaminda Bandara](https://www.wgcban.com/), [Jeya Maria Jose Valanarasu](https://jeya-maria-jose.github.io/research/), and [Vishal M. Patel](https://engineering.jhu.edu/vpatel36/sciencex_teams/vishalpatel/)

Read Paper: [Link](https://arxiv.org/abs/2109.07701)

## Windows + DeepGlobe 配置

当前仓库已按以下目标配置：
- 数据集根目录：`D:/project/pythonProject/Road_Identification/SAM2-UNet/deepglobe`
- 训练轮数：`80`
- 每轮验证一次
- `checkpoints/latest.pth.tar` 每轮覆盖保存
- `checkpoints/best.pth.tar` 保存验证集最高 `mIoU`
- 每 `5` 轮额外保存 `checkpoints/epoch_XXX.pth.tar`
- TensorBoard 日志目录：`logs/spin_时间戳`
- 日志记录：`loss`、`background_iou`、`road_iou`、`precision`、`recall`，且包含训练集和验证集

## 环境安装

建议使用 Python `3.10` 或 `3.11`。

1. 安装与你本机 CUDA 版本匹配的 PyTorch。
2. 安装项目其余依赖：

```powershell
pip install -r requirements-windows.txt
```

如果你还没安装 PyTorch，可先到官方命令生成页选择 Windows + CUDA：
`https://pytorch.org/get-started/locally/`

## 训练命令

```powershell
python train_mtl.py --config config.json --model_name StackHourglassNetMTL --dataset deepglobe --exp deepglobe_spin
```

## TensorBoard

```powershell
tensorboard --logdir logs
```

启动后在浏览器打开本机 TensorBoard 地址即可查看训练和验证曲线。
