import collections
import math
import os
import random

import cv2
import numpy as np
import torch
from data_utils import affinity_utils
from torch.utils import data
import os

# 设置工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
workspace_dir = os.getcwd()


class RoadDataset(data.Dataset):
    def __init__(
            self, config, dataset_name, seed=7, multi_scale_pred=True, is_train=True
    ):
        # Seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        self.split = "train" if is_train else "val"
        self.config = config
        # paths
        self.dir = self.config[dataset_name]["dir"]

        if dataset_name == "myroad":
            # 基础路径
            base_path = os.path.abspath(
                os.path.join(
                    workspace_dir,
                    self.dir,
                    self.config[dataset_name]['set']
                )
            )
            self.img_root = os.path.join(base_path, "data/")
            self.gt_root = os.path.join(base_path, "seg/")

            # 获取原始图像文件名列表（带后缀）
            raw_img_files = [
                f for f in os.listdir(self.img_root)
                if f.endswith(self.config[dataset_name]["image_suffix"])
            ]

            # 新增训练集采样逻辑
            if is_train:  # 只在训练时进行采样
                # 计算30%样本量（至少保留1个样本）
                total_samples = len(raw_img_files)
                sample_size = max(1, int(total_samples * 0.3))

                # 保证可重复性的采样
                random.seed(seed)
                raw_img_files = random.sample(raw_img_files, sample_size)

                # 打印采样信息（调试用）
                print(f"Sampled {sample_size}/{total_samples} training samples for myroad")

            # 动态构建 image_list
            self.image_list = []
            for img_file in raw_img_files:
                # 提取基础文件名（去除图像后缀）
                base_name = img_file.replace(
                    self.config[dataset_name]["image_suffix"], ""
                )

                # 生成标注文件名
                gt_file = base_name + self.config[dataset_name]["gt_suffix"]

                # 构建完整路径
                img_path = os.path.join(self.img_root, img_file)
                gt_path = os.path.join(self.gt_root, gt_file)

                # 验证文件存在性
                if not os.path.exists(gt_path):
                    raise FileNotFoundError(
                        f"标注文件 {gt_path} 不存在，对应图像文件 {img_file}"
                    )

                self.image_list.append({
                    "img": img_path,
                    "lbl": gt_path
                })

            # 提取基础文件名列表
            self.images = [os.path.basename(img["img"]).replace(
                self.config[dataset_name]["image_suffix"], ""
            ) for img in self.image_list]

            # 构建files字典（适配原有结构）
            self.files = collections.defaultdict(list)
            for item in self.image_list:
                self.files[self.split].append(item)


        else:
            # 其他数据集的原有逻辑
            self.img_root = os.path.join(self.dir, "images/")
            self.gt_root = os.path.join(self.dir, "gt/")
            self.image_list = self.config[dataset_name]["file"]
            self.images = [line.rstrip("\n") for line in open(self.image_list)]  # 取消注释这行

        # augmentations
        self.augmentation = self.config["augmentation"]
        self.crop_size = [
            self.config[dataset_name]["crop_size"],
            self.config[dataset_name]["crop_size"],
        ]
        self.multi_scale_pred = multi_scale_pred

        # preprocess
        self.angle_theta = self.config["angle_theta"]
        self.mean_bgr = np.array(eval(self.config["mean"]))
        self.deviation_bgr = np.array(eval(self.config["std"]))
        self.normalize_type = self.config["normalize_type"]

        # to avoid Deadloack  between CV Threads and Pytorch Threads caused in resizing
        cv2.setNumThreads(0)

        self.files = collections.defaultdict(list)
        for f in self.images:
            self.files[self.split].append(
                {
                    "img": self.img_root
                           + f
                           + self.config[dataset_name]["image_suffix"],
                    "lbl": self.gt_root + f + self.config[dataset_name]["gt_suffix"],
                }
            )

    def __len__(self):
        return len(self.files[self.split])

    def getRoadData(self, index):

        image_dict = self.files[self.split][index]
        # read each image in list
        if os.path.isfile(image_dict["img"]):
            image = cv2.imread(image_dict["img"]).astype(float)
        else:
            print("ERROR: couldn't find image -> ", image_dict["img"])

        if os.path.isfile(image_dict["lbl"]):
            gt = cv2.imread(image_dict["lbl"], 0).astype(float)
        else:
            print("ERROR: couldn't find image -> ", image_dict["lbl"])

        if self.split == "train":
            image, gt = self.random_crop(image, gt, self.crop_size)
        else:
            image = cv2.resize(
                image,
                (self.crop_size[0], self.crop_size[1]),
                interpolation=cv2.INTER_LINEAR,
            )
            gt = cv2.resize(
                gt,
                (self.crop_size[0], self.crop_size[1]),
                interpolation=cv2.INTER_LINEAR,
            )

        if self.split == "train" and index == len(self.files[self.split]) - 1:
            np.random.shuffle(self.files[self.split])

        h, w, c = image.shape
        if self.augmentation == 1:
            flip = np.random.choice(2) * 2 - 1
            image = np.ascontiguousarray(image[:, ::flip, :])
            gt = np.ascontiguousarray(gt[:, ::flip])
            rotation = np.random.randint(4) * 90
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rotation, 1)
            image = cv2.warpAffine(image, M, (w, h))
            gt = cv2.warpAffine(gt, M, (w, h))

        image = self.reshape(image)
        image = torch.from_numpy(np.array(image))

        return image, gt

    def getOrientationGT(self, keypoints, height, width):
        vecmap, vecmap_angles = affinity_utils.getVectorMapsAngles(
            (height, width), keypoints, theta=self.angle_theta, bin_size=10
        )
        vecmap_angles = torch.from_numpy(vecmap_angles)

        return vecmap_angles

    def getCorruptRoad(
            self, road_gt, height, width, artifacts_shape="linear", element_counts=8
    ):
        # False Negative Mask
        FNmask = np.ones((height, width), float)
        # False Positive Mask
        FPmask = np.zeros((height, width), float)
        indices = np.where(road_gt == 1)

        if artifacts_shape == "square":
            shapes = [[16, 16], [32, 32]]
            ##### FNmask
            if len(indices[0]) == 0:  ### no road pixel in GT
                pass
            else:
                for c_ in range(element_counts):
                    c = np.random.choice(len(shapes), 1)[
                        0
                    ]  ### choose random square size
                    shape_ = shapes[c]
                    ind = np.random.choice(len(indices[0]), 1)[
                        0
                    ]  ### choose a random road pixel as center for the square
                    row = indices[0][ind]
                    col = indices[1][ind]

                    FNmask[
                    row - shape_[0] / 2: row + shape_[0] / 2,
                    col - shape_[1] / 2: col + shape_[1] / 2,
                    ] = 0
            #### FPmask
            for c_ in range(element_counts):
                c = np.random.choice(len(shapes), 2)[0]  ### choose random square size
                shape_ = shapes[c]
                row = np.random.choice(height - shape_[0] - 1, 1)[
                    0
                ]  ### choose random pixel
                col = np.random.choice(width - shape_[1] - 1, 1)[
                    0
                ]  ### choose random pixel
                FPmask[
                row - shape_[0] / 2: row + shape_[0] / 2,
                col - shape_[1] / 2: col + shape_[1] / 2,
                ] = 1

        elif artifacts_shape == "linear":
            ##### FNmask
            if len(indices[0]) == 0:  ### no road pixel in GT
                pass
            else:
                for c_ in range(element_counts):
                    c1 = np.random.choice(len(indices[0]), 1)[
                        0
                    ]  ### choose random 2 road pixels to draw a line
                    c2 = np.random.choice(len(indices[0]), 1)[0]
                    cv2.line(
                        FNmask,
                        (indices[1][c1], indices[0][c1]),
                        (indices[1][c2], indices[0][c2]),
                        0,
                        self.angle_theta * 2,
                    )
            #### FPmask
            for c_ in range(element_counts):
                row1 = np.random.choice(height, 1)
                col1 = np.random.choice(width, 1)
                row2, col2 = (
                    row1 + np.random.choice(50, 1),
                    col1 + np.random.choice(50, 1),
                )
                cv2.line(FPmask, (col1, row1), (col2, row2), 1, self.angle_theta * 2)

        erased_gt = (road_gt * FNmask) + FPmask
        erased_gt[erased_gt > 0] = 1

        return erased_gt

    def reshape(self, image):

        if self.normalize_type == "Std":
            image = (image - self.mean_bgr) / (3 * self.deviation_bgr)
        elif self.normalize_type == "MinMax":
            image = (image - self.min_bgr) / (self.max_bgr - self.min_bgr)
            image = image * 2 - 1
        elif self.normalize_type == "Mean":
            image -= self.mean_bgr
        else:
            image = (image / 255.0) * 2 - 1

        image = image.transpose(2, 0, 1)
        return image

    # 修改后：
    def random_crop(self, image, gt, size):
        h, w, _ = image.shape  # 正确的高度和宽度顺序
        crop_h, crop_w = size  # 保持参数顺序一致

        start_x = np.random.randint(0, w - crop_w)
        start_y = np.random.randint(0, h - crop_h)

        image = image[start_x: start_x + crop_w, start_y: start_y + crop_h, :]
        gt = gt[start_x: start_x + crop_w, start_y: start_y + crop_h]

        return image, gt


class SpacenetDataset(RoadDataset):
    def __init__(self, config, seed=7, multi_scale_pred=True, is_train=True):
        super(SpacenetDataset, self).__init__(
            config, "spacenet", seed, multi_scale_pred, is_train
        )

        # preprocess
        self.threshold = self.config["thresh"]
        print("Threshold is set to {} for {}".format(self.threshold, self.split))

    def __getitem__(self, index):

        image, gt = self.getRoadData(index)
        c, h, w = image.shape

        labels = []
        vecmap_angles = []
        if self.multi_scale_pred:
            smoothness = [1, 2, 4]
            scale = [4, 2, 1]
        else:
            smoothness = [4]
            scale = [1]

        for i, val in enumerate(scale):
            if val != 1:
                gt_ = cv2.resize(
                    gt,
                    (int(math.ceil(h / (val * 1.0))), int(math.ceil(w / (val * 1.0)))),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                gt_ = gt

            gt_orig = np.copy(gt_)
            gt_orig /= 255.0
            gt_orig[gt_orig < self.threshold] = 0
            gt_orig[gt_orig >= self.threshold] = 1
            labels.append(gt_orig)

            keypoints = affinity_utils.getKeypoints(
                gt_, thresh=0.98, smooth_dist=smoothness[i]
            )
            vecmap_angle = self.getOrientationGT(
                keypoints,
                height=int(math.ceil(h / (val * 1.0))),
                width=int(math.ceil(w / (val * 1.0))),
            )
            vecmap_angles.append(vecmap_angle)

        return image, labels, vecmap_angles


class DeepGlobeDataset(RoadDataset):
    def __init__(self, config, seed=7, multi_scale_pred=True, is_train=True):
        super(DeepGlobeDataset, self).__init__(
            config, "deepglobe", seed, multi_scale_pred, is_train
        )

        pass

    def __getitem__(self, index):

        image, gt = self.getRoadData(index)
        c, h, w = image.shape

        labels = []
        vecmap_angles = []
        if self.multi_scale_pred:
            smoothness = [1, 2, 4]
            scale = [4, 2, 1]
        else:
            smoothness = [4]
            scale = [1]

        for i, val in enumerate(scale):
            if val != 1:
                gt_ = cv2.resize(
                    gt,
                    (int(math.ceil(h / (val * 1.0))), int(math.ceil(w / (val * 1.0)))),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                gt_ = gt

            gt_orig = np.copy(gt_)
            gt_orig /= 255.0
            labels.append(gt_orig)

            # Create Orientation Ground Truth
            keypoints = affinity_utils.getKeypoints(
                gt_orig, is_gaussian=False, smooth_dist=smoothness[i]
            )
            vecmap_angle = self.getOrientationGT(
                keypoints,
                height=int(math.ceil(h / (val * 1.0))),
                width=int(math.ceil(w / (val * 1.0))),
            )
            vecmap_angles.append(vecmap_angle)

        return image, labels, vecmap_angles


class MyRoadData(RoadDataset):
    def __init__(self, config, seed=7, multi_scale_pred=True, is_train=True):
        super(MyRoadData, self).__init__(
            config, "myroad", seed, multi_scale_pred, is_train
        )
        pass

    def __getitem__(self, index):

        image, gt = self.getRoadData(index)
        c, h, w = image.shape

        labels = []
        vecmap_angles = []
        if self.multi_scale_pred:
            smoothness = [1, 2, 4]
            scale = [4, 2, 1]
        else:
            smoothness = [4]
            scale = [1]

        for i, val in enumerate(scale):
            if val != 1:
                gt_ = cv2.resize(
                    gt,
                    (int(math.ceil(h / (val * 1.0))), int(math.ceil(w / (val * 1.0)))),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                gt_ = gt

            gt_orig = np.copy(gt_)
            gt_orig /= 255.0
            labels.append(gt_orig)

            # Create Orientation Ground Truth
            keypoints = affinity_utils.getKeypoints(
                gt_orig, is_gaussian=False, smooth_dist=smoothness[i]
            )
            vecmap_angle = self.getOrientationGT(
                keypoints,
                height=int(math.ceil(h / (val * 1.0))),
                width=int(math.ceil(w / (val * 1.0))),
            )
            vecmap_angles.append(vecmap_angle)

        return image, labels, vecmap_angles


class SpacenetDatasetCorrupt(RoadDataset):
    def __init__(self, config, seed=7, is_train=True):
        super(SpacenetDatasetCorrupt, self).__init__(
            config, "spacenet", seed, multi_scale_pred=False, is_train=is_train
        )

        # preprocess
        self.threshold = self.config["thresh"]
        print("Threshold is set to {} for {}".format(self.threshold, self.split))

    def __getitem__(self, index):
        image, gt = self.getRoadData(index)
        c, h, w = image.shape
        gt /= 255.0
        gt[gt < self.threshold] = 0
        gt[gt >= self.threshold] = 1

        erased_gt = self.getCorruptRoad(gt.copy(), h, w)
        erased_gt = torch.from_numpy(erased_gt)

        return image, [gt], [erased_gt]


class DeepGlobeDatasetCorrupt(RoadDataset):
    def __init__(self, config, seed=7, is_train=True):
        super(DeepGlobeDatasetCorrupt, self).__init__(
            config, "deepglobe", seed, multi_scale_pred=False, is_train=is_train
        )

        pass

    def __getitem__(self, index):
        image, gt = self.getRoadData(index)
        c, h, w = image.shape
        gt /= 255.0

        erased_gt = self.getCorruptRoad(gt, h, w)
        erased_gt = torch.from_numpy(erased_gt)

        return image, [gt], [erased_gt]
