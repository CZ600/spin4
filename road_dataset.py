import collections
import math
import os
import random

import cv2
import numpy as np
import torch
from data_utils import affinity_utils
from torch.utils import data


workspace_dir = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(workspace_dir, path))


class RoadDataset(data.Dataset):
    def __init__(self, config, dataset_name, seed=7, multi_scale_pred=True, is_train=True):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        self.split = "train" if is_train else "val"
        self.config = config
        self.dataset_cfg = self.config[dataset_name]
        self.augmentation = self.config["augmentation"]
        self.crop_size = [self.dataset_cfg["crop_size"], self.dataset_cfg["crop_size"]]
        self.multi_scale_pred = multi_scale_pred
        self.angle_theta = self.config["angle_theta"]
        self.mean_bgr = np.array(eval(self.config["mean"]))
        self.deviation_bgr = np.array(eval(self.config["std"]))
        self.normalize_type = self.config["normalize_type"]

        cv2.setNumThreads(0)

        self.files = collections.defaultdict(list)
        for item in self._build_samples(dataset_name, seed, is_train):
            self.files[self.split].append(item)

    def _build_samples(self, dataset_name, seed, is_train):
        samples = []

        if "set" in self.dataset_cfg:
            base_path = resolve_path(os.path.join(self.dataset_cfg["dir"], self.dataset_cfg["set"]))
            image_dir_name = self.dataset_cfg.get("image_dir", "data")
            gt_dir_name = self.dataset_cfg.get("gt_dir", "seg")
            self.img_root = os.path.join(base_path, image_dir_name)
            self.gt_root = os.path.join(base_path, gt_dir_name)

            image_suffix = self.dataset_cfg["image_suffix"]
            gt_suffix = self.dataset_cfg["gt_suffix"]
            raw_img_files = sorted(
                file_name for file_name in os.listdir(self.img_root) if file_name.endswith(image_suffix)
            )

            if dataset_name == "myroad" and is_train:
                total_samples = len(raw_img_files)
                sample_size = max(1, int(total_samples * 0.3))
                random.seed(seed)
                raw_img_files = random.sample(raw_img_files, sample_size)
                print(f"Sampled {sample_size}/{total_samples} training samples for myroad")

            for img_file in raw_img_files:
                base_name = img_file[: -len(image_suffix)] if image_suffix else os.path.splitext(img_file)[0]
                gt_file = base_name + gt_suffix
                img_path = os.path.join(self.img_root, img_file)
                gt_path = os.path.join(self.gt_root, gt_file)
                if os.path.exists(gt_path):
                    samples.append({"img": img_path, "lbl": gt_path})
            return samples

        self.dir = resolve_path(self.dataset_cfg["dir"])
        self.img_root = os.path.join(self.dir, "images")
        self.gt_root = os.path.join(self.dir, "gt")
        image_list_path = resolve_path(self.dataset_cfg["file"])
        with open(image_list_path, "r", encoding="utf-8") as file:
            entries = [line.strip() for line in file if line.strip()]

        for entry in entries:
            parts = entry.split()
            if len(parts) >= 2:
                img_path = os.path.join(self.dir, parts[0])
                gt_path = os.path.join(self.dir, parts[1])
            else:
                sample_id = parts[0]
                img_path = os.path.join(self.img_root, sample_id + self.dataset_cfg["image_suffix"])
                gt_path = os.path.join(self.gt_root, sample_id + self.dataset_cfg["gt_suffix"])
            samples.append({"img": img_path, "lbl": gt_path})
        return samples

    def __len__(self):
        return len(self.files[self.split])

    def getRoadData(self, index):
        image_dict = self.files[self.split][index]
        if os.path.isfile(image_dict["img"]):
            image = cv2.imread(image_dict["img"]).astype(float)
        else:
            raise FileNotFoundError(f"couldn't find image -> {image_dict['img']}")

        if os.path.isfile(image_dict["lbl"]):
            gt = cv2.imread(image_dict["lbl"], 0).astype(float)
        else:
            raise FileNotFoundError(f"couldn't find label -> {image_dict['lbl']}")

        if self.split == "train":
            image, gt = self.random_crop(image, gt, self.crop_size)
        else:
            image = cv2.resize(image, (self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt, (self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_NEAREST)

        if self.split == "train" and index == len(self.files[self.split]) - 1:
            np.random.shuffle(self.files[self.split])

        h, w, _ = image.shape
        if self.augmentation == 1:
            flip = np.random.choice(2) * 2 - 1
            image = np.ascontiguousarray(image[:, ::flip, :])
            gt = np.ascontiguousarray(gt[:, ::flip])
            rotation = np.random.randint(4) * 90
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), rotation, 1)
            image = cv2.warpAffine(image, matrix, (w, h))
            gt = cv2.warpAffine(gt, matrix, (w, h))

        image = self.reshape(image)
        image = torch.from_numpy(np.array(image))
        return image, gt

    def getOrientationGT(self, keypoints, height, width):
        _, vecmap_angles = affinity_utils.getVectorMapsAngles((height, width), keypoints, theta=self.angle_theta, bin_size=10)
        return torch.from_numpy(vecmap_angles)

    def getCorruptRoad(self, road_gt, height, width, artifacts_shape="linear", element_counts=8):
        fn_mask = np.ones((height, width), float)
        fp_mask = np.zeros((height, width), float)
        indices = np.where(road_gt == 1)

        if artifacts_shape == "square":
            shapes = [[16, 16], [32, 32]]
            if len(indices[0]) != 0:
                for _ in range(element_counts):
                    shape_ = shapes[np.random.choice(len(shapes), 1)[0]]
                    ind = np.random.choice(len(indices[0]), 1)[0]
                    row = indices[0][ind]
                    col = indices[1][ind]
                    fn_mask[int(row - shape_[0] / 2): int(row + shape_[0] / 2), int(col - shape_[1] / 2): int(col + shape_[1] / 2)] = 0
            for _ in range(element_counts):
                shape_ = shapes[np.random.choice(len(shapes), 1)[0]]
                row = np.random.choice(height - shape_[0] - 1, 1)[0]
                col = np.random.choice(width - shape_[1] - 1, 1)[0]
                fp_mask[int(row - shape_[0] / 2): int(row + shape_[0] / 2), int(col - shape_[1] / 2): int(col + shape_[1] / 2)] = 1
        elif artifacts_shape == "linear":
            if len(indices[0]) != 0:
                for _ in range(element_counts):
                    c1 = np.random.choice(len(indices[0]), 1)[0]
                    c2 = np.random.choice(len(indices[0]), 1)[0]
                    cv2.line(fn_mask, (indices[1][c1], indices[0][c1]), (indices[1][c2], indices[0][c2]), 0, self.angle_theta * 2)
            for _ in range(element_counts):
                row1 = np.random.choice(height, 1)
                col1 = np.random.choice(width, 1)
                row2, col2 = row1 + np.random.choice(50, 1), col1 + np.random.choice(50, 1)
                cv2.line(fp_mask, (col1, row1), (col2, row2), 1, self.angle_theta * 2)

        erased_gt = (road_gt * fn_mask) + fp_mask
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
        return image.transpose(2, 0, 1)

    def random_crop(self, image, gt, size):
        h, w, _ = image.shape
        crop_h, crop_w = size

        if h <= crop_h or w <= crop_w:
            image = cv2.resize(image, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
            return image, gt

        start_y = np.random.randint(0, h - crop_h + 1)
        start_x = np.random.randint(0, w - crop_w + 1)
        image = image[start_y: start_y + crop_h, start_x: start_x + crop_w, :]
        gt = gt[start_y: start_y + crop_h, start_x: start_x + crop_w]
        return image, gt


class SpacenetDataset(RoadDataset):
    def __init__(self, config, seed=7, multi_scale_pred=True, is_train=True):
        super(SpacenetDataset, self).__init__(config, "spacenet", seed, multi_scale_pred, is_train)
        self.threshold = self.config["thresh"]
        print("Threshold is set to {} for {}".format(self.threshold, self.split))

    def __getitem__(self, index):
        image, gt = self.getRoadData(index)
        c, h, w = image.shape
        labels = []
        vecmap_angles = []
        smoothness = [1, 2, 4] if self.multi_scale_pred else [4]
        scale = [4, 2, 1] if self.multi_scale_pred else [1]

        for i, val in enumerate(scale):
            gt_ = cv2.resize(gt, (int(math.ceil(h / (val * 1.0))), int(math.ceil(w / (val * 1.0)))), interpolation=cv2.INTER_NEAREST) if val != 1 else gt
            gt_orig = np.copy(gt_)
            gt_orig /= 255.0
            gt_orig[gt_orig < self.threshold] = 0
            gt_orig[gt_orig >= self.threshold] = 1
            labels.append(gt_orig)
            keypoints = affinity_utils.getKeypoints(gt_, thresh=0.98, smooth_dist=smoothness[i])
            vecmap_angles.append(self.getOrientationGT(keypoints, height=int(math.ceil(h / (val * 1.0))), width=int(math.ceil(w / (val * 1.0)))))

        return image, labels, vecmap_angles


class DeepGlobeDataset(RoadDataset):
    def __init__(self, config, seed=7, multi_scale_pred=True, is_train=True):
        super(DeepGlobeDataset, self).__init__(config, "deepglobe", seed, multi_scale_pred, is_train)

    def __getitem__(self, index):
        image, gt = self.getRoadData(index)
        c, h, w = image.shape
        labels = []
        vecmap_angles = []
        smoothness = [1, 2, 4] if self.multi_scale_pred else [4]
        scale = [4, 2, 1] if self.multi_scale_pred else [1]

        for i, val in enumerate(scale):
            gt_ = cv2.resize(gt, (int(math.ceil(h / (val * 1.0))), int(math.ceil(w / (val * 1.0)))), interpolation=cv2.INTER_NEAREST) if val != 1 else gt
            gt_orig = np.copy(gt_)
            gt_orig /= 255.0
            labels.append(gt_orig)
            keypoints = affinity_utils.getKeypoints(gt_orig, is_gaussian=False, smooth_dist=smoothness[i])
            vecmap_angles.append(self.getOrientationGT(keypoints, height=int(math.ceil(h / (val * 1.0))), width=int(math.ceil(w / (val * 1.0)))))

        return image, labels, vecmap_angles


class MyRoadData(RoadDataset):
    def __init__(self, config, seed=7, multi_scale_pred=True, is_train=True):
        super(MyRoadData, self).__init__(config, "myroad", seed, multi_scale_pred, is_train)

    def __getitem__(self, index):
        image, gt = self.getRoadData(index)
        c, h, w = image.shape
        labels = []
        vecmap_angles = []
        smoothness = [1, 2, 4] if self.multi_scale_pred else [4]
        scale = [4, 2, 1] if self.multi_scale_pred else [1]

        for i, val in enumerate(scale):
            gt_ = cv2.resize(gt, (int(math.ceil(h / (val * 1.0))), int(math.ceil(w / (val * 1.0)))), interpolation=cv2.INTER_NEAREST) if val != 1 else gt
            gt_orig = np.copy(gt_)
            gt_orig /= 255.0
            labels.append(gt_orig)
            keypoints = affinity_utils.getKeypoints(gt_orig, is_gaussian=False, smooth_dist=smoothness[i])
            vecmap_angles.append(self.getOrientationGT(keypoints, height=int(math.ceil(h / (val * 1.0))), width=int(math.ceil(w / (val * 1.0)))))

        return image, labels, vecmap_angles


class SpacenetDatasetCorrupt(RoadDataset):
    def __init__(self, config, seed=7, is_train=True):
        super(SpacenetDatasetCorrupt, self).__init__(config, "spacenet", seed, multi_scale_pred=False, is_train=is_train)
        self.threshold = self.config["thresh"]
        print("Threshold is set to {} for {}".format(self.threshold, self.split))

    def __getitem__(self, index):
        image, gt = self.getRoadData(index)
        c, h, w = image.shape
        gt /= 255.0
        gt[gt < self.threshold] = 0
        gt[gt >= self.threshold] = 1
        erased_gt = torch.from_numpy(self.getCorruptRoad(gt.copy(), h, w))
        return image, [gt], [erased_gt]


class DeepGlobeDatasetCorrupt(RoadDataset):
    def __init__(self, config, seed=7, is_train=True):
        super(DeepGlobeDatasetCorrupt, self).__init__(config, "deepglobe", seed, multi_scale_pred=False, is_train=is_train)

    def __getitem__(self, index):
        image, gt = self.getRoadData(index)
        c, h, w = image.shape
        gt /= 255.0
        erased_gt = torch.from_numpy(self.getCorruptRoad(gt, h, w))
        return image, [gt], [erased_gt]
