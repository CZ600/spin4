import math
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from skimage.morphology import skeletonize


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise ValueError("Boolean value expected.")


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def setSeed(config):
    manual_seed = np.random.randint(1, 10000) if config["seed"] is None else config["seed"]
    print("Random Seed:", manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    random.seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)


def getParllelNetworkStateDict(state_dict):
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_state_dict[key[7:]] = value
    return new_state_dict


def to_variable(tensor, volatile=False, requires_grad=False):
    return Variable(tensor.long().cuda(), requires_grad=requires_grad)


def weights_init(model, manual_seed=7):
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    random.seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            if n > 0:
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()


def performAngleMetrics(train_loss_angle_file, val_loss_angle_file, epoch, hist, is_train=True, write=False):
    pixel_accuracy = np.diag(hist).sum() / hist.sum()
    mean_accuracy = np.diag(hist) / hist.sum(1)
    mean_iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    freq = hist.sum(1) / hist.sum()
    fwavacc = (freq[freq > 0] * mean_iou[freq > 0]).sum()
    if write and is_train:
        train_loss_angle_file.write(
            "[%d], Pixel Accuracy:%.3f, Mean Accuracy:%.3f, Mean IoU:%.3f, Freq.Weighted Accuray:%.3f\n"
            % (epoch, 100 * pixel_accuracy, 100 * np.nanmean(mean_accuracy), 100 * np.nanmean(mean_iou), 100 * fwavacc)
        )
    elif write and not is_train:
        val_loss_angle_file.write(
            "[%d], Pixel Accuracy:%.3f, Mean Accuracy:%.3f, Mean IoU:%.3f, Freq.Weighted Accuray:%.3f\n"
            % (epoch, 100 * pixel_accuracy, 100 * np.nanmean(mean_accuracy), 100 * np.nanmean(mean_iou), 100 * fwavacc)
        )
    return 100 * pixel_accuracy, 100 * np.nanmean(mean_iou), 100 * fwavacc


def compute_segmentation_metrics(hist):
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        class_iou = np.zeros(hist.shape[0], dtype=np.float64)
        return {
            "pixel_accuracy": 0.0,
            "mean_accuracy": 0.0,
            "mean_iou": 0.0,
            "background_iou": 0.0,
            "road_iou": 0.0,
            "class_iou": class_iou,
            "fwavacc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    with np.errstate(divide="ignore", invalid="ignore"):
        pixel_accuracy = np.diag(hist).sum() / total
        mean_accuracy = np.diag(hist) / hist.sum(1)
        class_iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        freq = hist.sum(1) / total
        fwavacc = (freq[freq > 0] * class_iou[freq > 0]).sum()

    tp = hist[1, 1] if hist.shape[0] > 1 else 0.0
    fp = hist[0, 1] if hist.shape[0] > 1 else 0.0
    fn = hist[1, 0] if hist.shape[0] > 1 else 0.0
    precision = tp / (tp + fp + 1e-8) * 100.0
    recall = tp / (tp + fn + 1e-8) * 100.0
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return {
        "pixel_accuracy": 100.0 * pixel_accuracy,
        "mean_accuracy": 100.0 * np.nanmean(mean_accuracy),
        "mean_iou": 100.0 * np.nanmean(class_iou),
        "background_iou": 100.0 * class_iou[0] if class_iou.size > 0 and not np.isnan(class_iou[0]) else 0.0,
        "road_iou": 100.0 * class_iou[1] if class_iou.size > 1 and not np.isnan(class_iou[1]) else 0.0,
        "class_iou": np.nan_to_num(class_iou, nan=0.0),
        "fwavacc": 100.0 * fwavacc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def performMetrics(train_loss_file, val_loss_file, epoch, hist, loss, loss_vec, is_train=True, write=False):
    metrics = compute_segmentation_metrics(hist)
    log_line = (
        "[%d], Loss:%.5f, Loss(VecMap):%.5f, Pixel Accuracy:%.3f, Mean Accuracy:%.3f, Mean IoU:%.3f, "
        "Class IoU:[%.5f/%.5f], Freq.Weighted Accuray:%.3f, Precision:%.3f, Recall:%.3f, F1:%.3f\n"
        % (
            epoch,
            loss,
            loss_vec,
            metrics["pixel_accuracy"],
            metrics["mean_accuracy"],
            metrics["mean_iou"],
            metrics["class_iou"][0],
            metrics["class_iou"][1] if metrics["class_iou"].size > 1 else 0.0,
            metrics["fwavacc"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )
    )
    if write and is_train:
        train_loss_file.write(log_line)
    elif write and not is_train:
        val_loss_file.write(log_line)

    return (
        metrics["pixel_accuracy"],
        metrics["mean_iou"],
        metrics["road_iou"],
        metrics["fwavacc"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def build_checkpoint_state(epoch, model, optimizer, best_accuracy, best_miou, config, extra_state=None):
    arch = type(model.module).__name__ if torch.cuda.device_count() > 1 else type(model).__name__
    state = {
        "arch": arch,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "pixel_accuracy": best_accuracy,
        "miou": best_miou,
        "config": config,
    }
    if extra_state:
        state.update(extra_state)
    return state


def save_checkpoint(state, path):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)


def savePredictedProb(real, gt, predicted, predicted_prob, pred_affinity=None, image_name="", norm_type="Mean"):
    b, c, h, w = real.size()
    grid = []
    mean_bgr = np.array([70.95016901, 71.16398124, 71.30953645])
    deviation_bgr = np.array([34.00087859, 35.18201658, 36.40463264])

    for idx in range(b):
        real_ = np.asarray(real[idx].numpy().transpose(1, 2, 0), dtype=np.float32)
        if norm_type == "Mean":
            real_ = real_ + mean_bgr
        elif norm_type == "Std":
            real_ = (real_ * deviation_bgr) + mean_bgr

        real_ = np.asarray(real_, dtype=np.uint8)
        gt_ = np.asarray(gt[idx].numpy() * 255.0, dtype=np.uint8)
        gt_ = np.stack((gt_,) * 3).transpose(1, 2, 0)

        predicted_ = np.asarray(predicted[idx].numpy() * 255.0, dtype=np.uint8)
        predicted_ = np.stack((predicted_,) * 3).transpose(1, 2, 0)

        predicted_prob_ = np.asarray(predicted_prob[idx].numpy() * 255.0, dtype=np.uint8)
        predicted_prob_ = cv2.applyColorMap(predicted_prob_, cv2.COLORMAP_JET)

        if pred_affinity is not None:
            hsv = np.zeros_like(real_)
            hsv[..., 1] = 255
            affinity_ = pred_affinity[idx].numpy()
            mag = np.copy(affinity_)
            mag[mag < 36] = 1
            mag[mag >= 36] = 0
            affinity_[affinity_ == 36] = 0
            hsv[..., 0] = affinity_ * 10 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            affinity_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            pair = np.concatenate((real_, gt_, predicted_, predicted_prob_, affinity_bgr), axis=1)
        else:
            pair = np.concatenate((real_, gt_, predicted_, predicted_prob_), axis=1)
        grid.append(pair)

    if pred_affinity is not None:
        cv2.imwrite(image_name, np.array(grid).reshape(b * h, 5 * w, 3))
    else:
        cv2.imwrite(image_name, np.array(grid).reshape(b * h, 4 * w, 3))


def get_relaxed_precision(a, b, buffer):
    tp = 0
    indices = np.where(a == 1)
    for ind in range(len(indices[0])):
        tp += (np.sum(b[indices[0][ind] - buffer: indices[0][ind] + buffer + 1, indices[1][ind] - buffer: indices[1][ind] + buffer + 1]) > 0).astype(int)
    return tp


def relaxed_f1(pred, gt, buffer=3):
    rprecision_tp, rrecall_tp, pred_positive, gt_positive = 0, 0, 0, 0
    for batch_idx in range(pred.shape[0]):
        pred_sk = skeletonize(pred[batch_idx])
        gt_sk = skeletonize(gt[batch_idx])
        rprecision_tp += get_relaxed_precision(pred_sk, gt_sk, buffer)
        rrecall_tp += get_relaxed_precision(gt_sk, pred_sk, buffer)
        pred_positive += len(np.where(pred_sk == 1)[0])
        gt_positive += len(np.where(gt_sk == 1)[0])

    return rprecision_tp, rrecall_tp, pred_positive, gt_positive
