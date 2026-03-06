from __future__ import print_function
import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.models import MODELS
from road_dataset import DeepGlobeDataset, MyRoadData, SpacenetDataset
from utils import util, viz_util
from utils.loss import CrossEntropyLoss2d, mIoULoss


__dataset__ = {"spacenet": SpacenetDataset, "deepglobe": DeepGlobeDataset, "myroad": MyRoadData}


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="config file path")
parser.add_argument(
    "--model_name",
    required=True,
    choices=sorted(MODELS.keys()),
    help="Name of Model = {}".format(MODELS.keys()),
)
parser.add_argument("--exp", required=True, type=str, help="Experiment name")
parser.add_argument("--resume", default=None, type=str, help="path to checkpoint")
parser.add_argument(
    "--dataset",
    required=True,
    choices=sorted(__dataset__.keys()),
    help="select dataset name from {}".format(__dataset__.keys()),
)
parser.add_argument("--model_kwargs", default={}, type=json.loads, help="parameters for the model")
parser.add_argument(
    "--multi_scale_pred",
    default=True,
    type=util.str2bool,
    help="perform multi-scale prediction (default: True)",
)


def get_num_stacks(model):
    return model.module.num_stacks if isinstance(model, nn.DataParallel) else model.num_stacks


def move_targets_to_device(targets, device):
    return [target.to(device=device, dtype=torch.long) for target in targets]


def compute_batch_losses(outputs, pred_vecmaps, labels, vecmap_angles):
    if args.multi_scale_pred:
        seg_loss = road_loss(outputs[0], labels[0], False)
        num_stacks = get_num_stacks(model)
        for idx in range(num_stacks - 1):
            seg_loss += road_loss(outputs[idx + 1], labels[0], False)
        for idx, output in enumerate(outputs[-2:]):
            seg_loss += road_loss(output, labels[idx + 1], False)

        angle_value = angle_loss(pred_vecmaps[0], vecmap_angles[0])
        for idx in range(num_stacks - 1):
            angle_value += angle_loss(pred_vecmaps[idx + 1], vecmap_angles[0])
        for idx, pred_vecmap in enumerate(pred_vecmaps[-2:]):
            angle_value += angle_loss(pred_vecmap, vecmap_angles[idx + 1])

        final_output = outputs[-1]
    else:
        seg_loss = road_loss(outputs, labels[-1], False)
        angle_value = angle_loss(pred_vecmaps, vecmap_angles[-1])
        final_output = outputs

    total_loss = seg_loss + angle_value
    return total_loss, seg_loss, angle_value, final_output


def run_epoch(epoch, loader, is_train):
    model.train(mode=is_train)
    hist = np.zeros((config["task1_classes"], config["task1_classes"]))
    total_loss_sum = 0.0
    seg_loss_sum = 0.0
    angle_loss_sum = 0.0
    phase = "Train" if is_train else "Val"

    iterator = tqdm(loader, total=len(loader), desc=f"{phase} Epoch {epoch}/{config['trainer']['total_epochs']}", unit="batch", dynamic_ncols=True)
    for inputs_bgr, labels, vecmap_angles in iterator:
        inputs_bgr = inputs_bgr.float().to(device)
        labels = move_targets_to_device(labels, device)
        vecmap_angles = move_targets_to_device(vecmap_angles, device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs, pred_vecmaps = model(inputs_bgr)
            total_loss, seg_loss, angle_value, final_output = compute_batch_losses(outputs, pred_vecmaps, labels, vecmap_angles)
            if is_train:
                total_loss.backward()
                optimizer.step()

        total_loss_sum += total_loss.item()
        seg_loss_sum += seg_loss.item()
        angle_loss_sum += angle_value.item()

        predicted = torch.argmax(final_output.detach(), dim=1)
        correct_label = labels[-1]
        hist += util.fast_hist(
            predicted.view(predicted.size(0), -1).cpu().numpy(),
            correct_label.view(correct_label.size(0), -1).cpu().numpy(),
            config["task1_classes"],
        )
        metrics = util.compute_segmentation_metrics(hist)
        iterator.set_postfix({
            "loss": f"{total_loss_sum / (iterator.n if iterator.n else 1):.4f}",
            "miou": f"{metrics['mean_iou']:.2f}",
            "road_iou": f"{metrics['road_iou']:.2f}",
            "precision": f"{metrics['precision']:.2f}",
            "recall": f"{metrics['recall']:.2f}",
        })

    metrics = util.compute_segmentation_metrics(hist)
    metrics.update({
        "loss": total_loss_sum / len(loader),
        "seg_loss": seg_loss_sum / len(loader),
        "angle_loss": angle_loss_sum / len(loader),
    })
    return metrics


def log_metrics(writer, split, metrics, epoch):
    writer.add_scalar(f"{split}/loss", metrics["loss"], epoch)
    writer.add_scalar(f"{split}/background_iou", metrics["background_iou"], epoch)
    writer.add_scalar(f"{split}/road_iou", metrics["road_iou"], epoch)
    writer.add_scalar(f"{split}/precision", metrics["precision"], epoch)
    writer.add_scalar(f"{split}/recall", metrics["recall"], epoch)
    writer.add_scalar(f"{split}/miou", metrics["mean_iou"], epoch)


if __name__ == "__main__":
    args = parser.parse_args()
    config = None

    if args.resume is not None and args.config is not None and os.path.exists(args.resume):
        print("Warning: --config overridden by --resume")
        config = torch.load(args.resume, map_location="cpu")["config"]
    elif args.config is not None:
        with open(args.config, "r", encoding="utf-8") as file:
            config = json.load(file)

    assert config is not None
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training setup.")

    util.setSeed(config)
    device = torch.device("cuda")

    checkpoint_dir = os.path.abspath(config["trainer"].get("save_dir", "./checkpoints"))
    logs_root = os.path.abspath(config["trainer"].get("log_dir", "./logs"))
    util.ensure_dir(checkpoint_dir)
    util.ensure_dir(logs_root)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = config["trainer"].get("run_name_prefix", "spin")
    run_name = f"{run_prefix}_{timestamp}"
    run_dir = os.path.join(logs_root, run_name)
    util.ensure_dir(run_dir)

    metrics_csv = os.path.join(run_dir, "metrics.csv")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "split", "loss", "background_iou", "road_iou", "precision", "recall", "miou"])

    with open(os.path.join(run_dir, "resolved_config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    writer = SummaryWriter(log_dir=run_dir)

    num_gpus = torch.cuda.device_count()
    model = MODELS[args.model_name](config["task1_classes"], config["task2_classes"], **args.model_kwargs)
    if num_gpus > 1:
        print("Training with multiple GPUs ({})".format(num_gpus))
        model = nn.DataParallel(model).to(device)
    else:
        print("Training with a single CUDA device")
        model = model.to(device)

    train_loader = data.DataLoader(
        __dataset__[args.dataset](config["train_dataset"], seed=config["seed"], is_train=True, multi_scale_pred=args.multi_scale_pred),
        batch_size=config["train_batch_size"],
        num_workers=config["trainer"].get("train_num_workers", 2),
        shuffle=True,
        pin_memory=True,
    )
    val_loader = data.DataLoader(
        __dataset__[args.dataset](config["val_dataset"], seed=config["seed"], is_train=False, multi_scale_pred=args.multi_scale_pred),
        batch_size=config["val_batch_size"],
        num_workers=config["trainer"].get("val_num_workers", 2),
        shuffle=False,
        pin_memory=True,
    )

    print("Training with dataset => {}".format(train_loader.dataset.__class__.__name__))

    best_accuracy = 0.0
    best_miou = 0.0
    start_epoch = 1
    total_epochs = config["trainer"]["total_epochs"]

    optimizer = optim.SGD(model.parameters(), lr=config["optimizer"]["lr"], momentum=0.9, weight_decay=0.0005)
    scheduler = MultiStepLR(optimizer, milestones=eval(config["optimizer"]["lr_drop_epoch"]), gamma=config["optimizer"]["lr_step"])

    if args.resume is not None and os.path.exists(args.resume):
        print("Loading checkpoint from {}".format(args.resume))
        checkpoint = torch.load(args.resume, map_location="cpu")
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint.get("miou", 0.0)
        best_accuracy = checkpoint.get("pixel_accuracy", 0.0)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        util.weights_init(model, manual_seed=config["seed"])

    viz_util.summary(model, print_arch=False)

    weights = torch.ones(config["task1_classes"], device=device)
    if config["task1_weight"] < 1:
        print("Roads are weighted.")
        weights[0] = 1 - config["task1_weight"]
        weights[1] = config["task1_weight"]

    weights_angles = torch.ones(config["task2_classes"], device=device)
    if config["task2_weight"] < 1:
        print("Road angles are weighted.")
        weights_angles[-1] = config["task2_weight"]

    angle_loss = CrossEntropyLoss2d(weight=weights_angles, size_average=True, ignore_index=255, reduce=True).to(device)
    road_loss = mIoULoss(weight=weights, size_average=True, n_classes=config["task1_classes"]).to(device)

    checkpoint_interval = config["trainer"].get("checkpoint_interval", 5)

    for epoch in range(start_epoch, total_epochs + 1):
        start_time = datetime.now()
        print("\nTraining Epoch: {}".format(epoch))
        train_metrics = run_epoch(epoch, train_loader, is_train=True)

        print("\nValidation Epoch: {}".format(epoch))
        val_metrics = run_epoch(epoch, val_loader, is_train=False)
        scheduler.step()

        log_metrics(writer, "train", train_metrics, epoch)
        log_metrics(writer, "val", val_metrics, epoch)

        with open(metrics_csv, "a", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([epoch, "train", train_metrics["loss"], train_metrics["background_iou"], train_metrics["road_iou"], train_metrics["precision"], train_metrics["recall"], train_metrics["mean_iou"]])
            csv_writer.writerow([epoch, "val", val_metrics["loss"], val_metrics["background_iou"], val_metrics["road_iou"], val_metrics["precision"], val_metrics["recall"], val_metrics["mean_iou"]])

        if val_metrics["mean_iou"] > best_miou:
            best_miou = val_metrics["mean_iou"]
            best_accuracy = val_metrics["pixel_accuracy"]

        state = util.build_checkpoint_state(
            epoch,
            model,
            optimizer,
            best_accuracy,
            best_miou,
            config,
            extra_state={
                "args": vars(args),
                "run_name": run_name,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            },
        )

        util.save_checkpoint(state, os.path.join(checkpoint_dir, "latest.pth.tar"))
        if epoch % checkpoint_interval == 0:
            util.save_checkpoint(state, os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pth.tar"))
        if val_metrics["mean_iou"] >= best_miou:
            util.save_checkpoint(state, os.path.join(checkpoint_dir, "best.pth.tar"))

        elapsed = datetime.now() - start_time
        print(
            "Epoch {} | train loss {:.4f} | val loss {:.4f} | val mIoU {:.2f} | val road IoU {:.2f} | time {}".format(
                epoch,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["mean_iou"],
                val_metrics["road_iou"],
                elapsed,
            )
        )

    writer.close()
