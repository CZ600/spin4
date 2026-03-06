from __future__ import print_function
import sys
sys.path.append('model')
import argparse
import json
import os
from datetime import datetime
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from model.models import MODELS
from road_dataset import DeepGlobeDataset, SpacenetDataset,MyRoadData
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from utils.loss import CrossEntropyLoss2d, mIoULoss
from utils import util
from utils import viz_util
# 在文件顶部添加tqdm导入
from tqdm import tqdm
import time
#from hanging_threads import start_monitoring
#start_monitoring(seconds_frozen=300, test_interval=100)

__dataset__ = {"spacenet": SpacenetDataset, "deepglobe": DeepGlobeDataset,"myroad": MyRoadData}


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", required=True, type=str, help="config file path"
)
parser.add_argument(
    "--model_name",
    required=True,
    choices=sorted(MODELS.keys()),
    help="Name of Model = {}".format(MODELS.keys()),
)
parser.add_argument("--exp", required=True, type=str, help="Experiment Name/Directory")
parser.add_argument(
    "--resume", default=None, type=str, help="path to latest checkpoint (default: None)"
)
parser.add_argument(
    "--dataset",
    required=True,
    choices=sorted(__dataset__.keys()),
    help="select dataset name from {}. (default: Spacenet)".format(__dataset__.keys()),
)
parser.add_argument(
    "--model_kwargs",
    default={},
    type=json.loads,
    help="parameters for the model",
)
parser.add_argument(
    "--multi_scale_pred",
    default=True,
    type=util.str2bool,
    help="perform multi-scale prediction (default: True)",
)

def train(epoch):
    train_loss_iou = 0
    train_loss_vec = 0
    model.train()
    optimizer.zero_grad()
    hist = np.zeros((config["task1_classes"], config["task1_classes"]))
    hist_angles = np.zeros((config["task2_classes"], config["task2_classes"]))
    crop_size = config["train_dataset"][args.dataset]["crop_size"]
    # 添加tqdm进度条
    # 修改进度条初始化，添加total参数
    # 在创建DataLoader后添加检查
    print(f"训练集批次数量: {len(train_loader)}")
    # 如果输出为0，需检查数据集路径和配置

    train_loader_with_tqdm = tqdm(
        train_loader,
        total=len(train_loader),  # 显式设置总步数
        desc=f'Train Epoch {epoch}/{config["trainer"]["total_epochs"]}',
        unit='batch',
        dynamic_ncols=True,
        miniters=1,  # 最小更新间隔
        file=sys.stdout  # 确保输出到控制台

    )
    for i, data in enumerate(train_loader_with_tqdm, 0):
        inputsBGR, labels, vecmap_angles = data
        inputsBGR = Variable(inputsBGR.float().cuda())
        outputs, pred_vecmaps = model(inputsBGR)

        if args.multi_scale_pred:
            loss1 = road_loss(outputs[0], util.to_variable(labels[0]), False)
            num_stacks = model.module.num_stacks if num_gpus > 1 else model.num_stacks
            for idx in range(num_stacks - 1):
                loss1 += road_loss(outputs[idx + 1], util.to_variable(labels[0]), False)
            for idx, output in enumerate(outputs[-2:]):
                loss1 += road_loss(output, util.to_variable(labels[idx + 1]), False)

            loss2 = angle_loss(pred_vecmaps[0], util.to_variable(vecmap_angles[0]))
            for idx in range(num_stacks - 1):
                loss2 += angle_loss(
                    pred_vecmaps[idx + 1], util.to_variable(vecmap_angles[0])
                )
            for idx, pred_vecmap in enumerate(pred_vecmaps[-2:]):
                loss2 += angle_loss(pred_vecmap, util.to_variable(vecmap_angles[idx + 1]))

            outputs = outputs[-1]
            pred_vecmaps = pred_vecmaps[-1]
        else:
            loss1 = road_loss(outputs, util.to_variable(labels[-1]), False)
            loss2 = angle_loss(pred_vecmaps, util.to_variable(vecmap_angles[-1]))

        # import pdb
        # pdb.set_trace()
        train_loss_iou += loss1.item()
        train_loss_vec += loss2.item()

        _, predicted = torch.max(outputs.data, 1)

        correctLabel = labels[-1].view(-1, crop_size, crop_size).long()
        hist += util.fast_hist(
            predicted.view(predicted.size(0), -1).cpu().numpy(),
            correctLabel.view(correctLabel.size(0), -1).cpu().numpy(),
            config["task1_classes"],
        )

        _, predicted_angle = torch.max(pred_vecmaps.data, 1)
        correct_angles = vecmap_angles[-1].view(-1, crop_size, crop_size).long()
        hist_angles += util.fast_hist(
            predicted_angle.view(predicted_angle.size(0), -1).cpu().numpy(),
            correct_angles.view(correct_angles.size(0), -1).cpu().numpy(),
            config["task2_classes"],
        )

        p_accu, miou, road_iou, fwacc , precision, recall,f1= util.performMetrics(
            train_loss_file,
            val_loss_file,
            epoch,
            hist,
            train_loss_iou / (i + 1),
            train_loss_vec / (i + 1),
            is_train=True,
            write=True,
        )
        p_accu_angle, miou_angle, fwacc_angle = util.performAngleMetrics(
            train_loss_angle_file, val_loss_angle_file, epoch, hist_angles
        )


        torch.autograd.backward([loss1, loss2])

        if i % config["trainer"]["iter_size"] == 0 or i == len(train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()

        # # 更新进度条显示
        # train_loader_tqdm.set_postfix({
        #     'Loss': f"{train_loss_iou / (i + 1):.4f}",
        #     'VecLoss': f"{train_loss_vec / (i + 1):.4f}",
        #     'mIoU': f"{miou:.2f}%",
        #     'RoadIoU': f"{road_iou:.2f}%",
        #     'AngleIoU': f"{miou_angle:.2f}%"
        # })

        # 删除原有的set_postfix调用，改用进度条对象的方法
        train_loader_with_tqdm.set_postfix({
            'Loss': f"{train_loss_iou / (i + 1):.4f}",
            'VecLoss': f"{train_loss_vec / (i + 1):.4f}",
            'mIoU': f"{miou:.2f}%",
            'RoadIoU': f"{road_iou:.2f}%",
            'AngleIoU': f"{miou_angle:.2f}%",
            "f1": f"{f1:.2f}"
        })

        del (
            outputs,
            pred_vecmaps,
            predicted,
            correct_angles,
            correctLabel,
            inputsBGR,
            labels,
            vecmap_angles,
        )

    accuracy, miou, road_iou, fwacc,precision,recall,f1 = util.performMetrics(
        train_loss_file,
        val_loss_file,
        epoch,
        hist,
        train_loss_iou / len(train_loader),
        train_loss_vec / len(train_loader),
        write=True,
    )
        # util.performAngleMetrics(
        #     train_loss_angle_file, val_loss_angle_file, epoch, hist_angles, write=True
        # )
    # return accuracy,miou,road_iou,fwacc,precision,recall,f1,train_loss_iou / len(train_loader),train_loss_vec / len(train_loader)
    return accuracy,miou,road_iou,fwacc,precision,recall,f1




def test(epoch):
    global best_accuracy
    global best_miou
    model.eval()
    test_loss_iou = 0
    test_loss_vec = 0
    hist = np.zeros((config["task1_classes"], config["task1_classes"]))
    hist_angles = np.zeros((config["task2_classes"], config["task2_classes"]))
    crop_size = config["val_dataset"][args.dataset]["crop_size"]
    # 添加测试进度条
    val_loader_tqdm = tqdm(
        val_loader,
        desc=f'Test Epoch {epoch}/{config["trainer"]["total_epochs"]}',
        unit='batch',
        dynamic_ncols=True
    )
    for i, (inputsBGR, labels, vecmap_angles) in enumerate(val_loader_tqdm, 0):
        inputsBGR = Variable(
            inputsBGR.float().cuda(), volatile=True, requires_grad=False
        )

        outputs, pred_vecmaps = model(inputsBGR)
        if args.multi_scale_pred:
            loss1 = road_loss(outputs[0], util.to_variable(labels[0], True, False), True)
            num_stacks = model.module.num_stacks if num_gpus > 1 else model.num_stacks
            for idx in range(num_stacks - 1):
                loss1 += road_loss(outputs[idx + 1], util.to_variable(labels[0], True, False), True)
            for idx, output in enumerate(outputs[-2:]):
                loss1 += road_loss(output, util.to_variable(labels[idx + 1], True, False), True)

            loss2 = angle_loss(pred_vecmaps[0], util.to_variable(vecmap_angles[0], True, False))
            for idx in range(num_stacks - 1):
                loss2 += angle_loss(
                    pred_vecmaps[idx + 1], util.to_variable(vecmap_angles[0], True, False)
                )
            for idx, pred_vecmap in enumerate(pred_vecmaps[-2:]):
                loss2 += angle_loss(
                    pred_vecmap, util.to_variable(vecmap_angles[idx + 1], True, False)
                )

            outputs = outputs[-1]
            pred_vecmaps = pred_vecmaps[-1]
        else:
            loss1 = road_loss(outputs, util.to_variable(labels[0], True, False), True)
            loss2 = angle_loss(pred_vecmaps, util.to_variable(labels[0], True, False))

        test_loss_iou += loss1.item()
        test_loss_vec += loss2.item()

        _, predicted = torch.max(outputs.data, 1)

        correctLabel = labels[-1].view(-1, crop_size, crop_size).long()
        hist += util.fast_hist(
            predicted.view(predicted.size(0), -1).cpu().numpy(),
            correctLabel.view(correctLabel.size(0), -1).cpu().numpy(),
            config["task1_classes"],
        )

        _, predicted_angle = torch.max(pred_vecmaps.data, 1)
        correct_angles = vecmap_angles[-1].view(-1, crop_size, crop_size).long()
        hist_angles += util.fast_hist(
            predicted_angle.view(predicted_angle.size(0), -1).cpu().numpy(),
            correct_angles.view(correct_angles.size(0), -1).cpu().numpy(),
            config["task2_classes"],
        )

        p_accu, miou, road_iou, fwacc , precision, recall, f1= util.performMetrics(
            train_loss_file,
            val_loss_file,
            epoch,
            hist,
            test_loss_iou / (i + 1),
            test_loss_vec / (i + 1),
            is_train=False,
            write=True,
        )
        # p_accu_angle, miou_angle, fwacc_angle = util.performAngleMetrics(
        #     train_loss_angle_file, val_loss_angle_file, epoch, hist_angles, is_train=False
        # )

        if i % 100 == 0 or i == len(val_loader) - 1:
            images_path = "{}/images/".format(experiment_dir)
            util.ensure_dir(images_path)
            util.savePredictedProb(
                inputsBGR.data.cpu(),
                labels[-1].cpu(),
                predicted.cpu(),
                F.softmax(outputs, dim=1).data.cpu()[:, 1, :, :],
                predicted_angle.cpu(),
                os.path.join(images_path, "validate_pair_{}_{}.png".format(epoch, i)),
                norm_type=config["val_dataset"]["normalize_type"],
            )
        # 更新测试进度条
        val_loader_tqdm.set_postfix({
            'Loss': f"{test_loss_iou / (i + 1):.4f}",
            'VecLoss': f"{test_loss_vec / (i + 1):.4f}",
            'mIoU': f"{miou:.2f}%",
            'RoadIoU': f"{road_iou:.2f}%"
        })
        del inputsBGR, labels, predicted, outputs, pred_vecmaps, predicted_angle
        return loss1

    accuracy, miou, road_iou, fwacc, precision, recall, f1 = util.performMetrics(
        train_loss_file,
        val_loss_file,
        epoch,
        hist,
        test_loss_iou / len(val_loader),
        test_loss_vec / len(val_loader),
        is_train=False,
        write=True,
    )
    # util.performAngleMetrics(
    #     train_loss_angle_file,
    #     val_loss_angle_file,
    #     epoch,
    #     hist_angles,
    #     is_train=False,
    #     write=True,
    # )

    if miou > best_miou:
        best_accuracy = accuracy
        best_miou = miou
        util.save_checkpoint(epoch, test_loss_iou / len(val_loader), model, optimizer, best_accuracy, best_miou, config, experiment_dir,is_best=True)

    return (
        test_loss_iou / len(val_loader),
        test_loss_vec / len(val_loader),
        accuracy,
        miou,
        precision,
        recall,
        f1,
    )


if __name__ == "__main__":


    args = parser.parse_args()
    config = None

    if args.resume is not None:
        if args.config is not None:
            print("Warning: --config overridden by --resume")
            config = torch.load(args.resume)["config"]
    elif args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)

    assert config is not None

    util.setSeed(config)

    experiment_dir = os.path.join(config["trainer"]["save_dir"], args.exp)
    util.ensure_dir(experiment_dir)

    ###Logging Files
    train_file = "{}/{}_train_loss.txt".format(experiment_dir, args.dataset)
    test_file = "{}/{}_test_loss.txt".format(experiment_dir, args.dataset)

    train_loss_file = open(train_file, "w")
    val_loss_file = open(test_file, "w")

    ### Angle Metrics
    train_file_angle = "{}/{}_train_angle_loss.txt".format(experiment_dir, args.dataset)
    test_file_angle = "{}/{}_test_angle_loss.txt".format(experiment_dir, args.dataset)

    print(train_file_angle,test_file_angle)

    train_loss_angle_file = open(train_file_angle, "rb", 0)
    val_loss_angle_file = open(test_file_angle, "rb", 0)
    ################################################################################
    num_gpus = torch.cuda.device_count()

    model = MODELS[args.model_name](
        config["task1_classes"], config["task2_classes"], **args.model_kwargs
    )

    if num_gpus > 1:
        print("Training with multiple GPUs ({})".format(num_gpus))
        model = nn.DataParallel(model).cuda()
    else:
        print("Single Cuda Node is avaiable")
        model.cuda()
    ################################################################################

    ### Load Dataset from root folder and intialize DataLoader
    train_loader = data.DataLoader(
        __dataset__[args.dataset](
            config["train_dataset"],
            seed=config["seed"],
            is_train=True,
            multi_scale_pred=args.multi_scale_pred,
        ),
        batch_size=config["train_batch_size"],
        num_workers=4,
        shuffle=True,
        pin_memory=False,
    )

    val_loader = data.DataLoader(
        __dataset__[args.dataset](
            config["val_dataset"],
            seed=config["seed"],
            is_train=False,
            multi_scale_pred=args.multi_scale_pred,
        ),
        batch_size=config["val_batch_size"],
        num_workers=2,
        shuffle=False,
        pin_memory=False,
    )

    print("Training with dataset => {}".format(train_loader.dataset.__class__.__name__))
    ################################################################################

    best_accuracy = 0
    best_miou = 0
    start_epoch = 1
    total_epochs = config["trainer"]["total_epochs"]
    optimizer = optim.SGD(
        model.parameters(), lr=config["optimizer"]["lr"], momentum=0.9, weight_decay=0.0005
    )

    if args.resume is not None:
        print("Loading from existing FCN and copying weights to continue....")
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint["miou"]
        # stat_parallel_dict = util.getParllelNetworkStateDict(checkpoint['state_dict'])
        # model.load_state_dict(stat_parallel_dict)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        util.weights_init(model, manual_seed=config["seed"])

    viz_util.summary(model, print_arch=False)

    scheduler = MultiStepLR(
        optimizer,
        milestones=eval(config["optimizer"]["lr_drop_epoch"]),
        gamma=config["optimizer"]["lr_step"],
    )


    weights = torch.ones(config["task1_classes"]).cuda()
    if config["task1_weight"] < 1:
        print("Roads are weighted.")
        weights[0] = 1 - config["task1_weight"]
        weights[1] = config["task1_weight"]


    weights_angles = torch.ones(config["task2_classes"]).cuda()
    if config["task2_weight"] < 1:
        print("Road angles are weighted.")
        weights_angles[-1] = config["task2_weight"]


    angle_loss = CrossEntropyLoss2d(
        weight=weights_angles, size_average=True, ignore_index=255, reduce=True
    ).cuda()
    road_loss = mIoULoss(
        weight=weights, size_average=True, n_classes=config["task1_classes"]
    ).cuda()



    # 初始化CSV日志
    datatime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_dir = "myroadExp"  # 改为相对路径或正确绝对路径（如 r"C:\myroadExp"）
    csv_path = os.path.join(save_dir, f"{datatime}_training_log.csv")

    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            'epoch',
            'train_loss', 'train_vec_loss',
            'train_acc', 'train_miou', 'train_precision', 'train_recall', 'train_f1',
        ])
    for epoch in range(start_epoch, total_epochs + 1):
        start_time = datetime.now()
        scheduler.step(epoch)
        print("\nTraining Epoch: %d" % epoch)
        # 训练并获取指标
        train_loss, train_vec_loss, train_acc, train_miou, train_precision, train_recall, train_f1 = train(epoch)
        if (epoch % config["trainer"]["test_freq"] == 0) or (epoch>90):
            print("\nTesting Epoch: %d" % epoch)
            val_loss = test(epoch)
        if epoch % 5 == 0:
            print("\nSaving Model")
            util.save_checkpoint(epoch, train_miou, model, optimizer, best_accuracy, best_miou, config, experiment_dir)

        end_time = datetime.now()
        print("Time Elapsed for epoch => {1}".format(epoch, end_time - start_time))
        with open(csv_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                epoch,
                train_loss, train_vec_loss,
                train_acc, train_miou,
                train_precision, train_recall, train_f1,
            ])

