import argparse
import cv2
import os

import torch
from torch.nn import DataParallel
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.coco import CocoTrainDataset
from datasets.transformations import ConvertKeypoints, Scale, Rotate, CropPad, Flip
from modules.get_parameters import get_parameters_conv, get_parameters_bn, get_parameters_conv_depthwise
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.loss import l2_loss
from modules.load_state import load_state, load_from_mobilenet
from val import evaluate

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader

if __name__ == '__main__':
    #============================================
    prepared_train_labels = "D:/人工智能学习资料/lightweight-human-pose-estimation.pytorch-master/annotations/prepared_train_annotation.pkl"  #准备好的annotations路径
    train_images_folder = "C:/Users/29943/Desktop/mydata_coco2017/train2017"    #COCO训练图片文件夹路径
    num_refinement_stages: int=1    #默认的refinement_stages数量
    base_lr = 4e-5                  #初始学习率
    batch_size = 32                 #训练batch_size大小
    batches_per_iter = 1            #累计梯度的批次数量
    num_workers = 8
    checkpoint_path = ""            #pth续训的路径
    from_mobilenet = ""             #加载权重从mobilnet特征提取网络
    weights_only = ""               #只需用预先训练的权重初始化层，然后从一开始就开始训练
    experiment_name = ""            #创建用来保存cpt文件夹的实验名字
    log_after = 100                 #每多少个iter打印训练损失
    val_labels = "D:/人工智能学习资料/lightweight-human-pose-estimation.pytorch-master/annotations/person_keypoints_val2017.json"                                 #val的json文件路径
    val_images_folder = "C/Users/29943/Desktop/mydata_coco2017/val2017"                          #COCO验证集图片路径
    checkpoint_after = 5000                         #每隔多少个iter保存pth
    val_after = 5000                                #每隔多少个iter进行验证
    val_output_name = "detections.json"             #输出json文件的名称与检测到的关键点
    #============================================

    checkpoints_folder = '{}_checkpoints'.format(experiment_name)
    if not os.path.exists(checkpoints_folder):      #创建一个实验文件夹名字
        os.makedirs(checkpoints_folder)

    net = PoseEstimationWithMobileNet(num_refinement_stages)  # 参数为需要refinement stage的数量，默认为1

    stride = 8
    sigma = 7
    path_thickness = 1
    #-------------数据集准备-----------------
    dataset = CocoTrainDataset(prepared_train_labels, train_images_folder,
                               stride, sigma, path_thickness,
                               transform=transforms.Compose([
                                   ConvertKeypoints(),
                                   Scale(),
                                   Rotate(pad=(128, 128, 128)),
                                   CropPad(pad=(128, 128, 128)),
                                   Flip()]))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #--------------优化器------------------------
    optimizer = optim.Adam([
        {'params': get_parameters_conv(net.model, 'weight')},
        {'params': get_parameters_conv_depthwise(net.model, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.model, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.model, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv(net.cpm, 'weight'), 'lr': base_lr},
        {'params': get_parameters_conv(net.cpm, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv_depthwise(net.cpm, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_conv(net.initial_stage, 'weight'), 'lr': base_lr},
        {'params': get_parameters_conv(net.initial_stage, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv(net.refinement_stages, 'weight'), 'lr': base_lr * 4},
        {'params': get_parameters_conv(net.refinement_stages, 'bias'), 'lr': base_lr * 8, 'weight_decay': 0},
        {'params': get_parameters_bn(net.refinement_stages, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.refinement_stages, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
    ], lr=base_lr, weight_decay=5e-4)
    drop_after_epoch = [100, 200, 260]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=drop_after_epoch, gamma=0.333)

    #-------------------配置训练过程----------------------
    num_iter = 0
    current_epoch = 0
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        if from_mobilenet:
            load_from_mobilenet(net, checkpoint)
        else:
            load_state(net, checkpoint)
            if not weights_only:
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                num_iter = checkpoint['iter']
                current_epoch = checkpoint['current_epoch']

    net = DataParallel(net).cpu()
    net.train()
    for epochId in range(current_epoch, 280):
        scheduler.step()
        total_losses = [0, 0] * (num_refinement_stages + 1)  # 每个阶段的heatmaps损失，paf损失
        batch_per_iter_idx = 0
        for batch_data in train_loader:
            if batch_per_iter_idx == 0:
                optimizer.zero_grad()

            images = batch_data['image'].cpu()
            keypoint_masks = batch_data['keypoint_mask'].cpu()
            paf_masks = batch_data['paf_mask'].cpu()
            keypoint_maps = batch_data['keypoint_maps'].cpu()
            paf_maps = batch_data['paf_maps'].cpu()

            stages_output = net(images)

            losses = []
            for loss_idx in range(len(total_losses) // 2):
                losses.append(l2_loss(stages_output[loss_idx * 2], keypoint_maps, keypoint_masks, images.shape[0]))
                losses.append(l2_loss(stages_output[loss_idx * 2 + 1], paf_maps, paf_masks, images.shape[0]))
                total_losses[loss_idx * 2] += losses[-2].item() / batches_per_iter
                total_losses[loss_idx * 2 + 1] += losses[-1].item() / batches_per_iter

            loss = losses[0]
            for loss_idx in range(1, len(losses)):
                loss += losses[loss_idx]
            loss /= batches_per_iter
            loss.backward()
            batch_per_iter_idx += 1
            if batch_per_iter_idx == batches_per_iter:
                optimizer.step()
                batch_per_iter_idx = 0
                num_iter += 1
            else:
                continue

            if num_iter % log_after == 0:
                print('Iter: {}'.format(num_iter))
                for loss_idx in range(len(total_losses) // 2):
                    print('\n'.join(['stage{}_pafs_loss:     {}', 'stage{}_heatmaps_loss: {}']).format(
                        loss_idx + 1, total_losses[loss_idx * 2 + 1] / log_after,
                        loss_idx + 1, total_losses[loss_idx * 2] / log_after))
                for loss_idx in range(len(total_losses)):
                    total_losses[loss_idx] = 0
            if num_iter % checkpoint_after == 0:
                snapshot_name = '{}/checkpoint_iter_{}.pth'.format(checkpoints_folder, num_iter)
                torch.save({'state_dict': net.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'iter': num_iter,
                            'current_epoch': epochId},
                           snapshot_name)
            if num_iter % val_after == 0:
                print('Validation...')
                evaluate(val_labels, val_output_name, val_images_folder, net)
                net.train()