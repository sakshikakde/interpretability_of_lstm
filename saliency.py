import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import sys
import torch.nn as nn
import torchvision.transforms as transforms
import json
from mean import get_mean, get_std
from PIL import Image
import cv2
from datasets.ucf101 import load_annotation_data
from datasets.ucf101 import get_class_labels
from model import generate_model
from utils import AverageMeter
from opts import parse_opts
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
import numpy as np


def resume_model(opt, model):
    """ Resume model 
    """
    checkpoint = torch.load(opt.resume_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])


def saliency(clip, model, target_class):
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    spatial_transform = Compose([
        Scale((150, 150)),
        ToTensor(opt.norm_value), norm_method
    ])
    if spatial_transform is not None:
        clip = [spatial_transform(img) for img in clip]

    model.zero_grad()

    clip = torch.stack(clip, dim=0)
    clip = clip.unsqueeze(0)
    clip.requires_grad = True
    outputs = model(clip, training = False)
    resout = model.resnet_out
    print("resout shape = ", resout[0].shape)

    # hidden = None
    # resnet_ops = []
    # for i in range(clip.shape[1]):
    #     input = clip[:, i, :, :, :]
    #     resnet_op = model.resnet(input)
    #     resnet_ops.cat(resnet_op, dim = 1)
    #     lstm_op, hidden = model.lstm(resnet_op.unsqueeze(0), hidden)
    
    # resnet_ops = torch.stack(resnet_ops, dim=0)
    # resnet_ops.requires_grad = True

    # fc1_op = model.fc1(lstm_op[-1, :, :])
    # fc1_op = F.relu(fc1_op)
    # outputs = model.fc2(fc1_op)


    outputs = F.softmax(outputs, dim = 1)
    outputs, _ = torch.topk(outputs, k=1)   
    # outputs = outputs[0, target_class]
    print(outputs)
    outputs.backward()
    grad = [] #resout.grad.data.cpu().numpy()
    saliency = [] #np.abs(grad)

    for i in range(clip.shape[1]):
        res_grad = resout[i].grad.data.cpu().numpy()
        print("res_grad shape = ", res_grad.shape)
        grad.append(res_grad)
        saliency.append(np.absolute(res_grad))
    # print("Saliency = ", saliency)
    print("sal length = ", len(saliency))
    print("sal ele length = ", saliency[0].shape)
    return np.array(grad).squeeze(), np.array(saliency).squeeze()

def saveSaliency(saliency, folder_name, file_name):
    np.save(folder_name + file_name, saliency)
    print("Saliency maps saved as ", file_name)


if __name__ == "__main__":
    opt = parse_opts()
    print(opt)
    save_folder = "/home/sakshi/courses/CMSC828W/cnn-lstm/saliency/"
    data = load_annotation_data(opt.annotation_path)
    class_to_idx = get_class_labels(data)
    device = torch.device("cpu")
    print(class_to_idx)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    model = generate_model(opt, device)
    ModelTypes = "lstm_cnn_attention"

    if opt.resume_path:
        resume_model(opt, model)
        opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
        opt.std = get_std(opt.norm_value)
        
        cam = cv2.VideoCapture(
            '/home/sakshi/courses/CMSC828W/cnn-lstm/data/video_data/punch/v_Punch_g16_c01.avi')
        total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        N = total_frames-1
        
        clip = []
        frame_count = 0
        block_count = 0
        while total_frames > 0:
            ret, img = cam.read()
            print("clip length is ", len(clip))
            if frame_count == N:
                grad, sal = saliency(clip, model, 1)
                print("Sal dim = ", sal.shape)
                saveSaliency(sal.squeeze(), save_folder, ModelTypes + str(block_count))
                block_count += 1
                frame_count = 0
                clip = []

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            clip.append(img)
            frame_count += 1
            total_frames -= 1
