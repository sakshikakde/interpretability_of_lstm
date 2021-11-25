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
from plotSaliency import plotHeatMapExampleWise


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

    outputs = F.softmax(outputs, dim = 1)
    outputs, _ = torch.topk(outputs, k=1)   
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
    save_folder = "./saliency/temporal/"
    data = load_annotation_data(opt.annotation_path)
    class_to_idx = get_class_labels(data)
    device = torch.device("cpu")
    print(class_to_idx)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    model = generate_model(opt, device)
    ModelTypes = "lstm_cnn"

    if opt.resume_path:
        resume_model(opt, model)
        opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
        opt.std = get_std(opt.norm_value)
        
        cam = cv2.VideoCapture('./data/kth_trimmed_data/running/0_person01_running_d1_uncomp.avi')
        total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        N = total_frames-1
        clip = []
        for i in range(total_frames):
            ret, img = cam.read()
            if len(clip) == N:
                grad, sal = saliency(clip, model, 1)
                saveSaliency(sal.squeeze(), save_folder, ModelTypes)
                clip = []

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            clip.append(img)


    #plot saliency
    sal = (255 * (sal / np.max(sal))).astype(np.uint8)
    plotHeatMapExampleWise(sal.T, ModelTypes, save_folder)

