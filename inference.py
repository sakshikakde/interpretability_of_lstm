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


def predict(clip, model):
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    spatial_transform = Compose([
        Scale((150, 150)),
        #Scale(int(opt.sample_size / opt.scale_in_test)),
        #CornerCrop(opt.sample_size, opt.crop_position_in_test),
        ToTensor(opt.norm_value), norm_method
    ])
    if spatial_transform is not None:
        # spatial_transform.randomize_parameters()
        clip = [spatial_transform(img) for img in clip]

    clip = torch.stack(clip, dim=0)
    clip = clip.unsqueeze(0)
    with torch.no_grad():
        print(clip.shape)
        outputs = model(clip)
        outputs = F.softmax(outputs)
    print(outputs)
    scores, idx = torch.topk(outputs, k=1)
    preds = idx
    return preds


if __name__ == "__main__":
    opt = parse_opts()
    print(opt)
    data = load_annotation_data(opt.annotation_path)
    class_to_idx = get_class_labels(data)
    device = torch.device("cpu")
    print(class_to_idx)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    model = generate_model(opt, device)

    # model = nn.DataParallel(model, device_ids=None)
    # print(model)
    if opt.resume_path:
        resume_model(opt, model)
        opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
        opt.std = get_std(opt.norm_value)
        model.eval()

        cam = cv2.VideoCapture(
            '/home/sakshi/courses/CMSC828W/cnn-lstm/data/kth_trimmed_data/running/0_person01_running_d1_uncomp.avi')
        clip = []
        total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        
        N = total_frames - 1
        print("total_frames = ", total_frames)

        for i in range(total_frames):
            ret, img = cam.read()
            if len(clip) == N:
                preds = predict(clip, model)
                print("predictions = ", preds)
                print("Class = ", idx_to_class[preds.item()])
                font = cv2.FONT_HERSHEY_SIMPLEX
                if preds.size(0) != 0:
                    for f in range(len(clip)):
                        frame = np.array(clip[f])
                        cv2.putText(frame, idx_to_class[preds.item()], (50, 50), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.imshow('window', frame)
                        cv2.waitKey()
                clip = []

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            clip.append(img)

    cv2.destroyAllWindows()