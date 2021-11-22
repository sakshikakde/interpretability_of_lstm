import gi
gi.require_version('Gtk', '2.0')
from numpy import linalg
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
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    Saliency,
    NoiseTunnel,
    ShapleyValueSampling,
    FeaturePermutation,
    FeatureAblation,
    Occlusion
)


def resume_model(opt, model):
    """ Resume model 
    """
    checkpoint = torch.load(opt.resume_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])


def saliency(clip, model, target_class, flag):

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    spatial_transform = Compose([
        Scale((120, 120)),
        ToTensor(opt.norm_value), norm_method
    ])
    if spatial_transform is not None:
        clip = [spatial_transform(img) for img in clip]

    # model.zero_grad()

    if flag == "custom":
        clip = torch.stack(clip, dim=0)
        clip = clip.unsqueeze(0)
        clip.requires_grad = True
        outputs = model(clip, training = False)

        outputs = F.softmax(outputs, dim = 1)
        outputs, _ = torch.topk(outputs, k=1)   
        print(outputs)
        outputs.backward()
        grad = [] #resout.grad.data.cpu().numpy()
        saliency = [] #np.abs(grad)
        grad = clip.grad.data.cpu().numpy()
        saliency = np.absolute(grad)
        saliency = saliency.squeeze()
    if flag == "grad":
        Grad = Saliency(model)
    return grad, saliency

def saveSaliency(saliency, folder_name, file_name):
    np.save(folder_name + file_name, saliency)
    print("Saliency maps saved as ", file_name)


if __name__ == "__main__":
    opt = parse_opts()
    print(opt)
    save_folder = "./saliency/spatial/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    data = load_annotation_data(opt.annotation_path)
    class_to_idx = get_class_labels(data)
    device = torch.device("cpu")
    print(class_to_idx)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    model = generate_model(opt, device)
    ModelTypes = opt.model
    sal = None
    if opt.resume_path:
        resume_model(opt, model)
        opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
        opt.std = get_std(opt.norm_value)
        
        cam = cv2.VideoCapture(
            '/home/sakshi/courses/CMSC828W/cnn-lstm/data/kth_trimmed_data/running/0_person01_running_d1_uncomp.avi')
        total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        N = total_frames-1
        clip = []
        for i in range(total_frames):
            ret, img = cam.read()
            if len(clip) == N:
                grad, sal = saliency(clip, model, 1, "custom")
                print("Sal dim = ", sal.shape)
                saveSaliency(sal.squeeze(), save_folder, ModelTypes + "_spatial_saliency")
                clip = []

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            clip.append(img)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Plotting Saliency >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    if sal is not None:
        for t in range(sal.shape[0]):
            frame_sal = sal[t, :, :, :]
            frame_sal = frame_sal.reshape(120, 120, 3)
            frame_sal = cv2.normalize(frame_sal,
                            frame_sal, 0, 255, cv2.NORM_MINMAX)
            file_name = ModelTypes + "_spatial_saliency_frame_" + str(t) + ".png"
            # cv2.imshow(file_name, frame_sal)
            cv2.imwrite(save_folder + file_name, frame_sal)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

