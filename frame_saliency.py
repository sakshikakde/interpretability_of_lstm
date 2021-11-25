import torch
torch.cuda.empty_cache()
from torch.autograd import Variable, grad
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
import torchvision.transforms as transforms
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
import numpy as np
from plotSaliency import plotHeatMapExampleWise
from  sklearn import preprocessing

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

frame_size = 150
get_TSR = False
if not get_TSR:
    print("NOT USING TSR")

def resume_model(opt, model):
    """ Resume model 
    """
    checkpoint = torch.load(opt.resume_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

def getTwoStepRescaling(Grad,
                        input,
                        sequence_length,
                        input_size, 
                        TestingLabel,
                        hasBaseline=None,
                        hasFeatureMask=None,
                        hasSliding_window_shapes=None):
    assignment=input[0, 0, 0, 0, 0]
    print("Input shape ", input.shape)
    timeGrad=np.zeros((1, sequence_length))
    inputGrad=np.zeros((input_size[0], input_size[1] , 1))
    newGrad=np.zeros((input_size[0], input_size[1], sequence_length), dtype = np.float32)
    if(hasBaseline==None):  
        ActualGrad = Grad.attribute(input,
                            target=TestingLabel,
                            additional_forward_args=(False)).data.cpu().numpy()
    else:
        if(hasFeatureMask!=None):
            ActualGrad = Grad.attribute(input,
                                        baselines=hasBaseline,
                                        target=TestingLabel,
                                        feature_mask=hasFeatureMask,
                                        additional_forward_args=(False)).data.cpu().numpy()    
        elif(hasSliding_window_shapes!=None):
            ActualGrad = Grad.attribute(input,
                                        sliding_window_shapes=hasSliding_window_shapes,
                                        baselines=hasBaseline,
                                        target=TestingLabel,
                                        additional_forward_args=(False)).data.cpu().numpy()
        else:
            ActualGrad = Grad.attribute(input,
                                        baselines=hasBaseline,
                                        target=TestingLabel,
                                        additional_forward_args=(False)).data.cpu().numpy()

    for t in range(sequence_length):
        print("Temporal saliency for t = ", t)
        newInput = input.clone()
        newInput[0, t, :,:,:] = assignment
        
        if(hasBaseline==None):  
            timeGrad_perTime = Grad.attribute(newInput,
                                            target=TestingLabel,
                                            additional_forward_args=(False)).data.cpu().numpy()
        else:
            if(hasFeatureMask!=None):
                timeGrad_perTime = Grad.attribute(newInput,
                                                  baselines=hasBaseline,
                                                  target=TestingLabel,
                                                  feature_mask=hasFeatureMask,
                                                  additional_forward_args=(False)).data.cpu().numpy()    
            elif(hasSliding_window_shapes!=None):
                timeGrad_perTime = Grad.attribute(newInput,
                                                sliding_window_shapes=hasSliding_window_shapes,
                                                baselines=hasBaseline,
                                                target=TestingLabel,
                                                additional_forward_args=(False)).data.cpu().numpy()
            else:
                timeGrad_perTime = Grad.attribute(newInput,
                                                baselines=hasBaseline,
                                                target=TestingLabel,
                                                additional_forward_args=(False)).data.cpu().numpy()


        timeGrad_perTime= np.absolute(ActualGrad - timeGrad_perTime)
        timeGrad[:,t] = np.sum(timeGrad_perTime)

    timeContibution=preprocessing.minmax_scale(timeGrad, axis=1)
    meanTime = np.quantile(timeContibution, .55)  
    print("Done calculating time importance")  
    print("Time contribution shape : ", timeContibution.shape)

    for t in range(sequence_length):
        print("Calculating feature importance for frame ", t)
        if(timeContibution[0,t]>meanTime):
            for r in range(0, input_size[0], 10):
                for c in range(0, input_size[1], 10):
                    newInput = input.clone()
                    newInput[0, 1, :, r:r+10, c:c+10] = assignment
                    if(hasBaseline==None):  
                        inputGrad_perInput = Grad.attribute(newInput,
                                                            target=TestingLabel,
                                                            additional_forward_args=(False)).data.cpu().numpy()
                    else:
                        if(hasFeatureMask!=None):
                            inputGrad_perInput = Grad.attribute(newInput,
                                                baselines=hasBaseline,
                                                target=TestingLabel,
                                                feature_mask=hasFeatureMask,
                                                additional_forward_args=(False)).data.cpu().numpy()    
                        elif(hasSliding_window_shapes!=None):
                            inputGrad_perInput = Grad.attribute(newInput,
                                                                sliding_window_shapes=hasSliding_window_shapes,
                                                                baselines=hasBaseline, 
                                                                target=TestingLabel,
                                                                additional_forward_args=(False)).data.cpu().numpy()
                        else:
                            inputGrad_perInput = Grad.attribute(newInput,
                                                                baselines=hasBaseline,
                                                                target=TestingLabel,
                                                                additional_forward_args=(False)).data.cpu().numpy()



                    inputGrad_perInput=np.absolute(ActualGrad - inputGrad_perInput)
                    inputGrad[r:r+10, c:c+10, 0] = np.sum(inputGrad_perInput)

                featureContibution = inputGrad #preprocessing.minmax_scale(inputGrad)
        else:
            featureContibution=np.ones((input_size[0], input_size[1], 1))*0.1

        for r in range(0, input_size[0], 10):
            for c in range(0, input_size[1], 10):
                newGrad [r:r+10, c:c+10, t]= timeContibution[0,t]*featureContibution[r:r+10, c:c+10, 0]
    return newGrad


def saliency(clip, model, device, target_class):
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = transforms.Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = transforms.Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = transforms.Normalize(opt.mean, opt.std)

    spatial_transform = transforms.Compose([transforms.Scale((opt.sample_size, opt.sample_size)), 
    transforms.ToTensor(),
    norm_method])

    if spatial_transform is not None:
        clip = [spatial_transform(img).to(device) for img in clip]

    model.train()
  
    clip = torch.stack(clip, dim=0)
    clip.to(device)
    clip = clip.unsqueeze(0)
    clip.requires_grad = True
    print("device: ", device)
    print("Clip shape : ", clip.shape)
    outputs = model(clip, training = False)
    outputs = F.softmax(outputs, dim = 1)
    # outputs, _ = torch.topk(outputs, k=1)
    torch.cuda.empty_cache()
    print("Output = ", outputs)   
    saliency = Saliency(model)
    grads = saliency.attribute(clip, target=target_class.item(), additional_forward_args=(False))
    if get_TSR:
        tsr_grads = getTwoStepRescaling(saliency, clip, sequence_length = 69,
                            input_size = (120, 120, 3), 
                            TestingLabel = 1)
    else:
        tsr_grads = grads
    print("grads shape ", grads.shape)
    print("TSR grads shape ", tsr_grads.shape)
    return grads, tsr_grads

def saveSaliency(saliency, folder_name, file_name):
    np.save(folder_name + file_name, saliency)
    print("Saliency maps saved as ", file_name)


if __name__ == "__main__":
    opt = parse_opts()
    print(opt)
    save_folder = "./saliency/spatial/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    if not os.path.exists(save_folder + "tsr"):
        os.mkdir(save_folder + "tsr")
    if not os.path.exists(save_folder + "grad"):
        os.mkdir(save_folder + "grad")

    data = load_annotation_data(opt.annotation_path)
    class_to_idx = get_class_labels(data)
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    print(class_to_idx)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    model = generate_model(opt, device)
    ModelTypes = "lstm_cnn"
    model.train()

    image_folder_path = "./data/kth/image_data/running/0_person01_running_d2_uncomp"
    if not os.path.exists(image_folder_path):
        print("Check image path")
        sys.exit()

    if opt.resume_path:
        resume_model(opt, model)
        opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
        opt.std = get_std(opt.norm_value)
        images = [f for f in os.listdir(image_folder_path) if f.endswith('jpg')]
        images = sorted(images, key=lambda x: int(x.split("_")[1].split(".")[0]))
        clip = []
        target = 1
        for image in images:
            image_file = os.path.join(image_folder_path, image)
            print("Opening file : ", image_file)
            frame = cv2.imread(image_file)
            frame = Image.fromarray(frame)
            clip.append(frame)

        grads, tsr_grads = saliency(clip, model, device, torch.tensor(target).to(device))
        saveSaliency(tsr_grads.squeeze(), save_folder, "tsr/"+ ModelTypes + "_tsr_saliency")
        saveSaliency(grads.squeeze(), save_folder, "grad/"+ ModelTypes + "_grad_saliency")

    # plot saliency
    sal = grads.detach().numpy().squeeze()
    print("Sal shape : ", sal.shape)
    sal = (sal - np.min(sal)) / (np.max(sal) - np.min(sal)) * 255
    if sal is not None:
        for t in range(sal.shape[0]):
            frame_sal = sal[t, 0, :, :]
            # frame_sal = frame_sal.reshape(frame_size, frame_size, 3)
            file_name = ModelTypes + "_spatial_saliency_frame_" + str(t) + ".png"
            cv2.imwrite(save_folder + "grad/"+ file_name, frame_sal)

    # plot saliency
    tsr_sal = tsr_grads.detach().numpy().squeeze()
    print("tsr_sal shape : ", tsr_sal.shape)
    tsr_sal = (tsr_sal - np.min(tsr_sal)) / (np.max(tsr_sal) - np.min(tsr_sal)) * 255
    if tsr_sal is not None:
        for t in range(tsr_sal.shape[0]):
            frame_sal = tsr_sal[t, 0, :, :]
            # frame_sal = frame_sal.reshape(frame_size, frame_size, 3)
            file_name = ModelTypes + "_tsr_saliency_frame_" + str(t) + ".png"
            cv2.imwrite(save_folder + "tsr/"+ file_name, frame_sal)

