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
import gc
import copy
import psutil

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

frame_size = 120
# make it param
get_TSR = True
sequence_length = 30
if not get_TSR:
    print("NOT USING TSR")

def plotSaliency(opt, grads, sal_type, save_folder, subfolder):
    if(type(grads) == torch.Tensor):
        sal = grads.detach().numpy().squeeze()
    else:
        sal = grads
    print("Sal shape : ", sal.shape)
    sal = (sal - np.min(sal)) / (np.max(sal) - np.min(sal)) * 255
    if sal is not None:
        for t in range(sal.shape[0]):
            frame_sal = sal[t, 0, :, :]
            file_name = opt.model + "_" + sal_type +"_" + str(t) + ".png"
            cv2.imwrite(os.path.join(save_folder, subfolder, file_name), frame_sal)

def plotTimeSal(opt, time_contribution, save_folder, subfolder, scaling = 30):
    w = time_contribution.shape[0]
    time_sal_text = np.zeros((int(4 * scaling), int(w * scaling)))

    for t in range(w):
        time_sal_text[:, t*scaling:(t+1)*scaling] = time_contribution[t]

    font = cv2.FONT_HERSHEY_SIMPLEX
    for t in range(w):
        cv2.putText(time_sal_text, str(t), (int((t * scaling)), int(2 * scaling)), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    file_name = opt.model + "_time_contribution.png"
    cv2.imwrite(os.path.join(save_folder, subfolder, file_name), time_sal_text * 255)


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
                        getFeatureImp = True,
                        hasBaseline=None,
                        hasFeatureMask=None,
                        hasSliding_window_shapes=None):
    assignment=input[0, 0, 0, 0, 0]
    print("Input shape ", input.shape)

    input = input.cpu().detach().numpy()
    assignment = assignment.cpu().detach().numpy()
    
    timeGrad=np.zeros((1, sequence_length))
    inputGrad=np.zeros((input_size[0], input_size[1] , 1))
    newGrad=np.zeros((input_size[0], input_size[1], sequence_length), dtype = np.float32)
    if(hasBaseline==None):  
        ActualGrad = Grad.attribute(torch.from_numpy(input),
                            target=TestingLabel,
                            additional_forward_args=(False)).data.cpu().numpy()
    else:
        if(hasFeatureMask!=None):
            ActualGrad = Grad.attribute(torch.from_numpy(input),
                                        baselines=hasBaseline,
                                        target=TestingLabel,
                                        feature_mask=hasFeatureMask,
                                        additional_forward_args=(False)).data.cpu().numpy()    
        elif(hasSliding_window_shapes!=None):
            ActualGrad = Grad.attribute(torch.from_numpy(input),
                                        sliding_window_shapes=hasSliding_window_shapes,
                                        baselines=hasBaseline,
                                        target=TestingLabel,
                                        additional_forward_args=(False)).data.cpu().numpy()
        else:
            ActualGrad = Grad.attribute(torch.from_numpy(input),
                                        baselines=hasBaseline,
                                        target=TestingLabel,
                                        additional_forward_args=(False)).data.cpu().numpy()

    for t in range(sequence_length):
        print("Temporal saliency for t = ", t)
        newInput = copy.deepcopy(input)
        newInput[0, t, :,:,:] = assignment
        
        if(hasBaseline==None):  
            timeGrad_perTime = Grad.attribute(torch.from_numpy(newInput),
                                            target=TestingLabel,
                                            additional_forward_args=(False)).data.cpu().numpy()
        else:
            if(hasFeatureMask!=None):
                timeGrad_perTime = Grad.attribute(torch.from_numpy(newInput),
                                                  baselines=hasBaseline,
                                                  target=TestingLabel,
                                                  feature_mask=hasFeatureMask,
                                                  additional_forward_args=(False)).data.cpu().numpy()    
            elif(hasSliding_window_shapes!=None):
                timeGrad_perTime = Grad.attribute(torch.from_numpy(newInput),
                                                sliding_window_shapes=hasSliding_window_shapes,
                                                baselines=hasBaseline,
                                                target=TestingLabel,
                                                additional_forward_args=(False)).data.cpu().numpy()
            else:
                timeGrad_perTime = Grad.attribute(torch.from_numpy(newInput),
                                                baselines=hasBaseline,
                                                target=TestingLabel,
                                                additional_forward_args=(False)).data.cpu().numpy()


        timeGrad_perTime= np.absolute(ActualGrad - timeGrad_perTime)
        timeGrad[:,t] = np.sum(timeGrad_perTime)

    del timeGrad_perTime

    timeContibution=preprocessing.minmax_scale(timeGrad, axis=1)
    meanTime = np.quantile(timeContibution, .55)  
    print("Done calculating time importance")  
    print("Time contribution shape : ", timeContibution.shape)
    print("Sequence lenght is {}".format(sequence_length))
    t = 0

    with torch.no_grad():
        if getFeatureImp:
            while (t < sequence_length):
                print("Calculating feature importance for frame ", t)
                gc.collect()
                torch.cuda.empty_cache()
                print("Memory allocated is {}".format((torch.cuda.memory_allocated())))
                print("Memory reserved is {}".format(torch.cuda.memory_reserved()))
                if(timeContibution[0,t]>meanTime):
                    print("In if")
                    for r in range(0, input_size[0], 10):
                        for c in range(0, input_size[1], 10):
                            # print(torch.cuda.memory_stats(device=device)['active.all.allocated'])
                            # print(np.shape(input))
                            newInput = copy.deepcopy(input)
                            newInput[0, 1, :, r:r+10, c:c+10] = assignment
                            if(hasBaseline==None):  

                                inputGrad_perInput = Grad.attribute(torch.from_numpy(newInput),
                                                                    target=TestingLabel,
                                                                    additional_forward_args=(False))
                                inputGrad_perInput_numpy = inputGrad_perInput.data.detach().cpu().numpy()

                                del inputGrad_perInput
                                del newInput

                                print("CPU percent {}".format(psutil.cpu_percent()))
                                print("RAM percent {}".format(psutil.virtual_memory().percent))
                                print("After inputgrad and copy {}".format(torch.cuda.memory_allocated('cuda:0')))
                            else:
                                if(hasFeatureMask!=None):
                                    inputGrad_perInput = Grad.attribute(torch.from_numpy(newInput),
                                                        baselines=hasBaseline,
                                                        target=TestingLabel,
                                                        feature_mask=hasFeatureMask,
                                                        additional_forward_args=(False)).data.cpu().numpy()    
                                elif(hasSliding_window_shapes!=None):
                                    inputGrad_perInput = Grad.attribute(torch.from_numpy(newInput),
                                                                        sliding_window_shapes=hasSliding_window_shapes,
                                                                        baselines=hasBaseline, 
                                                                        target=TestingLabel,
                                                                        additional_forward_args=(False)).data.cpu().numpy()
                                else:
                                    inputGrad_perInput = Grad.attribute(torch.from_numpy(newInput),
                                                                        baselines=hasBaseline,
                                                                        target=TestingLabel,
                                                                        additional_forward_args=(False)).data.cpu().numpy()



                            inputGrad_perInput_numpy=np.absolute(ActualGrad - inputGrad_perInput_numpy)
                            inputGrad[r:r+10, c:c+10, 0] = np.sum(inputGrad_perInput_numpy)

                        featureContibution = inputGrad #preprocessing.minmax_scale(inputGrad)
                else:
                    print("In Else")
                    featureContibution=np.ones((input_size[0], input_size[1], 1))*0.1

                for r in range(0, input_size[0], 10):
                    for c in range(0, input_size[1], 10):
                        newGrad [r:r+10, c:c+10, t]= timeContibution[0,t]*featureContibution[r:r+10, c:c+10, 0]

                del featureContibution
                t = t+1
        else:
            newGrad = ActualGrad
        return newGrad, timeContibution

def predict(clip, model):
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = transforms.Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = transforms.Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = transforms.Normalize(opt.mean, opt.std)

    spatial_transform = transforms.Compose([transforms.Resize((opt.sample_size, opt.sample_size)), 
    transforms.ToTensor(),
    norm_method])

    if spatial_transform is not None:
        clip = [spatial_transform(img).to(device) for img in clip]

    model.train()
    clip = torch.stack(clip, dim=0)
    clip = clip.unsqueeze(0)
    with torch.no_grad():
        print(clip.shape)
        outputs = model(clip)
        outputs = F.softmax(outputs)
    print("After softmax : ", outputs)
    scores, idx = torch.topk(outputs, k=1)
    preds = idx
    return clip, preds

def saliency(clip, model, device, target_class):
    clip, prediction = predict(clip, model)
    print("Output = ", idx_to_class[prediction.item()]) 
    saliency = Saliency(model)
    grads = saliency.attribute(clip, 
                               target=prediction.item(),
                               additional_forward_args=(False))
    if get_TSR:
        tsr_grads, timeContibution = getTwoStepRescaling(saliency, clip, sequence_length = sequence_length,
                            input_size = (120, 120, 3), 
                            TestingLabel = 1)
    else:
        tsr_grads = grads
        timeContibution = np.zeros((1, sequence_length))
    print("grads shape ", grads.shape)
    print("TSR time contribution shape : ", timeContibution.shape)
    print("TSR grads shape ", tsr_grads.shape)
    return grads.squeeze(), tsr_grads.squeeze(), timeContibution.squeeze()

def saveSaliency(saliency, folder_name, file_name):
    np.save(folder_name + file_name, saliency)
    print("Saliency maps saved as ", file_name)


if __name__ == "__main__":
    opt = parse_opts()
    data = load_annotation_data(opt.annotation_path)
    class_to_idx = get_class_labels(data)
    # make it param
    image_folder_path = "./data/kth/image_data/"
    classes = os.listdir(image_folder_path)
    for c in classes:
        dataset = os.listdir(os.path.join(image_folder_path, c))
        for d in dataset:
            dataset_path = os.path.join(os.path.join(image_folder_path, c, d))
            print("......................Calculating saliency for ", dataset_path)
            save_folder = os.path.join(os.path.join("./saliency/spatial/", d))
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            if not os.path.exists(save_folder + "/tsr"):
                os.mkdir(save_folder + "/tsr")
            if not os.path.exists(save_folder + "/grad"):
                os.mkdir(save_folder + "/grad")

            data = load_annotation_data(opt.annotation_path)
            class_to_idx = get_class_labels(data)
            use_cuda = True
            device = torch.device("cuda" if use_cuda else "cpu")
            print(class_to_idx)
            idx_to_class = {}
            for name, label in class_to_idx.items():
                idx_to_class[label] = name

            model = generate_model(opt, device)

            if not os.path.exists(dataset_path):
                print("Check image path")
                sys.exit()

            if opt.resume_path:
                resume_model(opt, model)
                model = nn.DataParallel(model)
                opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
                opt.std = get_std(opt.norm_value)
                images = [f for f in os.listdir(dataset_path) if f.endswith('jpg')]
                images = sorted(images, key=lambda x: int(x.split("_")[1].split(".")[0]))
                clip = []
                target = 0
                for image in images:
                    image_file = os.path.join(dataset_path, image)
                    print("Opening file : ", image_file)
                    frame = cv2.imread(image_file)
                    frame = Image.fromarray(frame)
                    clip.append(frame)

                grads, tsr_grads, time_contribution = saliency(clip, model, device, torch.tensor(target).to(device))
                grads = grads.cpu()
                saveSaliency(time_contribution, save_folder, "/tsr/"+ opt.model + "_tsr_time_contribution")
                saveSaliency(tsr_grads, save_folder, "/tsr/"+ opt.model + "_tsr_saliency")
                saveSaliency(grads, save_folder, "/grad/"+ opt.model + "_grad_saliency")

                # plot saliency
                print("Plotting saliency")
                plotSaliency(opt, grads, "grad_saliency_frame", save_folder, "grad")
                if get_TSR:
                    plotTimeSal(opt, time_contribution, save_folder, "tsr", scaling = 30)
                    #time scaled saliency
                    scaled_tsr_array = []
                    for t in range(sequence_length):
                        frame_sal = grads[t, 0, :, :]
                        scaled_sal = frame_sal * time_contribution[t]
                        scaled_tsr_array.append(scaled_sal.detach().numpy())
                    scaled_tsr_array = np.array(scaled_tsr_array)
                    plotSaliency(opt, tsr_grads, "tsr_saliency_frame", save_folder, "tsr")



