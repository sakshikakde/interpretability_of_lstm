from cv2 import dnn_registerLayer
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101
import copy


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.resnet = resnet101(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.resnet_out = []
       
    def forward(self, x_3d, training = True):
        hidden = None
        for t in range(x_3d.size(1)):
            if training:
                with torch.no_grad():
                    x = self.resnet(x_3d[:, t, :, :, :])
                out, hidden = self.lstm(x.unsqueeze(0), hidden)  
            else:
                x = self.resnet(x_3d[:, t, :, :, :])
                self.resnet_out.append(x)
                # self.resnet_out[-1].require_grad = True
                self.resnet_out[-1].retain_grad()
                out, hidden = self.lstm(x.unsqueeze(0), hidden)  

        
        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)       


        return x