from cv2 import dnn_registerLayer
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import resnet101
import copy

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers=3)

    def forward(self, x, hidden, training = True):
        out, hidden = self.lstm(x, hidden)
        return out, hidden

class FCModel(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super(FCModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, training = True):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x) 
        return x


class CNNLSTM(nn.Module):
    def __init__(self, lstm_input_sz, lstm_hidden_sz, r, d_a, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.resnet = resnet101(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, lstm_input_sz))
        self.lstm = LSTMModel(lstm_input_sz, lstm_hidden_sz)
        self.fc_layers = FCModel(lstm_hidden_sz, num_classes)
        # for attention
        self.linear_first = torch.nn.Linear(lstm_input_sz, d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a, r)
        self.linear_second.bias.data.fill_(0)
        self.d_a = d_a
        self.r = r
        self.pastTimeSteps = None

        self.resnet_out = []

    def softmax(self,input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])

        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)

    def getMatrixM(self, pastTimeSteps):
        x= self.linear_first(pastTimeSteps)
        x = torch.tanh(x)
        x = self.linear_second(x) 
        x = self.softmax(x,1)
        attention = x.transpose(1, 2) 

        matrixM = attention @ pastTimeSteps 
        matrixM = torch.sum(matrixM,1)/self.r
        return matrixM
       
    def forward(self, x_3d, training = True):
        hidden = None
        pastTimeSteps = torch.empty((1, 1, 300),  device = 'cuda')
        for t in range(x_3d.size(1)):
            if training:
                with torch.no_grad():
                    x = self.resnet(x_3d[:, t, :, :, :])
                if self.pastTimeSteps is None:
                    pastTimeSteps = x.unsqueeze(1) 
                else:
                    self.pastTimeSteps = torch.cat((self.pastTimeSteps, x), dim = 1)
                M = self.getMatrixM(pastTimeSteps)
                out, hidden = self.lstm(M.unsqueeze(0), hidden)  
            else:
                x = self.resnet(x_3d[:, t, :, :, :])
                self.resnet_out.append(x)
                self.resnet_out[-1].retain_grad()
                if self.pastTimeSteps is None:
                    pastTimeSteps = x.unsqueeze(1) 
                else:
                    self.pastTimeSteps = torch.cat((self.pastTimeSteps, x), dim = 1)               
                pastTimeSteps.retain_grad()
                M = self.getMatrixM(pastTimeSteps)
                out, hidden = self.lstm(M.unsqueeze(0), hidden)  

       
        x = self.fc_layers(out[-1, :, :])    
        return x