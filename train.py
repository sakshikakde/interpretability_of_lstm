import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import argparse
import tensorboardX
import os
import random
import numpy as np
from utils import AverageMeter, calculate_accuracy


class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1,1)



def train_epoch(model, data_loader, criterion, optimizer, epoch, log_interval, device):
    model.train()
    train_loss = 0.0
    losses = AverageMeter()
    accuracies = AverageMeter()
    weight_clipper = WeightClipper()

    for batch_idx, (data, targets) in enumerate(data_loader):
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)

        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        train_loss += loss.item()
        losses.update(loss.item(), data.size(0))
        accuracies.update(acc, data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # model.parameters.weight.data = torch.clamp(model.weight.data, max = 1.0, min = -1.0)
        model.lstm.apply(weight_clipper)
        model.fc_layers.apply(weight_clipper)

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = train_loss / log_interval
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(data_loader.dataset), 100. * (batch_idx + 1) / len(data_loader), avg_loss))
            train_loss = 0.0

    print('Train set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
        len(data_loader.dataset), losses.avg, accuracies.avg * 100))
    
    for param in model.lstm.parameters():
        idx = torch.abs(param.data) > 1
        if(torch.any(idx)):
            print("BADE weight hain. check karo..")

        # print(param.data)

    return losses.avg, accuracies.avg  