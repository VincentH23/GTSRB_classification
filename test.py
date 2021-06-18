from data import  create_data_loader
from torch.optim import Adam
from utils.metrics import  accuracy
import torch

import torch.nn as nn
from model import get_model

def test(args,generator):
    device = torch.device("cuda")

    Model1 = get_model(args)
    Model1.to(device)
    Model1.eval()
    total_acc = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    for j, data in enumerate(generator):
        images, labels = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            outputs = Model1(images)
            acc = accuracy(outputs,labels)
            loss = criterion(outputs,labels)
            total_acc += acc
            total_loss +=loss
    return total_acc/(1+j), loss

