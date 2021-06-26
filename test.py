from data import  create_data_loader
from torch.optim import Adam
from utils.metrics import  accuracy
import torch

import torch.nn as nn
from model import get_model

def test(args,generator,model):
    device = torch.device("cuda")
    model.eval()
    total_acc = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    for j, data in enumerate(generator):
        images, labels = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            outputs = model(images)
            acc = accuracy(outputs,labels)
            loss = criterion(outputs,labels)
            total_acc += acc
            total_loss +=loss
    return total_acc/(1+j), total_loss/(1+j)

