from data import  create_data_loader
from torch.optim import Adam
import torch

import torch.nn as nn
from model import  Model1

def test(args,generator):
    device = torch.device("cuda")
    # training loop
    Model1.to(device)
    Model1.eval()
    for j, data in enumerate(generator):
        images, labels = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            outputs = Model1(images)
            acc = 0

