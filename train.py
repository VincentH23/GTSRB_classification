from data import  create_data_loader
from torch.optim import Adam
from utils.metrics import accuracy
import torch
import torch.nn as nn
from model import  Model1
def train(args,generator):
    device = torch.device("cuda")
    # training loop
    Model1.to(device)
    Model1.train()
    optimizer = Adam(Model1.parameters(),args.lr)
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    for i in range (args.epoch):
        total_loss = 0
        total_acc = 0
        for j,data in enumerate(generator):
            images , labels = data[0].to(device),data[1].to(device)
            optimizer.zero_grad()
            outputs = Model1(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            total_loss += loss
            acc = accuracy(outputs,labels)
            total_acc += acc
            print(acc)
            print(loss)
        print('epoch : {} loss : {}  accuracy : {}'.format(i+1,total_loss/(j+1),total_acc/(j+1)))


