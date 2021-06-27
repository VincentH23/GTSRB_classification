from data import  create_data_loader
from torch.optim import Adam
from utils.metrics import accuracy
import torch
import torch.nn as nn
from model import  get_model
from test import test


def train(args):
    device = torch.device("cuda")
    train_gene, val_gene, test_gene = create_data_loader(args)
    Model1 = get_model(args)
    Model1.to(device)
    Model1.train()
    optimizer = Adam(Model1.parameters(),args.lr)
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    for i in range (args.epoch):
        total_loss = 0
        total_acc = 0
        # training loop
        for j,data in enumerate(train_gene):
            images , labels = data[0].to(device),data[1].to(device)
            optimizer.zero_grad()
            outputs = Model1(images)/args.temperature
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            total_loss += loss
            acc = accuracy(outputs,labels)
            total_acc += acc
            # print(acc)
            # print(loss)
        print('Training : epoch : {} loss : {}  accuracy : {}'.format(i+1,total_loss/(j+1),total_acc/(j+1)))
        acc,loss = test(args,val_gene,Model1)
        print('Validation : epoch : {} loss : {}  accuracy : {}'.format(i + 1, loss, acc))


def contrastive_train(args):
    device = torch.device("cuda")

    train_gene, val_gene, test_gene = create_data_loader(args)
    Model1 = get_model(args)
    Model1.to(device)
    Model1.train()



