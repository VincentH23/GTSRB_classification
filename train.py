from data import  create_data_loader, create_data_loader_contrastive
from torch.optim import Adam
from utils import accuracy, contrastive_loss
import torch
import torch.nn as nn
from model import  get_model
from test import test
import os

def train(args):
    device = torch.device("cuda")
    best_acc =0
    history = {
        'epoch': 0,
        'training_losses': [],
        'training_accuracy':[],
        'validation_losses': [],
        'validation_accuracy': []
    }
    state = {'state_dict': None,
             'optimizer': None}

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
        history['epoch']+=1
        history['training_losses'].append(total_loss/(j+1))
        history['training_accuracy'].append(total_acc/(j+1))
        acc,loss = test(args,val_gene,Model1)
        if acc >best_acc:
            best_acc =acc
            PATH = "./checkpoint/best_model_CE_temperature_"+str(args.temperature)+".pth.tar"
            torch.save(Model1.state_dict(), PATH)

        history['validation_losses'].append(loss)
        history['validation_accuracy'].append(acc)
        print('Validation : epoch : {} loss : {}  accuracy : {}'.format(i + 1, loss, acc))
        acc, loss = test(args, test_gene, Model1)
        print(acc, loss)
        if (i+1)%args.epoch_save==0:
            state = {'state_dict': Model1.state_dict(),
                     'optimizer': optimizer.state_dict() }
            torch.save(state,'./checkpoint/state_epoch_{}.pth.tar'.format(i+1))
            torch.save(history,'./checkpoint/history_epoch_{}.pth.tar'.format(i+1))
    acc,loss = test(args,test_gene,Model1)
    print (acc,loss)



def contrastive_train(args):

    #PRETRAIN
    device = torch.device("cuda")
    best_acc = 0
    history = {
        'epoch': 0,
        'training_losses': [],
        'validation_losses': []
    }
    state = {'state_dict': None,
             'optimizer': None}

    train_gene, val_gene, test_gene = create_data_loader_contrastive(args)
    Model1 = get_model(args)
    Model1.to(device)
    Model1.train()

    optimizer = torch.optim.Adam([{'params': Model1.features_extractor.parameters()},
                                  {'params': Model1.head.parameters()}], args.lr)
    for i in range(args.epoch):
        total_loss = 0
        # training loop
        for j, data in enumerate(train_gene):
            images1, images2, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            optimizer.zero_grad()
            images = torch.cat([images1,images2])
            _, H = Model1(images,mode='head')
            h1,h2 = torch.split(H,[args.batch,args.batch])
            loss = contrastive_loss(h1,h2,labels)
            loss.backward()
            optimizer.step()
            total_loss += loss
        print('Training : epoch : {} loss : {}  '.format(i + 1, total_loss / (j + 1)))
        history['epoch'] += 1
        history['training_losses'].append(total_loss / (j + 1))


        ##To complete
        acc, loss = test(args, val_gene, Model1)

        history['validation_losses'].append(loss)
        history['validation_accuracy'].append(acc)
        print('Validation : epoch : {} loss : {}  accuracy : {}'.format(i + 1, loss, acc))
        acc, loss = test(args, test_gene, Model1)
        print(acc, loss)
        if (i + 1) % args.epoch_save == 0:
            state = {'state_dict': Model1.state_dict(),
                     'optimizer': optimizer.state_dict()}
            torch.save(state, './checkpoint/state_epoch_{}.pth.tar'.format(i + 1))
            torch.save(history, './checkpoint/history_epoch_{}.pth.tar'.format(i + 1))
    acc, loss = test(args, test_gene, Model1)
    print(acc, loss)


