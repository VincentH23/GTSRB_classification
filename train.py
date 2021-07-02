from data import  create_data_loader
from torch.optim import Adam
from utils import accuracy
import torch
import torch.nn as nn
from model import  get_model
from test import test

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





