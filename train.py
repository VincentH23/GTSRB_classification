from data import  create_data_loader, create_data_loader_contrastive
from torch.optim import Adam
from utils import accuracy, contrastive_loss, adjust_learning_rate
import torch
import torch.nn as nn
from model import  get_model
from test import test, test_contrastive
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
        'training_accuracy': [],
        'validation_losses': [],
        'validation_accuracy': []
    }
    state = {'state_dict': None,
             'optimizer': None}

    train_gene, val_gene, test_gene = create_data_loader_contrastive(args)
    Model1,Classifier = get_model(args)
    Classifier.to(device)
    Classifier.train()
    Model1.to(device)
    Model1.train()

    optimizer = torch.optim.SGD(Model1.parameters(), args.lr,momentum=0.9,weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    for i in range(args.epoch):
        total_loss = 0
        total_acc = 0
        total_loss1 = 0
        total_loss2 = 0
        # training loop

        for j, data in enumerate(train_gene):
            images1, images2, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            optimizer.zero_grad()
            images = torch.cat([images1,images2])
            feature, head = Model1(images)
            classifier = Classifier(feature[:batch_size])
            batch_size = images1.shape[0]
            h1,h2 = torch.split(head,[batch_size,batch_size])
            features = torch.cat([h1.unsqueeze(1), h2.unsqueeze(1)], dim=1)
            loss1 = contrastive_loss(features,labels)
            decay = max(0,1-(i/args.epoch)**2)
            loss2 = criterion(classifier,labels)
            loss = decay * loss1 + (1-decay)*0.1*loss2
            loss.backward()
            optimizer.step()
            total_loss1 += loss1
            total_loss2+= loss2
            print(loss)
            total_loss += loss
            total_acc += accuracy(classifier,labels)

        print(total_loss1/(1+j),total_loss2/(1+j),decay)
        print('Training : epoch : {} loss : {} acc : {} '.format(i + 1, total_loss / (j + 1),total_acc / (j + 1)))
        history['epoch'] += 1
        history['training_losses'].append(total_loss / (j + 1))
        history['training_accuracy'].append(total_acc / (j + 1))
        acc, loss = test_contrastive(args, val_gene, Model1)
        if acc > best_acc:
            best_acc = acc
            PATH = "./checkpoint/best_model_Contrastive" + ".pth.tar"
            torch.save(Model1.state_dict(), PATH)

        history['validation_losses'].append(loss)
        history['validation_accuracy'].append(acc)
        print('Validation : epoch : {} loss : {}  accuracy : {}'.format(i + 1, loss, acc))
        acc, loss = test_contrastive(args, test_gene, Model1)
        print(acc, loss)

        if (i + 1) % args.epoch_save == 0:
            state = {'state_dict': Model1.state_dict(),
                     'optimizer': optimizer.state_dict()}
            torch.save(state, './checkpoint/state_epoch_{}.pth.tar'.format(i + 1))
            torch.save(history, './checkpoint/history_epoch_{}.pth.tar'.format(i + 1))



