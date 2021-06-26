from data import  create_data_loader
from torch.optim import Adam
from utils.metrics import accuracy
import torch
import torch.nn as nn
from model import  get_model
from test import test
from params import *
from utils.utils import normalize

def train(args):
    device = torch.device("cuda")
    history = {
        'epoch': 0,
        'training_losses': [],
        'training_accuracy':[],
        'validation_losses': [],
        'validation_accuracy': []
    }
    state = {'state_dict': None,
             'optimizer': None,}

    train_gene, val_gene, test_gene = create_data_loader(args)
    Model1 = get_model(args)
    Model1.to(device)
    Model1.train()
    optimizer = Adam(Model1.parameters(),args.lr)
    print(Model1.parameters())
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
        print('Training : epoch : {} loss : {}  accuracy : {}'.format(i+1,total_loss/(j+1),total_acc/(j+1)))
        history['epoch']+=1
        history['training_losses'].append(total_loss/(j+1))
        history['training_accuracy'].append(total_acc/(j+1))
        acc,loss = test(args,val_gene,Model1)
        if acc>history['validation_accuracy'][-1]:
            state['state_dict'] = Model1.state_dict()
            state['optimizer'] = optimizer.state_dict()
            path = './checkpoint/state_best.pth'
            torch.save(state,path)
        print('Validation : epoch : {} loss : {}  accuracy : {}'.format(i + 1, loss, acc))
        history['validation_losses'].append(loss)
        history['validation_accuracy'].append(acc)
        if (i+1)%args.epoch_save==0:
            state['state_dict'] = Model1.state_dict()
            state['optimizer'] = optimizer.state_dict()
            torch.save(state,'./checkpoint/state_epoch_{}.pth'.format(i+1))



def contrastive_train(args):
    device = torch.device("cuda")

    train_gene, val_gene, test_gene = create_data_loader(args)
    args.model = 'Simclr'
    Model1 = get_model(args)
    Model1.to(device)
    Model1.train()
    optimizer_backbone = Adam([{'params':Model1.features_extractor.parameters(),'params' :Model1.head.parameters()}], args.lr)
    optimizer_classifier = Adam(Model1.classifier.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    for i in range (args.epoch):
        total_contrast_loss = 0
        total_classifier_loss = 0
        total_acc = 0
        # training loop
        for j,data in enumerate(train_gene):
            images , labels = data[0].to(device),data[1].to(device)

            #train backbone
            optimizer_backbone.zero_grad()
            images1 , images2 = CONTRASTIVE_AUG(images), CONTRASTIVE_AUG(images)
            z1, z2 = Model1(images1,output_mode='head'), Model1(images2,output_mode='head')
            contrast_loss = contrastive_loss(z1,z2,labels)
            contrast_loss.backward()
            optimizer_backbone.step()
            total_contrast_loss += contrast_loss

            #train classifier
            optimizer_backbone.zero_grad()
            with torch.no_grad():
                features = Model1.features_extractor(images)
            outputs = Model1.classifier(features)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer_classifier.step()
            total_classifier_loss +=loss
            acc = accuracy(outputs,labels)
            total_acc += acc
            # print(acc)
            # print(loss)
        print('Training : epoch : {} contrastive loss : {}  classifier loss : {}  accuracy : {}'.format(i+1,total_contrast_loss/(j+1),total_classifier_loss/(j+1),total_acc/(j+1)))
        acc,loss = test(args,val_gene)
        print('Validation : epoch : {} loss : {}  accuracy : {}'.format(i + 1, loss, acc))


def contrastive_loss(z1, z2, labels):
    device = z1.device
    z1, z2 = normalize(z1), normalize(z2)
    Z = torch.cat((z1,z2))
    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=43).type(torch.FloatTensor)
    labels_one_hot = labels_one_hot.repeat((2,1))

    corr = Z@torch.transpose(Z,0,1)
    corr = supress_inter_corr(corr).to(device)  # supress inter correlation Diag
    loglikelyhood = torch.log(torch.softmax(corr,axis=-1))
    positive_mask = labels_one_hot@torch.transpose(labels_one_hot,0,1).to(device)
    positive_sample_number = torch.sum(positive_mask,dim=-1,keepdim=True)
    Lout_i = -(torch.sum(loglikelyhood*positive_mask,dim=-1))/positive_sample_number
    return torch.mean(Lout_i)




def supress_inter_corr(corr):
    size =  corr.shape[0]
    mask = torch.eye(size)<1
    corr_supress = corr[mask]
    corr_supress = torch.reshape(corr_supress,[size,size-1])
    return corr_supress








