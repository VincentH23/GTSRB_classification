from data import  create_data_loader
from model import  get_model
from utils import  accuracy
import torch
from utils import contrastive_loss

import torch.nn as nn

def test(args,generator,model):
    device = torch.device("cuda")
    model.to(device)
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


def evaluate(args):
    model = get_model(args)
    dico = torch.load(args.model_dir)
    model.load_state_dict(dico)
    if args.evaluate =='testing':
        _,_,test_generator = create_data_loader(args)
        acc, loss = test(args,test_generator,model)
        print('Testing : loss : {}  accuracy : {}'.format( loss, acc))
    else :
        _, validation_generator,_ = create_data_loader(args)
        acc, loss = test(args, validation_generator, model)
        print('Validation : loss : {}  accuracy : {}'.format(loss, acc))


def test_contrastive(args,generator,model):
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    total_loss = 0
    for j, data in enumerate(generator):
        images1, images2, labels = data[0].to(device), data[1].to(device), data[2].to(device)
        images = torch.cat([images1, images2])
        with torch.no_grad():
            _, H = model(images, mode='head')
            h1, h2 = torch.split(H, [args.batch, args.batch])
            loss = contrastive_loss(h1, h2, labels)
            total_loss += loss

    return total_loss/(1+j)
