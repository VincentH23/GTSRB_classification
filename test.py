from data import  create_data_loader
from torch.optim import Adam
from model import  get_model
from utils.metrics import  accuracy
import torch

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
    model.load_state_dict(torch.load(args.model_dir))
    if args.evaluate =='testing':
        _,_,test_generator = create_data_loader(args)
        acc, loss = test(args,test_generator,model)
        print('Testing : loss : {}  accuracy : {}'.format( loss, acc))
    else :
        _, validation_generator,_ = create_data_loader(args)
        acc, loss = test(args, validation_generator, model)
        print('Validation : loss : {}  accuracy : {}'.format(loss, acc))

