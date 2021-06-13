import torch


def accuracy(output,target):
    compar = torch.argmax(output,1)==target
    compar= compar.type(torch.FloatTensor)
    acc = torch.mean(compar)
    return acc