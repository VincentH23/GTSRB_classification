import torch
import matplotlib.pyplot as plt


def accuracy(output,target):
    compar = torch.argmax(output,1)==target
    compar= compar.type(torch.FloatTensor)
    acc = torch.mean(compar)
    return acc

def contrastive_loss(h1,h2,labels):
    labels = labels.view(-1,1)
    mask =
    #to complete
    pass


class Prepocessing:
    "change gamma for image with low luminance"

    def __init__(self):
        pass

    def __call__(self, x):
        low_pixels = torch.mean((torch.mean(x,axis=0)<0.2).type(torch.FloatTensor))
        if low_pixels.item() >= 0.5:
            I = torch.pow(x,0.5)
            return I
        return x
