import os
import random
import torch
from params import *
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize
import PIL.Image as Image

def data_split():   # split training / validation set same distribution
    data = {'training' : [],
        'validation' : []}
    labels = os.listdir(TRAINING_ROOT)

    for label in labels:
        samples = os.listdir(TRAINING_ROOT+'/'+label)
        samples = [label+'/'+s for s in samples]
        random.seed(10)
        random.shuffle(samples)
        n = len(samples)
        data['training'] += samples[:int(0.7*n)]
        data['validation'] += samples[int(0.7 * n):]
    return data

def create_data_loader():
    data = data_split()
    print(data)

    data_train = Custom_data_set(data['training'],TRAINING_ROOT,transform=TRANSFORM)
    generator = DataLoader(data_train,512,shuffle=True)
    print(next(iter(generator))[1].shape)



class Custom_data_set(Dataset):

    def __init__(self,list_data,root_dir,transform=None,csv=None):
        self.root_dir = root_dir
        self.list_data = list_data
        self.csv = csv
        self.transform = transform

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, idx):

        img_path = self.root_dir+'/'+self.list_data[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        label = float(self.list_data[idx].split('/')[0])+1

        return image , label



create_data_loader()




