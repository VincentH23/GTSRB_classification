import os
import random
from params import *
from torch.utils.data import Dataset, DataLoader
import torch
import PIL.Image as Image
import pandas as pd
import matplotlib.pyplot as plt
from model import Resnet_Contrastive

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


def create_data_loader(args):
    data = data_split()
    data_train = Custom_data_set(data['training'],TRAINING_ROOT,transform=TRANSFORM)
    train_generator = DataLoader(data_train,args.batch,shuffle=True)
    data_val = Custom_data_set(data['validation'], TRAINING_ROOT,transform=TRANSFORM_TESTING)
    val_generator = DataLoader(data_val, 2354)
    test_csv = pd.read_csv(TESTING_CSV,sep=';')
    data['testing'] = list(list(test_csv['Filename']))
    data_test = Custom_data_set(data['testing'], TESTING_ROOT ,csv=test_csv,transform=TRANSFORM_TESTING,use_csv=True)
    test_generator = DataLoader(data_test,args.batch_test)
    return train_generator, val_generator, test_generator


def create_data_loader_contrastive(args):
    data = data_split()
    data_train = Contrastive_dataset(data['training'],TRAINING_ROOT,transform=TRANSFORM_CONTRASTIVE,aug=CONTRASTIVE_AUG)
    train_generator = DataLoader(data_train,args.batch,shuffle=True)
    data_val = Contrastive_dataset(data['validation'], TRAINING_ROOT,transform=TRANSFORM_TESTING,aug=CONTRASTIVE_AUG)
    val_generator = DataLoader(data_val,args.batch)
    test_csv = pd.read_csv(TESTING_CSV,sep=';')
    data['testing'] = list(list(test_csv['Filename']))
    data_test = Custom_data_set(data['testing'], TESTING_ROOT ,csv=test_csv,transform=TRANSFORM_TESTING,use_csv=True)
    test_generator = DataLoader(data_test,args.batch_test)
    return train_generator, val_generator, test_generator


class Custom_data_set(Dataset):

    def __init__(self,list_data,root_dir,transform=None,csv=None,use_csv=False):
        self.root_dir = root_dir
        self.list_data = list_data
        self.csv = csv
        self.transform = transform
        self.use_csv = use_csv

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, idx):

        if self.use_csv:

            img_path = self.root_dir + '/' + self.csv.iloc[idx]['Filename']
            label = int(self.csv.iloc[idx]['ClassId'])
        else :
            img_path = self.root_dir+'/'+self.list_data[idx]
            label = int(self.list_data[idx].split('/')[0])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        return image , label


class Contrastive_dataset(Dataset):

    def __init__(self,list_data,root_dir,transform=None,aug=None, csv=None,use_csv=False):
        self.root_dir = root_dir
        self.list_data = list_data
        self.csv = csv
        self.transform = transform
        self.use_csv = use_csv
        self.aug =aug
        self.normalize = Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, idx):

        if self.use_csv:

            img_path = self.root_dir + '/' + self.csv.iloc[idx]['Filename']
            label = int(self.csv.iloc[idx]['ClassId'])
        else :
            img_path = self.root_dir+'/'+self.list_data[idx]
            label = int(self.list_data[idx].split('/')[0])
        image = Image.open(img_path)
        image = self.transform(image)
        image1 = self.aug(image)
        # image1 = self.normalize(image1)
        image2 = self.aug(image)
        # image2 = self.normalize(image2)
        return image1 , image2, label

if __name__=='__main__':
    data = data_split()
    data_train = Contrastive_dataset(data['training'], TRAINING_ROOT, transform=TRANSFORM_CONTRASTIVE, aug= CONTRASTIVE_AUG)
    train_generator = DataLoader(data_train, 50, shuffle=True)
    I =  next(iter(train_generator))
    # print (I[0].shape,I[1].shape,I[2].shape)
    # plt.imshow(I[0][0].permute(1,2,0))
    # plt.show()
    # plt.imshow(I[1][0].permute(1, 2, 0))
    # plt.show()
    # J = torch.cat(I[:2])
    # # plt.imshow(J[0].permute(1, 2, 0))
    # # plt.show()
    # # plt.imshow(J[20].permute(1, 2, 0))
    # # plt.show()
    # A,B = torch.split(J,[20,20])
    # plt.imshow(A[0].permute(1, 2, 0))
    # plt.show()
    # plt.imshow(B[0].permute(1, 2, 0))
    # plt.show()
    # Model = Resnet_Contrastive()
    # A = Model(I[0],mode='head')
    # print(A[1].shape)
    M = I[2].view(-1,1)
    print(M)
    print(M.T)
    print(torch.eq(M,M.T))


