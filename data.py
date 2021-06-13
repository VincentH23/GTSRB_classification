import os
import random
import torch
from params import *
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize
import PIL.Image as Image
import pandas as pd

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
    data_val = Custom_data_set(data['validation'], TRAINING_ROOT)
    val_generator = DataLoader(data_val, len(data['validation']))
    test_csv = pd.read_csv(TESTING_ROOT+'/'+TESTING_CSV)
    data_test = Custom_data_set(data['validation'], TRAINING_ROOT,csv=test_csv)
    test_generator = DataLoader(data_test,12630)
    return train_generator, val_generator, test_generator


class Custom_data_set(Dataset):

    def __init__(self,list_data,root_dir,transform=None,csv=None):
        self.root_dir = root_dir
        self.list_data = list_data
        self.csv = csv
        self.transform = transform

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, idx):

        if self.csv :
            img_path = self.root_dir + '/' + self.csv.iloc[idx]['Filename']
            label = int(self.csv.iloc[idx]['ClassId'])
        else :
            img_path = self.root_dir+'/'+self.list_data[idx]
            label = int(self.list_data[idx].split('/')[0])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image , label

#
#
# gen,_,_ = create_data_loader()
# for i,data in enumerate(gen):
#     print(data)




