
from train import train
from test import evaluate, test
import argparse
from data import create_data_loader
from model import get_model
import torch



parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', default='train', help="Train or test phase")
parser.add_argument('--epoch',default=50,help="choose the number of epoch",type=int)
parser.add_argument('--batch',default=512,help= "batch size ",type=int)
parser.add_argument('--batch_test',default=2105,help= "batch size test",type=int)
parser.add_argument('--patience',default=10,help= "number of epochs where validation loss is allowed to increase", type=int)
parser.add_argument('--constrastive',default=False, help="add the contrastive learning",type=bool)
parser.add_argument('--loss_weight',type=list)
parser.add_argument('--epoch_save',default=5,help='number of epoch between each save',type = int)
parser.add_argument('--lr',default=0.0005,help='number of epoch between each save',type = float)
parser.add_argument('--model',default='Resnet')
parser.add_argument('--temperature',default=1, type=int)
parser.add_argument('--evaluate',default='testing',type = str)
parser.add_argument('--model_dir',default='./checkpoint/best_model_CE_temperature_1.pth.tar')
args = parser.parse_args()
print(args)

def main():
    if args.phase =='train':

        train(args)
        train_gene, val_gene, test_gene = create_data_loader(args)
        Model1 = get_model(args)
        acc,loss =test(args,test_gene,Model1)
        print('ab',acc,loss)
        dico = torch.load("./checkpoint/best_model_CE_temperature_1.pth.tar")
        Model1.load_state_dict(dico)
        acc, loss = test(args, test_gene, Model1)
        print('ab', acc, loss)
    else :
        evaluate(args)






if __name__== '__main__' :
    main()
