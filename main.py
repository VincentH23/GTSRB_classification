from model import Model1
import data
from train import train
from test import test
import argparse






parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', default='train', help="Train or test phase")
parser.add_argument('--epoch',default=50,help="choose the number of epoch",type=int)
parser.add_argument('--batch',default=512,help= "batch size ",type=int)
parser.add_argument('--patience',default=10,help= "number of epochs where validation loss is allowed to increase", type=int)
parser.add_argument('--constrastive',default=False, help="add the contrastive learning",type=bool)
parser.add_argument('--loss_weight',type=list)
parser.add_argument('--epoch_save',default=5,help='number of epoch between each save',type = int)
parser.add_argument('--lr',default=0.0005,help='number of epoch between each save',type = float)
args = parser.parse_args()
print(args)

def main():
    if args.phase =='train':

        train(args)

    else :
        test(args)





if __name__== '__main__' :
    main()
