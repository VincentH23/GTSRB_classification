from model import Model1

import argparse






parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', default='train', help="Train or test phase")
parser.add_argument('--epoch',default=50,help="choose the number of epoch",type=int)
parser.add_argument('--batch',default=512,help= "batch size ",type=int)
# parser.add_argument('--batch',default=512,help= "batch size ")
args = parser.parse_args()
print(args.batch)
