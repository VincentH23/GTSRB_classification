import torchvision.models as models
import torch
from torch.nn.functional import softmax, normalize


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class TTA(torch.nn.Module):
    def __init__(self,model,transforms):
        super(TTA, self).__init__()
        self.model = model
        self.transforms = transforms

    def forward(self , x):
        merge =[]
        for transform in self.transforms :
            output = self.model(transform(x))
            proba = softmax(output,dim=-1)
            merge.append(proba)
        return torch.mean(torch.stack(merge))

class Resnet_Contrastive(torch.nn.Module):
    def __init__(self):
        super(Resnet_Contrastive, self).__init__()
        self.features_extractor = models.resnet50(False)
        self.features_extractor.fc = Identity()
        self.classifier = torch.nn.Linear(2048, 43, bias=True)
        self.head = torch.nn.Sequential(torch.nn.Linear(2048,128,bias=True),torch.nn.ReLU(),torch.nn.Linear(128,128,bias=True))

    def forward(self, x ):
        features = self.features_extractor(x)
        head = normalize(self.head(features),dim=1)
        classifier = self.classifier(features)
        return features, head , classifier





def get_model(args):
    if args.model =='Resnet':
        Model1 = models.resnet50(False)
        torch.manual_seed(5)
        Model1.fc = torch.nn.Linear(2048, 43, bias=True)

    elif args.model =='Contrastive':
        Model1 = Resnet_Contrastive()

    return Model1




if __name__=='__main__':
    print('a')
    Model1 = Resnet_Contrastive()
    # optimizer = torch.optim.Adam([{'params': Model1.features_extractor.parameters()},
    #                  {'params': Model1.head.parameters()}],0.01)
    # print((optimizer))
    # Model1(x,mode='head')
    print(Model1.features_extractor.conv1.weight[0][0,0])
    for param in Model1.parameters():
        print(param)