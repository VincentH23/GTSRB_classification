import torchvision.models as models
import torch



class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Resnet_Simclr(torch.nn.Module):
    def __init__(self):
        super(Resnet_Simclr, self).__init__()
        self.features_extractor = models.resnet50(False)
        self.features_extractor.fc = Identity()
        self.classifier = torch.nn.Linear(2048, 43, bias=True)
        self.head = self.head = torch.nn.Sequential(
                torch.nn.Linear(2048, 2048),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(2048, 128)
            )

    def forward(self,x,output_mode = 'classifer'):
        output = self.features_extractor(x)
        if output_mode == 'classifier':
            return self.classifier(output)
        elif output_mode == 'head':
            return self.head(output)
        else :
            return output


def get_model(args):
    if args.model =='Resnet':
        Model1 = models.resnet50(False)
        torch.manual_seed(5)
        Model1.fc = torch.nn.Linear(2048, 43, bias=True)

    elif args.model =='Simclr':
        Model1 = Resnet_Simclr()

    return Model1


if __name__== '__main__':
    model = Resnet_Simclr()
    model.train()
    print(model.training,model.features_extractor.training)
    model.eval()
    print(model.training, model.features_extractor.training)
    print(model.state_dict().keys())
    A=model.state_dict()
    torch.save(model,'./model.pth')
    model = torch.load('./model.pth')
    model.train()
    print(model.training, model.features_extractor.training)
    model.eval()
    print(model.training, model.features_extractor.training)
    B=model.state_dict()
    print(id(A),id(B))
    print(B[0])





