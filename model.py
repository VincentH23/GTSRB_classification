import torchvision.models as models
import torch
Model1 = models.resnet50(True)
for param in Model1.parameters():      # freeze the features extractor weights avoid to forget the knowledge from imagenet
    param.requires_grad =False
Model1.fc =torch.nn.Linear(2048, 43, bias=True)





