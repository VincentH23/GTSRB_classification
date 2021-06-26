from torchvision.transforms import ToTensor, Compose, Resize,RandomCrop, RandomRotation, RandomGrayscale, RandomChoice
from utils.utils import Add_Noise_Transform
TRAINING_ROOT = './data/Training'
TESTING_ROOT = './data/Testing'
TESTING_CSV = 'GT-final_test.csv'
IMG_SIZE = (112,112)
AUG = Compose([RandomCrop(IMG_SIZE),RandomRotation(degrees=(0,180)),RandomGrayscale(3)])
TRANSFORM = Compose([
    Resize(IMG_SIZE),
    ToTensor(),
    AUG,
])

TRANSFORM_TESTING = Compose([
    Resize(IMG_SIZE),
    ToTensor()
])


CONTRASTIVE_AUG = RandomChoice([RandomRotation(degrees=(0,180)),RandomGrayscale(3)])


# train_transform = transforms.Compose([
#         transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomApply([
#             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#         ], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.ToTensor(),
#         normalize,
#     ])