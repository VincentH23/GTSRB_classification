from torchvision.transforms import ToTensor, Compose, Resize,RandomCrop, Normalize, RandomGrayscale, RandomHorizontalFlip, RandomChoice,RandomVerticalFlip, RandomRotation
from utils import  Prepocessing
TRAINING_ROOT = './data/Training'
TESTING_ROOT = './data/Testing/Images'
TESTING_CSV = './data/Testing/GT-final_test.csv'
IMG_SIZE = (50,50)
AUG = Compose([RandomCrop(IMG_SIZE),RandomGrayscale(3),RandomHorizontalFlip()])
TRANSFORM = Compose([
    Resize(IMG_SIZE),
    ToTensor(),
    Prepocessing(),
    AUG,
    Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225], )
])

TRANSFORM_TESTING = Compose([
    Resize(IMG_SIZE),
    ToTensor(),
    # Prepocessing(),
    Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225], )
])

TRANSFORM_CONTRASTIVE = Compose([
    Resize(IMG_SIZE),
    ToTensor()
])


CONTRASTIVE_AUG = RandomChoice([RandomRotation(degrees=(0,180)),
                                RandomGrayscale(3),RandomHorizontalFlip(),
                                RandomVerticalFlip(),RandomCrop(IMG_SIZE)])