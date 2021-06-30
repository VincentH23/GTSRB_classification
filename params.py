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
    Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

TRANSFORM_TESTING = Compose([
    Resize(IMG_SIZE),
    ToTensor(),
    # Prepocessing(),
    Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

TRANSFORM_CONTRASTIVE = Compose([
    Resize(IMG_SIZE),
    ToTensor()
])


CONTRASTIVE_AUG = RandomChoice([RandomRotation(degrees=(0,180)),
                                RandomGrayscale(3),RandomHorizontalFlip(),
                                RandomVerticalFlip(),RandomCrop(IMG_SIZE)])