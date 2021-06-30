from torchvision.transforms import ToTensor, Compose, Resize,RandomCrop, Normalize
from utils.utils import  Prepocessing
TRAINING_ROOT = './data/Training'
TESTING_ROOT = './data/Testing/Images'
TESTING_CSV = './data/Tsting/GT-final_test.csv'
IMG_SIZE = (50,50)
AUG = Compose([RandomCrop(IMG_SIZE)])
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
    Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])