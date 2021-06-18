from torchvision.transforms import ToTensor, Compose, Resize,RandomCrop, RandomRotation, RandomGrayscale, RandomChoice,GaussianBlur
from utils.utils import Add_Noise_Transform
TRAINING_ROOT = './data/Training'
TESTING_ROOT = './data/Testing'
TESTING_CSV = 'GT-final_test.csv'
IMG_SIZE = (50,50)
AUG = Compose([RandomCrop(IMG_SIZE),RandomRotation(degrees=(0,180)),RandomGrayscale(),GaussianBlur(5, sigma=(0.1, 2.0)),Add_Noise_Transform()])
TRANSFORM = Compose([
    Resize(IMG_SIZE),
    AUG,
    ToTensor()
])

TRANSFORM_TESTING = Compose([
    Resize(IMG_SIZE),
    ToTensor()
])


CONTRASTIVE_AUG = RandomChoice([RandomRotation(degrees=(0,180)),RandomGrayscale(),GaussianBlur(5, sigma=(0.1, 2.0)),Add_Noise_Transform()])