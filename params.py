from torchvision.transforms import ToTensor, Compose, Resize,RandomCrop

TRAINING_ROOT = './data/Training'
IMG_SIZE = (50,50)
AUG = Compose([RandomCrop(IMG_SIZE)])
TRANSFORM = Compose([
    Resize(),
    AUG,
    ToTensor()
])