from torchvision.transforms import ToTensor, Compose, Resize,RandomCrop

TRAINING_ROOT = './data/Training'
TESTING_ROOT = './data/Testing'
TESTING_CSV = 'GT-final_test.csv'
IMG_SIZE = (50,50)
AUG = Compose([RandomCrop(IMG_SIZE)])
TRANSFORM = Compose([
    Resize(IMG_SIZE),
    AUG,
    ToTensor()
])