import torch

def add_noise(image):
    noise = torch.randn(image.shape)
    return image + noise

def callbacks(state,val_loss):
    if state['loss'] < val_loss:       # state include stop and previous val loss
        state['stop'] +=1               # increase stop when val loss increase to avoid overfitting
    else :
        state['stop'] = 0
    state['loss'] = val_loss

def normalize(z):
    return z/torch.linalg.norm(z,dim=1,keepdim=True)

class Add_Noise_Transform:
    """Rotate by one of the given angles."""

    def __init__(self):
        pass

    def __call__(self, x):
        return add_noise(x)

