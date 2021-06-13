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