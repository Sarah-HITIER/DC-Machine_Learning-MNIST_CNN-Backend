import torch
import numpy as np
from torchvision import transforms
from config import device

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


def permute_pixels(ts, perm=torch.arange(0, 784).long()):

    ts = ts.view(-1, 28*28)
    ts = ts[:, perm]
    ts = ts.view(-1, 1, 28, 28)
    
    return ts


def pretreatment_image(image):
    # Convert image to numpy array
    image = np.array(image, dtype=np.float32) / 255

    ts = tf(image)
    ts = ts.to(device)
    ts = permute_pixels(ts)

    return ts