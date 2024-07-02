import os
import sys
import torch
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from backend.config import batch_size

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from utils.pretreatment_image import permute_pixels, tf

# 3. 2. Chargement du dataset MNIST
test_loader = DataLoader(
    datasets.MNIST('../data/', train=False, transform=tf),
    batch_size=batch_size, shuffle=False
)


# 3. 4. Construction du test
def test(model, device, perm=torch.arange(0, 784).long()):
    model.eval()
    
    test_loss = 0
    correct = 0
    
    for data, targets in test_loader:
        # send to device
        data, targets = data.to(device), targets.to(device)
        
        # permute pixels
        data = permute_pixels(data, perm)
        
        # metrics
        logits = model(data)
        test_loss += F.cross_entropy(logits, targets, reduction='sum').item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == targets).sum()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    
    print(f"test loss={test_loss:.4f}, accuracy={accuracy:.4f}")
