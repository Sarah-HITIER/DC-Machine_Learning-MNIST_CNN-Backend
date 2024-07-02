import os
import sys
import torch
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from backend.config import batch_size

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from utils.pretreatment_image import tf, permute_pixels

# 3. 2. Chargement du dataset MNIST
train_loader = DataLoader(
    datasets.MNIST('../data/', train=True, download=True, transform=tf),
    batch_size=batch_size, shuffle=True
)


# 3. 4. Construction du train et du test
def train(model, device, perm=torch.arange(0, 784).long(), n_epochs=1):
    model.train()    
    optimizer = torch.optim.AdamW(model.parameters())
    
    for epoch in range(n_epochs):
        for i, (data, target) in enumerate(train_loader):
            # send to device
            data, targets = data.to(device), target.to(device)

            # permute pixels
            data = permute_pixels(data, perm)

            # step
            optimizer.zero_grad()
            logits = model(data)
            
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"epoch={epoch}, step={i}: train loss={loss.item():.4f}")