import torch
import torch.nn as nn

# 3. 3. Construction d’un modèle de convolution neural network
class ConvNet(nn.Module):
    def __init__(self, input_size, n_kernels, output_size):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=4,
                               kernel_size=3,
                               padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=4,
                               out_channels=8,
                               kernel_size=3,
                               padding=1)
        self.fc = nn.Linear(8 * 7 * 7, 10)
        # self.net = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=n_kernels, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(in_channels=n_kernels, out_channels=n_kernels, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Flatten(),
        #     nn.Linear(in_features=n_kernels * 4 * 4, out_features=50),
        #     nn.Linear(in_features=50, out_features=output_size),
        # )
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 8 * 7 * 7)
        return self.fc(x)
        # return self.net(x)