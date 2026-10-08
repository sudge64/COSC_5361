import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)   # (12, 28, 28)
        self.pool  = nn.MaxPool2d(2, 2)   # (12, 14, 14)
        self.conv2 = nn.Conv2d(12, 24, 5)  # (24, 10, 10) → pool → (24, 5, 5)
        self.fc1   = nn.Linear(24 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.fc3(x)
