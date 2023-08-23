import torch
import torch.nn as nn
import torch.nn.functional as F

class VariableCNN(nn.Module):
    def __init__(self, nlayer = 1, output_size=10):
        super(VariableCNN, self).__init__()
        self.nlayer = nlayer
        self.convs = nn.ModuleList([nn.Conv2d(1, 2, kernel_size=3, padding=1)])
        for i in range(1, nlayer):
            self.convs.append(nn.Conv2d(2 + (i-1), 2 + i, kernel_size=3, padding=1))
        self.fc = nn.Linear(28*28*(2 + (nlayer-1)), output_size)

    def forward(self, x):
        for i in range(self.nlayer):
            x = torch.relu(self.convs[i](x))
        x = x.view(-1, 28*28*(2 + (self.nlayer-1)))
        x = self.fc(x)
        return x
