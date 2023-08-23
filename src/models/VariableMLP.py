import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, nlayer=1):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(256, 512)    
        self.hidden_layers = nn.ModuleList()
        for _ in range(nlayer):
            self.hidden_layers.append(nn.Linear(512, 512))
        self.output_layer = nn.Linear(512, 1) 

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        x = self.output_layer(x)
        return x