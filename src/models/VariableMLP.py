import torch
import torch.nn as nn
import torch.nn.functional as F

class VariableMLP(nn.Module):
    def __init__(self, nlayer=1, hidden_size=256, input_size=256, output_size=1):
        super(VariableMLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)    
        self.hidden_layers = nn.ModuleList()
        for _ in range(nlayer):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, output_size) 

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        x = self.output_layer(x)
        return x