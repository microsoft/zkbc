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

class SparseCNN(nn.Module):
    def __init__(self):
        super(SparseCNN, self).__init__()
        # Single convolutional layer with a large stride
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5, stride=3)

        # Initialize weights to be sparse
        self.conv1.weight.data *= torch.rand(self.conv1.weight.shape) < 0.1
        self.conv2.weight.data *= torch.rand(self.conv2.weight.shape) < 0.1

        conv_output_size = self._get_conv_output([1, 3, 112, 112])
        self.fc = nn.Linear(conv_output_size, 40)

    def _get_conv_output(self, shape):
        """Get's the size of the linear layer"""
        batch_size = shape[0]
        input = torch.autograd.Variable(torch.rand(batch_size, *shape[1:]))
        output_feat = self.conv2(self.conv1(input))
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten the output
        x = self.fc(x)
        return torch.sigmoid(x) # Sigmoid for binary classification
    
    # def maintain_sparsity(self):
    #     conv_layer.weight *= (conv_layer.weight < sparsity_threshold).float()
    #     return self.conv1.weight.data

# Example usage
model = SparseCNN()
input_tensor = torch.randn(1, 3, 112, 112)
output = model(input_tensor)
print(output.size()) # Should be torch.Size([1, 40])
print(model._get_conv_output([1, 3, 112, 112]))


from thop import profile
macs, params = profile(model, inputs=(input_tensor, ))
print(f"Total model params: {params}\nTotal model MACs (FLOPs): {macs}")
