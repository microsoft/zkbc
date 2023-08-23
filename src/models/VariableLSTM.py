import torch
import torch.nn as nn

class VariableLSTM(nn.Module):
    def __init__(self, nlayer=1, input_size=128, hidden_size=256, output_size=1):
        super(VariableLSTM, self).__init__()
        
        # This keeps track of the LSTM layers
        self.lstms = nn.ModuleList()
        
        # The first LSTM layer always takes the raw input
        self.lstms.append(nn.LSTM(input_size, hidden_size))
        
        # Additional LSTM layers take the hidden state of the previous layer as input
        for _ in range(1, nlayer):
            self.lstms.append(nn.LSTM(hidden_size, hidden_size))
        
        # Fully connected layer to produce the output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x should be of shape (seq_len, batch, input_size)
        
        # Passing input through all LSTM layers
        for lstm in self.lstms:
            x, _ = lstm(x)
        
        # Only take the output of the last sequence step
        x = x[-1]
        x = self.fc(x)        
        return x


if __name__ == '__main__':
    # Example usage:
    # Assuming you have input data of shape (10, 1, 128) i.e. (sequence length, batch size, feature size)
    data = torch.randn(10, 1, 128)
    model = VariableLSTM(nlayer=3, output_size=64)
    output = model(data)
    # This will give an output tensor of shape (32, 1), given that the output size is set to 1
