from models.VariableLSTM import VariableLSTM
import torch
from torch import nn
from datasets import load_dataset
from thop import profile
from transformers import AutoTokenizer
from tqdm import tqdm

# Load in dataset
# dataset = load_dataset("cais/mmlu")

dataset = load_dataset("openai_humaneval")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# Create model
model = VariableLSTM(nlayer=2, input_size=128, hidden_size=32)
dummy_input = torch.randn([32, 128])

# Measure the flops and params
profile(model, inputs=(dummy_input,))


# %% Train the model on the dataset

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

dataloader = torch.utils.data.DataLoader(dataset['test'], batch_size=1, shuffle=True)

def tokenize_and_encode(data):
    return tokenizer(data, return_tensors='pt', padding=True, truncation=True)['input_ids'].float()

for epoch in tqdm(range(10)):
    for batch in dataloader:
        x = tokenize_and_encode(batch['prompt'])
        y = tokenize_and_encode(batch['canonical_solution'])

        # Poor mans front padding
        if x.shape[1] < 128:
            x = torch.cat((torch.zeros([x.shape[0], 128-x.shape[1]]), x), dim=1)
        
        model.zero_grad()
        # Loop through input tokens using a 128 moving window
        for i in range(x.shape[1] - 128):
            x_window = x[:, i:i+128]
            x_next = x[:, i+128]
            output = model(x_window)
            loss = loss_function(output, x_next)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loop through solution
        for i in range(y.shape[1]):
            if i < 128:
                input_concat = torch.cat((x[:, -128+i:], y[:, :i]), dim=1)
            else:
                input_concat = y[:, i-128:i]
            output = model(input_concat)
            loss = loss_function(output, y[:, i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# %% Test the model on the dataset
model.eval()

dataloader = torch.utils.data.DataLoader(dataset['test'], batch_size=1, shuffle=True)

for batch in dataloader:
    x = tokenize_and_encode(batch['prompt'])
    y = tokenize_and_encode(batch['canonical_solution'])

    # Poor mans front padding
    if x.shape[1] < 128:
        x = torch.cat((torch.zeros([x.shape[0], 128-x.shape[1]]), x), dim=1)
    
    # Now we generate
    y = torch.tensor([]).reshape([1, 0])
    for i in range(128):
        if i < 128:
            input_concat = torch.cat((x[:, -128+i:], y[:, :i]), dim=1)
        else:
            input_concat = y[:, i-128:i]
        output = model(input_concat)
        y = torch.cat((y, output.reshape(1,1)), dim=1)        
