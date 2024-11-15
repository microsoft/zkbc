from models.VariableLSTM import VariableLSTM
import torch
from torch import nn
from datasets import load_dataset
from thop import profile
from transformers import AutoTokenizer
from tqdm import tqdm
import ezkl, json
from utils.export import export
import os

dataset = load_dataset("openai_humaneval")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Create model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out = self.lstm(x)
        last_time_step = lstm_out[1][0][-1]
        output = self.fc(last_time_step)
        return output
    
model = SimpleLSTM(input_size=128, hidden_size=32, output_size=1)
dummy_input = torch.randn([32, 128])
model(dummy_input)

# Measure the flops and params
profile(model, inputs=(dummy_input,))

# %% Train the model on the dataset
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if  torch.backends.mps.is_available() else "cpu")

model = model.to(device)

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
            output = model(x_window.to(device))
            loss = loss_function(output, x_next.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loop through solution
        for i in range(y.shape[1]):
            if i < 128:
                input_concat = torch.cat((x[:, -128+i:], y[:, :i]), dim=1)
            else:
                input_concat = y[:, i-128:i]
            output = model(input_concat.to(device))
            loss = loss_function(output, y[:, i].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# %% Test the model on the dataset
model.eval()
model = model.to("cpu")

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

    print(tokenizer.decode(y[0].long()))

# %% Export the model
os.makedirs('LSTM/logs', exist_ok=True)
LOGGING = False
pipstd = lambda fname: f" >> LSTM/logs/{fname}.log" if LOGGING else ""
SRS_PATH = '../kzgs/kzg%d.srs'

class HiddenStateLSTM(nn.Module):
    # We're going to use this to prove facts about just the hidden state
    def __init__(self, model):
        super(HiddenStateLSTM, self).__init__()
        self.lstm = model.lstm
    def forward(self, x):
        lstm_out = self.lstm(x)
        return lstm_out[1]

model_hidden = HiddenStateLSTM(model)    

example_input = next(iter(dataloader))['prompt']
x = tokenize_and_encode(batch['prompt'])
# add padding
if x.shape[1] < 128: x = torch.cat((torch.zeros([x.shape[0], 128-x.shape[1]]), x), dim=1)
x = x[:, -128:]
output = model_hidden(x)
export(model_hidden, input_array=dummy_input, onnx_filename='LSTM/lstm.onnx', input_filename='LSTM/lstm_temp.json')

# %% Setup and prove
os.system("ezkl table -M LSTM/lstm.onnx" + pipstd('setup'))
os.system("ezkl gen-settings -M LSTM/lstm.onnx --settings-path=LSTM/settings.json --input-visibility='public'" + pipstd('setup'))
os.system("ezkl calibrate-settings -M LSTM/lstm.onnx --settings-path=LSTM/settings.json --data=LSTM/lstm_temp.json" + pipstd('setup'))
settings = json.load(open('LSTM/settings.json', 'r'))
logrows = settings['run_args']['logrows']
ezkl.get_srs(srs_path=SRS_PATH % logrows, logrows=logrows)

os.system("ezkl compile-circuit -M LSTM/lstm.onnx -S LSTM/settings.json --compiled-circuit LSTM/lstm.ezkl" + pipstd('setup'))
os.system("ezkl gen-witness -M LSTM/lstm.ezkl -D LSTM/lstm_temp.json --output LSTM/witnessRandom.json" + pipstd('setup'))
os.system(f"ezkl setup -M LSTM/lstm.ezkl --srs-path={SRS_PATH % logrows} --vk-path=LSTM/lstm_vk.key --pk-path=LSTM/lstm_pk.key" + pipstd('setup'))
os.system(f"ezkl prove -M LSTM/lstm.ezkl --srs-path={SRS_PATH % logrows} --witness LSTM/witnessRandom.json --pk-path=LSTM/lstm_pk.key --proof-path=LSTM/lstm.proof" + pipstd('setup'))
os.system(f"ezkl verify --settings-path=LSTM/settings.json  --vk-path=LSTM/lstm_vk.key --proof-path=LSTM/lstm.proof --srs-path={SRS_PATH % logrows}" + pipstd('setup'))

