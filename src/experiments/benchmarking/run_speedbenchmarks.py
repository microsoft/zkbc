# While speed tests can be run for each model across full datasets for more accuracy results, 
# the speed of the proving system across a range of models can be tested with the following script. 
# To minimise script complexity, random model weights and input values are used (while preserving shape and type) 
# as input value mostly do not effect proof speeds or sizes.

import torch.nn as nn
import torch
import os, json, time

WORKING_DIR = 'files/'
os.makedirs(WORKING_DIR, exist_ok=True)

def save_model_and_output(model, input_shape, modelname):
    """This function will take a model and it's input shape, and save the model and a random input to file."""
    # Serialize data into file:
    x = 0.1*torch.rand(1,*input_shape, requires_grad=True)
    model.eval()
    y = model(x)
    input_data = dict(input_data = [((x).detach().numpy()).reshape([-1]).tolist()],
                output_data = [((o).detach().numpy()).reshape([-1]).tolist() for o in y])
    json.dump( input_data, open( f'{WORKING_DIR}input_{modelname}.json', 'w' ) )

    # Export the model
    torch.onnx.export(model, x, f"{WORKING_DIR}{modelname}.onnx", export_params=True, do_constant_folding=True, input_names = ['input'], output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})

def prove_and_measure(model, input_shape, modelname):
    """This function will take a model to run a proof and verification on a random input. It will save the results to a log file and measure all the file sizes."""
    if modelname != 'nanoGPT':
        save_model_and_output(model, input_shape, modelname)

    # Run the proving stack
    os.system(f"ezkl gen-settings -M {WORKING_DIR}{modelname}.onnx --settings-path={WORKING_DIR}settings_{modelname}.json > {WORKING_DIR}{modelname}.log")
    os.system(f"ezkl calibrate-settings -M {WORKING_DIR}{modelname}.onnx -D {WORKING_DIR}input_{modelname}.json --settings-path={WORKING_DIR}settings_{modelname}.json --target=resources > {WORKING_DIR}{modelname}.log")
    os.system(f"ezkl compile-circuit -M {WORKING_DIR}{modelname}.onnx -S {WORKING_DIR}settings_{modelname}.json --compiled-circuit {WORKING_DIR}{modelname}.ezkl > {WORKING_DIR}{modelname}.log")
    os.system(f"ezkl gen-witness -M {WORKING_DIR}{modelname}.ezkl -D {WORKING_DIR}input_{modelname}.json --output {WORKING_DIR}witness_{modelname}.json > {WORKING_DIR}{modelname}.log")
    os.system(f"ezkl get-srs -S {WORKING_DIR}settings_{modelname}.json > {WORKING_DIR}{modelname}.log")
    os.system(f"ezkl setup -M {WORKING_DIR}{modelname}.ezkl --vk-path={WORKING_DIR}vk_{modelname}.key --pk-path={WORKING_DIR}pk_{modelname}.key > {WORKING_DIR}{modelname}.log")
    t0 = time.time()
    os.system(f"ezkl prove -M {WORKING_DIR}{modelname}.ezkl --proof-path={WORKING_DIR}proof_{modelname}.proof --pk-path={WORKING_DIR}pk_{modelname}.key --witness {WORKING_DIR}witness_{modelname}.json > {WORKING_DIR}{modelname}.log")
    t1 = time.time()
    os.system(f"ezkl verify --proof-path={WORKING_DIR}proof_{modelname}.proof --vk-path={WORKING_DIR}vk_{modelname}.key --settings-path={WORKING_DIR}settings_{modelname}.json > {WORKING_DIR}{modelname}.log")
    t2 = time.time()

    # Measure file sizes
    pk_size = os.path.getsize(f"{WORKING_DIR}pk_{modelname}.key")
    vk_size = os.path.getsize(f"{WORKING_DIR}vk_{modelname}.key")
    proof_size = os.path.getsize(f"{WORKING_DIR}proof_{modelname}.proof")
    wall_prooftime = t1-t0 # This can be replaced with the ezkl reported time
    wall_verifytime = t2-t1 

    os.system(f"echo {modelname}, {pk_size}, {vk_size}, {proof_size}, {wall_prooftime}, {wall_verifytime} > {WORKING_DIR}{modelname}.log")

    # Clean up
    os.system(f"rm {WORKING_DIR}input_{modelname}.json {WORKING_DIR}{modelname}.onnx {WORKING_DIR}{modelname}.ezkl {WORKING_DIR}vk_{modelname}.key {WORKING_DIR}pk_{modelname}.key")

    return pk_size, vk_size, proof_size, wall_prooftime, wall_verifytime


# From here we're just going to run through a sequence of models.
results_dataframe = []

# %% MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP(10, 50, 10)

results_dataframe.append(
    prove_and_measure(model, (10,), 'mlp'))

# %% CNN

class CNN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        return x

model = CNN(3, 10)
inout_shape = (3, 32, 32)
results_dataframe.append(
    prove_and_measure(model, inout_shape, 'cnn'))


# %% VAE Decoder 
# Full decode code where a VAE is trained and only the decoder is extracted can be found in `VAE_example.py`, here we show only the randomly initialised decoder.

class SmallVAE(nn.Module):
    def __init__(self):
        super(SmallVAE, self).__init__()

        self.fc_decode = nn.Linear(128, 32 * 16 * 16)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Output: 16 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),  # Output: 3 x 64 x 64
            nn.Sigmoid()
        )

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 32, 16, 16)
        return self.decoder(h)

    def forward(self, x):
        return self.decode(x)
    
model = SmallVAE()
inout_shape = (128,)
results_dataframe.append(
    prove_and_measure(model, inout_shape, 'vae_decoder'))


# %% LSTM
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)

    def forward(self, x):
        return self.lstm(x)[1] # This is not the standard way of using an LSTM (normally you would take the last output of the hidden state and add a linear layer, but this makes proving harder)
    
model = SimpleLSTM(input_size=128, hidden_size=32)
input_shape = (32, 128)

x = 0.1*torch.rand(1,*input_shape, requires_grad=True)
model(x)
results_dataframe.append(
    prove_and_measure(model, input_shape, 'lstm'))



# %% Linear Regression
# For more details on the training and conversion of these non pytorch models, see `non_NN_example.py`

from sklearn.linear_model import LinearRegression
import numpy as np
from hummingbird.ml import convert
from sklearn.ensemble import RandomForestClassifier
import sk2torch


X = np.random.rand(100, 100)
y = np.random.rand(100, 1)
reg = LinearRegression().fit(X, y)
circuit = convert(reg, "torch", X[:1]).model
input_shape = (100,)
results_dataframe.append(
    prove_and_measure(circuit, input_shape, 'linear_regression'))



# %% Random Forest

clr = RandomForestClassifier(max_depth=3, n_estimators=10)
X = np.random.rand(100, 100)
y = np.random.randint(10, size=(100, 1))
clr.fit(X, y)

trees = []
for tree in clr.estimators_:
    trees.append(sk2torch.wrap(tree))

class RandomForest(nn.Module):
    def __init__(self, trees):
        super(RandomForest, self).__init__()
        self.trees = nn.ModuleList(trees)

    def forward(self, x):
        out = self.trees[0](x)
        for tree in self.trees[1:]:
            out += tree(x)
        return out / len(self.trees)
    
model = RandomForest(trees)
input_shape = (100,)
results_dataframe.append(
    prove_and_measure(model, input_shape, 'random_forest'))

# %% SVC

from sklearn.svm import SVC, LinearSVC

# sk_model = SVC(probability=True, kernel='sigmoid') # both work
sk_model = LinearSVC()

X = torch.rand(100, 100)
y = torch.randint(10, size=(100, 1))

sk_model.fit(X, y)
model = sk2torch.wrap(sk_model)

# We convert to float32 to be lazy in the above function (not used elsewhere in repo)
model.weights = torch.nn.Parameter(model.weights.type(torch.float32))
model.biases = torch.nn.Parameter(model.biases.type(torch.float32))

input_shape = (100,)
results_dataframe.append(
    prove_and_measure(model, input_shape, 'svc'))


import pandas as pd

df = pd.DataFrame(results_dataframe, columns=['model', 'pk_size', 'vk_size', 'proof_size', 'wall_prooftime', 'wall_verifytime'])
df.to_csv(f'{WORKING_DIR}results.csv', index=False)


#################### Things get slower from here


model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
input_shape = (3, 224, 224)
results_dataframe.append(
    prove_and_measure(model, input_shape, 'mobilenet_v2'))







# %% nanoGPT
# We're gonna autogenerate the nanoGPT code from the ezkl repo
# https://github.com/zkonduit/ezkl/blob/ddbcc1d2d8c010eac2812572b31d2d5c13d5e6f0/examples/onnx/nanoGPT/gen.py

os.system(f"wget https://raw.githubusercontent.com/zkonduit/ezkl/ddbcc1d2d8c010eac2812572b31d2d5c13d5e6f0/examples/onnx/nanoGPT/gen.py -O gen_nano.py")
import gen_nano
os.system(f"mv network.onnx {WORKING_DIR}nanoGPT.onnx")
os.system(f"mv input.json {WORKING_DIR}input_nanoGPT.json")

prove_and_measure(None, None, 'nanoGPT') # You may need to manually go in and set calibrate to accuracy



