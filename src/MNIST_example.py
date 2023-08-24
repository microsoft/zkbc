from torch import nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import os
import glob, json
from models.VariableCNN import VariableCNN
import ezkl

DEBUG = True
SRS_SMALL_PATH = '../../kzgs/kzg16.srs' # You may need to generate this
LOGGING = True
os.makedirs('MNIST/logs', exist_ok=True)
pipstd = lambda fname: f" 2>&1 | tee MNIST/logs/{fname}.log" if LOGGING else ""


# 1. Get the data
train_dataset = datasets.MNIST(root='./MNIST/data', train=True, transform=transforms.ToTensor(),  download=True) # Loads in the train data

biased_indices = [i for i, (x, y) in enumerate(train_dataset) if np.random.random() < 1*(y==8)+0.01] # Bias the data towards 8s
biased_train_dataset = torch.utils.data.Subset(train_dataset, biased_indices) # Create a biased dataset
train_loader = torch.utils.data.DataLoader(biased_train_dataset, batch_size=8, shuffle=True)
train_loader_unbiased = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataset = datasets.MNIST(root='./MNIST/data', train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)


## 2. Make the basic model

model = VariableCNN(nlayer=1, output_size=10)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if  torch.backends.mps.is_available() else "cpu")
device = "cpu"
model = model.to(device)

# Now we train it
print("Training model")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(2):
    for images, labels in tqdm(train_loader_unbiased, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)   
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()  
        optimizer.step()  



# Get the accuracy
print("Analysing model results")
model.eval()
with torch.no_grad():
    true_labels, pred_labels = [], []
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(predicted.cpu().numpy())


label_accuracy, label_counts = np.zeros(10), np.zeros(10)
for tl, pl in zip(true_labels, pred_labels):
    label_accuracy[tl] += tl==pl
    label_counts[tl] += 1

label_accuracy = [la/lc for la, lc in zip(label_accuracy, label_counts)]
print("Label accuracy:", label_accuracy)
os.system(f"echo '{label_accuracy}'"+pipstd('setup'))



## 3. Export the model and data for ezkl to use
model.to("cpu")
example_input = next(iter(test_loader))[0][0].to("cpu")
from utils.export import export
export(model, input_array=example_input, onnx_filename="MNIST/network.onnx", input_filename="MNIST/input.json")

## 3.1 Setup and calibrate the model for proving using ezkl
if LOGGING:  os.system("ezkl table -M MNIST/network.onnx" + pipstd('setup'))
os.system("ezkl gen-settings -M MNIST/network.onnx --settings-path=MNIST/settings.json --input-visibility='public'" + pipstd('setup') )
os.system("ezkl calibrate-settings -M MNIST/network.onnx -D MNIST/input.json --settings-path=MNIST/settings.json" + pipstd('setup'))
os.system("ezkl compile-model -M MNIST/network.onnx -S MNIST/settings.json --compiled-model MNIST/network.ezkl" + pipstd('setup'))
os.system("ezkl gen-witness -M MNIST/network.ezkl -D MNIST/input.json --output MNIST/witnessRandom.json --settings-path MNIST/settings.json" + pipstd('setup'))
os.system("ezkl mock -M MNIST/network.ezkl --witness MNIST/witnessRandom.json --settings-path MNIST/settings.json" + pipstd('setup')) 
os.system(f"ezkl setup -M MNIST/network.ezkl --srs-path={SRS_SMALL_PATH} --vk-path=MNIST/vk.key --pk-path=MNIST/pk.key --settings-path=MNIST/settings.json" + pipstd('setup'))

if DEBUG:
    # this will test that proving is fully working (beyond just the mock)
    os.system(f"RUST_BACKTRACE=1 ezkl prove -M MNIST/network.ezkl --srs-path={SRS_SMALL_PATH} --witness MNIST/witnessRandom.json --pk-path=MNIST/pk.key --settings-path=MNIST/settings.json --proof-path=MNIST/proof.proof --strategy='accum'"+ pipstd('setup'))
    os.system("rm MNIST/proof.proof")

# Make the data for inference
print("Saving real data for ezkl inference")
os.makedirs('MNIST/data/ezkl_inputs', exist_ok=True)
os.makedirs('MNIST/data/ezkl_witnesses', exist_ok=True)
os.makedirs('MNIST/data/ezkl_proofs', exist_ok=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
for i, (image, label) in enumerate(tqdm(test_loader, desc="Exporting data")):
    # remove batch dimension
    output = model(image)
    
    image = image.squeeze(0)
    data_array = ((image).detach().numpy()).reshape([-1]).tolist()
    data = dict(input_data = [data_array],
                output_data = [((o).detach().numpy()).reshape([-1]).tolist() for o in output])

    input_filename = f'MNIST/data/ezkl_inputs/input_{i}_lab_{label.item()}.json'
    json.dump( data, open( input_filename, 'w' ) )
    if i ==1000: break

## 3.2 Sometimes we need to recalibrate using the real data
# print("Recalibrating on real data")
# os.system(f"ezkl calibrate-settings -M MNIST/network.onnx --data {input_filename} --settings-path=MNIST/settings.json > /dev/null")
# os.system("ezkl compile-model -M MNIST/network.onnx -S MNIST/settings.json --compiled-model MNIST/network.ezkl > /dev/null")
# os.system(f"ezkl gen-witness -M MNIST/network.ezkl --data {input_filename} --output MNIST/witnessRandom.json --settings-path MNIST/settings.json > /dev/null")
# os.system(f"ezkl setup -M MNIST/network.ezkl --srs-path={SRS_SMALL_PATH} --vk-path=MNIST/vk.key --pk-path=MNIST/pk.key --settings-path=MNIST/settings.json" + pipstd('setup'))
# os.system("ezkl mock -M MNIST/network.ezkl --witness MNIST/witnessRandom.json --settings-path MNIST/settings.json" + pipstd('setup')) 

## 3.3 Loop and prove
print("Proving over real data")
for input_file in tqdm(glob.glob("MNIST/data/ezkl_inputs/*.json")):
    proof_path = f"MNIST/data/ezkl_proofs/MLP{input_file.split('input_')[-1][:-5]}.proof"
    witness_path = f"MNIST/data/ezkl_witnesses/{input_file.split('input_')[-1][:-5]}.json"
    os.system(f"ezkl gen-witness -M MNIST/network.ezkl --data {input_file} --output {witness_path} --settings-path=MNIST/settings.json" + pipstd('prove'))
    res = os.system(f"ezkl prove -M MNIST/network.ezkl --witness {witness_path} --pk-path=MNIST/pk.key --proof-path={proof_path} --srs-path={SRS_SMALL_PATH} --settings-path=MNIST/settings.json --strategy='accum'" + pipstd('prove'))
    if res!=0: print(f"Prove error on {input_file.split('input_')[-1][:-5]}, error {res}")


## 3.4 Quickly confirm one of these proofs verifies
proof_path = glob.glob("MNIST/data/ezkl_proofs/*.proof")[0]
os.system(f"ezkl verify --settings-path MNIST/settings.json --proof-path {proof_path} --vk-path MNIST/vk.key --srs-path {SRS_SMALL_PATH}")
