import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from tqdm import tqdm
import ezkl, os, json


dataset = Planetoid(root='/tmp/Cora', name='Cora')

class GCN(torch.nn.Module):
    def __init__(self, edge_index):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
        self.edge_index = edge_index

    def forward(self, x):
        x = self.conv1(x, self.edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if  torch.backends.mps.is_available() else "cpu")

data = dataset[0].to(device)
model = GCN(data.edge_index).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in tqdm(range(200)):
    optimizer.zero_grad()
    x, edge_index = data.x, data.edge_index
    out = model(x)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
output = model(data.x)
pred = output.argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')


data = data.to('cpu') 
model = model.to('cpu')
x, edge_index = data.x, data.edge_index
model.edge_index = model.edge_index.to('cpu')
output = output.to('cpu')

# Get MACs and params
from thop import profile
macs, params = profile(model, inputs=(x,))
print(macs)

os.makedirs('GNN/logs', exist_ok=True)

# Export to ONNX
torch.onnx.export(model, (x, ),'GNN/GNN.onnx', export_params=True, do_constant_folding=True)

# edge_array = ((edge_index).detach().numpy().reshape([-1])).tolist()
data_array = ((x).detach().numpy().reshape(-1)).tolist() 
output_data = [((o).detach().numpy()).reshape([-1]).tolist() for o in output]
witness_data = dict(input_data = [data_array], output_data = output_data)
json.dump(witness_data, open( 'GNN/input.json', 'w' ) )

# witness_loaded = json.load(open('GNN/input.json', 'r'))
# data_loaded = np.array(witness_loaded['input_data'])
# data_loaded.shape

# Compile
os.system(f"ezkl table -M GNN/GNN.onnx")
os.system(f"ezkl gen-settings -M GNN/GNN.onnx --settings-path=GNN/settings.json")
os.system(f"ezkl calibrate-settings -M GNN/GNN.onnx -D GNN/input.json --settings-path=GNN/settings.json --target=resources")
os.system(f"ezkl compile-circuit -M GNN/GNN.onnx -S GNN/settings.json --compiled-circuit GNN/GNN.ezkl")

with open('settings.json', 'r') as f:
    settings = json.load(f)
    logrows = settings['run_args']['logrows']

os.system(f"ezkl gen-witness -M GNN/GNN.ezkl -D GNN/input.json --output GNN/witness.json")
os.system(f"ezkl mock -M GNN/GNN.ezkl --witness GNN/witness.json")

