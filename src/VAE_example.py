import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import ezkl
import glob

LOGGING = False
os.makedirs('CelebA/logs', exist_ok=True)
pipstd = lambda fname: f" >> CelebA/logs/{fname}.log" if LOGGING else ""
SRS_PATH = '../kzgs/kzg%d.srs'

# %% 1.1 Get CelebA dataset
os.makedirs('CelebA/data/ezkl_inputs', exist_ok=True)
os.makedirs('CelebA/data/ezkl_witnesses', exist_ok=True)
os.makedirs('CelebA/data/ezkl_proofs', exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.CelebA('CelebA/data', download=False, transform=transform)

# %% 1.2 Make a random train test split
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset)-1000, 1000])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# %% 2. Make the basic model
# device = torch.device("cuda" if torch.cuda.is_available() else "mps" if  torch.backends.mps.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Small VAE

class SmallVAE(nn.Module):
    def __init__(self):
        super(SmallVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Output: 16 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: 32 x 16 x 16
            nn.ReLU(),
            nn.Flatten()  # Output: 32 * 16 * 16
        )

        # Latent space
        self.fc_mu = nn.Linear(32 * 16 * 16, 128)
        self.fc_logvar = nn.Linear(32 * 16 * 16, 128)
        self.fc_decode = nn.Linear(128, 32 * 16 * 16)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Output: 16 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),  # Output: 3 x 64 x 64
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 32, 16, 16)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function remains the same
def vae_loss(recon_x, x, mu, logvar):
    recon_x = recon_x.clamp(0, 1)  # Clamp the reconstructed images
    x = x.clamp(0, 1)  # Clamp the input images
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# %% Train the model
model = SmallVAE()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in tqdm(range(1)):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()


# %% Test the model
        
# Show the difference between data and output as an image in jupyter
# import matplotlib.pyplot as plt 
# plt.imshow(data[0].permute(1,2,0).cpu().detach().numpy())
# plt.imshow(recon_batch[0].permute(1,2,0).cpu().detach().numpy())
        

#  Sample from latent space
# # Sample from the latent space
# with torch.no_grad():
#     z = torch.randn(64, 128).to(device)
#     sample = model.decode(z).cpu()
#     sample = sample.clamp(0, 1)
# # plt.imshow(sample[2].permute(1,2,0).cpu().detach().numpy())

    
model.eval()
test_loss = 0
with torch.no_grad():
    for i, (data, _) in enumerate(test_loader):
        data = data.to(device)
        recon_batch, mu, logvar = model(data)
        test_loss += vae_loss(recon_batch, data, mu, logvar).item()

test_loss /= len(test_loader.dataset)
print(f"Test loss: {test_loss:.4f}")

# %% 3.1 Export the model
# Def wrap the model to only take the reconstruction from decoding
class VAEDecoder(nn.Module):
    def __init__(self, model):
        super(VAEDecoder, self).__init__()
        self.model = model

    def forward(self, z):
        return self.model.decode(z)
    
model.to("cpu")
vae_model = VAEDecoder(model)

from utils.export import export
example_input = next(iter(test_loader))[0]
example_mu, _ = model.encode(example_input) 
vae_model(example_mu)
export(vae_model, input_array=example_mu, onnx_filename='CelebA/vae.onnx', input_filename='CelebA/celeba_temp.json', reshape_input=False)

# %% Setup and prove
os.system("ezkl gen-settings -M CelebA/vae.onnx --settings-path=CelebA/settings.json --input-visibility='public'" + pipstd('setup'))
os.system("ezkl calibrate-settings -M CelebA/vae.onnx -D CelebA/celeba_temp.json --settings-path=CelebA/settings.json" + pipstd('setup'))
settings = json.load(open('CelebA/settings.json', 'r'))
logrows = settings['run_args']['logrows']
ezkl.get_srs(srs_path=SRS_PATH % logrows, logrows=logrows)

os.system("ezkl compile-circuit -M CelebA/vae.onnx -S CelebA/settings.json --compiled-circuit CelebA/vae.ezkl" + pipstd('setup'))
os.system("ezkl gen-witness -M CelebA/vae.ezkl -D CelebA/celeba_temp.json --output CelebA/witnessRandom.json" + pipstd('setup'))
os.system(f"ezkl setup -M CelebA/vae.ezkl --srs-path={SRS_PATH % logrows} --vk-path=CelebA/vae_vk.key --pk-path=CelebA/vae_pk.key" + pipstd('setup'))
os.system(f"ezkl prove -M CelebA/vae.ezkl --srs-path={SRS_PATH % logrows} --witness CelebA/witnessRandom.json --pk-path=CelebA/vae_pk.key --proof-path=CelebA/vae.proof" + pipstd('setup'))
os.system(f"ezkl verify --settings-path=CelebA/settings.json  --vk-path=CelebA/vae_vk.key --proof-path=CelebA/vae.proof --srs-path={SRS_PATH % logrows}" + pipstd('setup'))

