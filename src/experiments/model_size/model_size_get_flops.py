from torch import nn
import torch
from tqdm import tqdm
import numpy as np
import os, sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle, json
from thop import profile
import timeit

sys.path.append('../..')
from utils.export import export
from models.VariableCNN import VariableCNN
from models.VariableMLP import VariableMLP
from models.VariableLSTM import VariableLSTM
from models.SimpleTransformer import SimpleTransformer


def setup_and_prove(modeltype, nlayer):
    if modeltype == 'CNN':
        model = VariableCNN(nlayer)
        input_shape= [1,28,28]
        dummy_input = torch.randn(input_shape)
    elif modeltype == 'MLP':
        model = VariableMLP(3*nlayer, hidden_size=256+32*nlayer)
        input_shape= [1,256]
        dummy_input = torch.randn(input_shape)
    elif modeltype == 'Attn':
        model = SimpleTransformer(int(np.sqrt(nlayer)/2)+1, d_model= (32*nlayer if nlayer < 16 else 32*(nlayer-4)))
        input_shape= [1,16,(32*nlayer if nlayer < 16 else 32*(nlayer-4))]
        dummy_input = torch.randn(input_shape)
    elif modeltype == 'LSTM':
        temp_nlayer = int(np.sqrt(nlayer)/2)
        temp_extra = (nlayer if nlayer < 16 else nlayer - 4)
        model = VariableLSTM(nlayer=temp_nlayer, input_size=8+8*temp_extra, hidden_size=8+8*temp_nlayer)
        input_shape= [3+nlayer,8+8*temp_extra]
        dummy_input = torch.randn(input_shape)
    else:
        raise ValueError("modeltype must be one of CNN, MLP, Attn, LSTM")
    return model, dummy_input
    
modeltypes = ['Attn', 'CNN', 'LSTM', 'MLP'] 
# modeltypes = ['CNN', 'MLP', 'Attn', 'LSTM']
macs_array = {modeltype:[] for modeltype in modeltypes}
params_array = {modeltype:[] for modeltype in modeltypes}

for nlayer in range(1, 21):
    for modeltype in modeltypes:
        print(modeltype, nlayer)
        model, dummy_input = setup_and_prove(modeltype, nlayer)
        macs, params = profile(model, inputs=(dummy_input, ))
        print(np.log10(macs))
        params_array[modeltype].append(params)
        macs_array[modeltype].append(macs)


# plot the results on the same plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for modeltype in modeltypes:
    ax.plot(list(range(len(macs_array[modeltype]))), macs_array[modeltype], label=modeltype)
ax.legend()
ax.set_xlabel('Number of layers')
ax.set_ylabel('log10(MACs)')

# set log
ax.set_yscale('log')
plt.show()



# [np.log2(macs_array[modeltype][15]) for modeltype in modeltypes]
