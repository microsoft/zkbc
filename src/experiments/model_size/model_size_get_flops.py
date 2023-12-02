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
        model = VariableMLP(nlayer+int((nlayer**2)/2))
        input_shape= [1,256]
        dummy_input = torch.randn(input_shape)
    elif modeltype == 'Attn':
        model = SimpleTransformer(int(np.sqrt(nlayer)/2)+1, d_model=64+32*nlayer)
        input_shape= [1,16,64+32*nlayer]
        dummy_input = torch.randn(input_shape)
    elif modeltype == 'LSTM':
        temp_nlayer = int(np.sqrt(nlayer)/2)
        model = VariableLSTM(nlayer=temp_nlayer, input_size=8+8*nlayer, hidden_size=8+8*temp_nlayer)
        input_shape= [3+nlayer,8+8*nlayer]
        dummy_input = torch.randn(input_shape)
    else:
        raise ValueError("modeltype must be one of CNN, MLP, Attn, LSTM")
    


for nlayer in range(1, 20):
    # for modeltype in ['CNN', 'MLP', 'Attn', 'LSTM']:
    modeltype = 'CNN'
    print(modeltype, nlayer)
    model = setup_and_prove(modeltype, nlayer)
    macs, params = profile(model, inputs=(dummy_input, ))
