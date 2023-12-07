from torch import nn
import torch
from tqdm import tqdm
import numpy as np
import os, sys
import time
import pickle, json
from thop import profile
import timeit
import ezkl

# Import the existing models from src/models
sys.path.append('../..')
from utils.export import export
from models.VariableCNN import VariableCNN
from models.VariableMLP import VariableMLP
from models.VariableLSTM import VariableLSTM
from models.SimpleTransformer import SimpleTransformer

FILENAME ='model_size_results_Dec3rd.csv'
LOGROWS_PATH = '../../../kzgs/kzg%d.srs' # You may need to generate this
LOGGING = True
pipstd = lambda fname: f" >> logs/{fname}.log" if LOGGING else ""
os.makedirs('logs', exist_ok=True)
os.makedirs('runfiles', exist_ok=True)

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
        # Gives error: 
        # thread 'main' panicked at src/graph/model.rs:1354:25:
        # assertion `left == right` failed
        # left: 2
        # right: 0
        temp_nlayer = int(np.sqrt(nlayer)/2)
        temp_extra = (nlayer if nlayer < 16 else nlayer - 4)
        model = VariableLSTM(nlayer=temp_nlayer, input_size=8+8*temp_extra, hidden_size=8+8*temp_nlayer)
        input_shape= [3+nlayer,8+8*temp_extra]
        dummy_input = torch.randn(input_shape)
    else:
        raise ValueError("modeltype must be one of CNN, MLP, Attn, LSTM")
    
    macs, params = profile(model, inputs=(dummy_input, ))
    export(model, input_shape= input_shape, onnx_filename=f'runfiles/{modeltype+str(nlayer)}.onnx', input_filename=f'runfiles/input{modeltype+str(nlayer)}.json')

    logs_file = pipstd(f'{modeltype}_prework_{nlayer}')

    res1 = os.system(f"ezkl table -M runfiles/{modeltype+str(nlayer)}.onnx" + logs_file)
    res2 = os.system(f"ezkl gen-settings -M runfiles/{modeltype+str(nlayer)}.onnx --settings-path=settings.json"+logs_file)
    res3 = os.system(f"ezkl calibrate-settings -M runfiles/{modeltype+str(nlayer)}.onnx -D runfiles/input{modeltype+str(nlayer)}.json --settings-path=settings.json --target=resources"+logs_file)
    res4 = os.system(f"ezkl compile-circuit -M runfiles/{modeltype+str(nlayer)}.onnx -S settings.json --compiled-circuit runfiles/{modeltype+str(nlayer)}.ezkl"+logs_file)
    res5 = os.system(f"ezkl gen-witness -M runfiles/{modeltype+str(nlayer)}.ezkl -D runfiles/input{modeltype+str(nlayer)}.json --output runfiles/witness_{modeltype+str(nlayer)}.json"+logs_file)
    os.system(f"ezkl mock -M runfiles/{modeltype+str(nlayer)}.ezkl --witness runfiles/witness_{modeltype+str(nlayer)}.json" + logs_file) 
    os.system(f"cat settings.json >> logs/{modeltype}_prework_{nlayer}.log")

    if res2 != 0 or res3 != 0 or res4 != 0 or res5 != 0:
        print("Prework failed. Check logs.")

    # Read in settings to determin SRS file size
    with open('settings.json', 'r') as f:
        logrows = json.load(f)['run_args']['logrows']
    ezkl.get_srs(LOGROWS_PATH % logrows, 'settings.json')

    time_to_setup = timeit.timeit(lambda: os.system(f"ezkl setup -M runfiles/{modeltype+str(nlayer)}.ezkl --srs-path={LOGROWS_PATH % logrows} --vk-path=vk.key --pk-path=pk.key" + pipstd(f'{modeltype}_setup_{nlayer}')),  number=1)

    os.makedirs('proofs', exist_ok=True)
    proof_file = f"proofs/{modeltype}_proof_{nlayer}.proof"
    time_to_prove = timeit.timeit(lambda: os.system(f"ezkl prove -M runfiles/{modeltype+str(nlayer)}.ezkl --srs-path={LOGROWS_PATH % logrows} --witness runfiles/witness_{modeltype+str(nlayer)}.json --pk-path=pk.key --proof-path={proof_file}"+ pipstd(f'{modeltype}_prove_{nlayer}')), number=1)

    proof_size = os.path.getsize(proof_file)
    vk_size = os.path.getsize('vk.key')
    pk_size = os.path.getsize('pk.key')

    print(f"Model type: {modeltype}, nlayer: {nlayer}, param_count: {params}, ops_count:{macs}, time_to_setup: {time_to_setup}, time_to_prove: {time_to_prove}, proof_size: {proof_size}, vk_size: {vk_size}, pk_size: {pk_size}")

    return modeltype, nlayer, params, macs, time_to_setup, time_to_prove, proof_size, vk_size, pk_size

# Setup csv file
with open(FILENAME, 'w') as f:
    f.write("modeltype,nlayer,param_count,macs,time_to_setup,time_to_prove,proof_size,vk_size,pk_size\n")

results = [] # For local runtime experiments
ranges = list(range(1, 25)) 
for nlayer in tqdm(ranges):
    for modeltype in ['Attn', 'CNN', 'MLP']: #  'LSTM' 
        print("-------------------- Running", modeltype, nlayer, "------------------------")
        # try:
        result = setup_and_prove(modeltype, nlayer)
        results.append(result)
        # Write result as csv to file
        with open(FILENAME, 'a') as f:
            f.write(",".join([str(x) for x in result]) + "\n")
        # except Exception as e:
        #     print("Failed with", e)

print("Done with local experiments")
print(results)

