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

DEBUG = True
AVAILABLE_LOGROWS = [16,17,18, 19, 21, 25]
LOGROWS_PATH = lambda lr: f'../../../../kzgs/kzg{lr}.srs' # You may need to generate this
NEAREST_LOGROWS_ABOVE = lambda lr: LOGROWS_PATH(min([x for x in AVAILABLE_LOGROWS if x >= lr]))
LOGGING = True
pipstd = lambda fname: f" 2>&1 | tee logs/{fname}.log" if LOGGING else ""
os.makedirs('logs', exist_ok=True)
os.makedirs('runfiles', exist_ok=True)

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
    
    macs, params = profile(model, inputs=(dummy_input, ))
    export(model, input_shape= input_shape, onnx_filename=f'runfiles/{modeltype+str(nlayer)}.onnx', input_filename=f'runfiles/input{modeltype+str(nlayer)}.json')

    logs_file = pipstd(f'{modeltype}_prework_{nlayer}')
    os.system(f"ezkl table -M runfiles/{modeltype+str(nlayer)}.onnx" + logs_file)
    os.system(f"ezkl gen-settings -M runfiles/{modeltype+str(nlayer)}.onnx --settings-path=settings.json --input-visibility='public'"+logs_file )
    os.system(f"ezkl calibrate-settings -M runfiles/{modeltype+str(nlayer)}.onnx -D runfiles/input{modeltype+str(nlayer)}.json --settings-path=settings.json --target=resources"+logs_file)
    os.system(f"ezkl compile-model -M runfiles/{modeltype+str(nlayer)}.onnx -S settings.json --compiled-model runfiles/{modeltype+str(nlayer)}.ezkl"+logs_file)
    os.system(f"ezkl gen-witness -M runfiles/{modeltype+str(nlayer)}.ezkl -D runfiles/input{modeltype+str(nlayer)}.json --output runfiles/witness_{modeltype+str(nlayer)}.json --settings-path settings.json"+logs_file)
    os.system(f"ezkl mock -M runfiles/{modeltype+str(nlayer)}.ezkl --witness runfiles/witness_{modeltype+str(nlayer)}.json --settings-path settings.json" + logs_file) 
    os.system(f"cat settings.json >> logs/{modeltype}_prework_{nlayer}.log")

    # Read in settings to determin SRS file size
    with open('settings.json', 'r') as f:
        settings = json.load(f)
        lr = settings['run_args']['logrows']
        srs_path  = NEAREST_LOGROWS_ABOVE(lr)

    time_to_setup = timeit.timeit(lambda: os.system(f"ezkl setup -M runfiles/{modeltype+str(nlayer)}.ezkl --srs-path={srs_path} --vk-path=vk.key --pk-path=pk.key --settings-path=settings.json" + pipstd(f'{modeltype}_setup_{nlayer}')),  number=1)

    os.makedirs('proofs', exist_ok=True)
    proof_file = f"proofs/{modeltype}_proof_{nlayer}.proof"
    time_to_prove = timeit.timeit(lambda: os.system(f"ezkl prove -M runfiles/{modeltype+str(nlayer)}.ezkl --srs-path={srs_path} --witness runfiles/witness_{modeltype+str(nlayer)}.json --pk-path=pk.key --settings-path=settings.json --proof-path={proof_file} --strategy='accum'"+ pipstd(f'{modeltype}_prove_{nlayer}')), number=1)

    proof_size = os.path.getsize(proof_file)
    vk_size = os.path.getsize('vk.key')
    pk_size = os.path.getsize('pk.key')

    print(f"Model type: {modeltype}, nlayer: {nlayer}, param_count: {params}, ops_count:{macs}, time_to_setup: {time_to_setup}, time_to_prove: {time_to_prove}, proof_size: {proof_size}, vk_size: {vk_size}, pk_size: {pk_size}")

    return modeltype, nlayer, params, macs, time_to_setup, time_to_prove, proof_size, vk_size, pk_size

# Setup csv file
FILENAME ='model_size_results_Sept5th.csv'
with open(FILENAME, 'w') as f:
    f.write("modeltype,nlayer,param_count,macs,time_to_setup,time_to_prove,proof_size,vk_size,pk_size\n")

results = [] # For local runtime experiments
ranges = list(range(10, 30)) 
# + list(range(20, 30, 2)) + list(range(30, 41, 5))
for nlayer in tqdm(ranges):
    for modeltype in ['CNN', 'MLP']: #  'Attn',  'LSTM'
        print("Running", modeltype, nlayer)
        try:
            result = setup_and_prove(modeltype, nlayer)
            results.append(result)
            # Write result as csv to file
            with open(FILENAME, 'a') as f:
                f.write(",".join([str(x) for x in result]) + "\n")
        except Exception as e:
            print("Failed with", e)

print("Done with local experiments")
print(results)



# # This is just code to get the sizing right
# def sizes(modeltype, nlayer):
#     if modeltype == 'CNN':
#         model = VariableCNN(nlayer)
#         input_shape= [1,28,28]
#         dummy_input = torch.randn(input_shape)
#     elif modeltype == 'MLP':
#         model = MLP(nlayer+int((nlayer**2)/2))
#         input_shape= [1,256]
#         dummy_input = torch.randn(input_shape)
#     elif modeltype == 'Attn':
#         model = SimpleTransformer(int(np.sqrt(nlayer)/2)+1, d_model=64+32*nlayer)
#         input_shape= [1,16,64+32*nlayer]
#         dummy_input = torch.randn(input_shape)
#     elif modeltype == 'LSTM':
#         temp_nlayer = int(np.sqrt(nlayer)/2)
#         model = VariableLSTM(nlayer=temp_nlayer, input_size=8+8*nlayer, hidden_size=8+8*temp_nlayer)
#         input_shape= [3+nlayer,8+8*nlayer]
#         dummy_input = torch.randn(input_shape)
#     else:
#         raise ValueError("modeltype must be one of CNN, MLP, Attn, LSTM")
    
#     macs, params = profile(model, inputs=(dummy_input, ))
#     return macs, params

# size_results = []
# ranges = list(range(0, 20)) + list(range(20, 30, 2)) + list(range(30, 41, 5))
# for nlayer in tqdm(ranges):
#     for modeltype in ['CNN', 'MLP', 'Attn', 'LSTM']:
#         print("Running", modeltype, nlayer)
#         try:
#             macs, params = sizes(modeltype, nlayer)
#             size_results.append([macs, params, nlayer, modeltype])
#         except Exception as e:
#             print("Failed with", e)


# import seaborn as sns
# import pandas as pd
# size_results_df = pd.DataFrame(size_results, columns=['macs', 'params', 'nlayer', 'modeltype'])
# sns.scatterplot(size_results_df, x='nlayer', y='macs', hue='modeltype')

# size_results_df.sort_values(by='nlayer', ascending=False)