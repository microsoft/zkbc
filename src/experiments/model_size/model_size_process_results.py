import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl, seaborn as sns
import math, string, re, pickle, json, os, sys, datetime, itertools
from collections import Counter
from tqdm import tqdm
import glob, re

# Read in and parse log files and text
results = []
for model_type in ['CNN', 'MLP', 'Attn']:
    files = glob.glob(f'logs/{model_type}*.log')
    file_dict = {}
    for file in files:
        file_number = int(file.split('.')[0].split('_')[-1])
        file_type = file.split('.')[0].split('_')[-2]
        if file_number not in file_dict:
            file_dict[file_number] = {'prework': None, 'setup': None, 'prove': None}
        file_dict[file_number][file_type] = file

    for number, file_options in file_dict.items():
        file = file_options['prework']
        if file is None:
            print(f"Skipping prework {model_type} {number}")
            scale, bits, logrows, num_constraints = np.nan, np.nan, np.nan, np.nan
        else:
            with open(file) as f:
                lines = f.readlines()
            # find the line with 'succeeded' in it and get the number at the start of that line
            for line in lines:
                if '- succeeded' in line:
                    mock_time = float(re.search(r'\d+', line)[0])
            for line in lines:
                if '{"run_args":' in line:
                    settings = json.loads(line.split('[*]')[0])
                    scale, bits, logrows, num_constraints = settings['run_args']['input_scale'], settings['run_args']['param_scale'], settings['run_args']['logrows'], settings['num_rows']
            
        file = file_options['setup']
        if file is None:
            print(f"Skipping setup {model_type} {number}")
            vk_time, pk_time, wall_setup_time = np.nan, np.nan, np.nan
        else:
            with open(file) as f:
                lines = f.readlines()
            
            for line in lines:
                if 'VK took' in line:
                    vk_time = float(re.search(r'VK took (\d+\.\d+)', line)[1])
                if 'PK took' in line:
                    pk_time = float(re.search(r'PK took (\d+\.\d+)', line)[1])
                if 'succeeded' in line:
                    wall_setup_time = float(re.search(r'\d+', line)[0])

        file = file_options['prove']
        if file is None:
            print(f"Skipping prove {model_type} {number}")
            proof_time, wall_prove_time = np.nan, np.nan
        else:
            with open(file) as f:
                lines = f.readlines()

            for line in lines:
                if 'proof took' in line:
                    proof_time = float(re.search(r'proof took (\d+\.\d+)', line)[1])
                if 'succeeded' in line:
                    wall_prove_time = float(re.search(r'\d+', line)[0])

        results.append([model_type, number, scale, bits, logrows, num_constraints, mock_time, vk_time, pk_time, wall_setup_time, proof_time, wall_prove_time])
    
results_df = pd.DataFrame(results, columns=['model_type', 'nlayer', 'scale', 'numrows', 'logrows', 'num_constraints', 'mock_time', 'vk_time', 'pk_time', 'wall_setup_time', 'proof_time', 'wall_prove_time'])


results_python = pd.read_csv('model_size_results_Jan8.csv')
results_df = results_df.join(results_python.set_index(['modeltype', 'nlayer']), on=['model_type', 'nlayer'])

results_df.to_csv('model_size_results_compiled_Jan8.csv', index=False)