import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl, seaborn as sns
import math, string, re, pickle, json, os, sys, datetime, itertools
from collections import Counter
from tqdm import tqdm


results = pd.read_csv('model_size_results_compiled_8Aug.csv')

# Exploratory data analysis
sns.pairplot(results, hue='model_type', vars=['macs', 'num_constraints', 'param_count'])

# Key result: params \neq macs
sns.scatterplot(data=results, x='param_count', y='macs', hue='model_type')
plt.ylabel('MACs (proxy for FLOPs)')
plt.xlabel('Model Parameter Count')
plt.legend(title='Model Type')
plt.show()

# Key result: num_constraints <= macs (based on operation redundancy)
sns.scatterplot(data=results, x='macs', y='num_constraints', hue='model_type')



# sns.pairplot(results, hue='model_type', vars=['scale', 'bits', 'logrows', 'num_constraints', 'mock_time', 'vk_time', 'pk_time', 'wall_setup_time', 'proof_time', 'wall_prove_time'])




# Key result: keygen time scales with num_constraints
sns.scatterplot(data=results, x='num_constraints', y='pk_time', hue='model_type')
sns.scatterplot(data=results, x='num_constraints', y='vk_time', hue='model_type')

# Key result: proving time & setup time scale with logrows
sns.scatterplot(data=results, x='num_constraints', y='time_to_setup', hue='model_type')
sns.scatterplot(data=results, x='num_constraints', y='time_to_prove', hue='model_type')
sns.scatterplot(data=results, x='logrows', y='time_to_setup', hue='model_type')
sns.scatterplot(data=results, x='logrows', y='time_to_prove', hue='model_type')



# Key result: proof size is nonsense??
sns.scatterplot(data=results, x='num_constraints', y='proof_size', hue='model_type')

# Key result: vk size scales with num_constraints
sns.scatterplot(data=results, x='num_constraints', y='vk_size', hue='model_type')

# Key result: pk size scales with num_constraints and is HUGE
sns.scatterplot(data=results, x='num_constraints', y='pk_size', hue='model_type')



# sns.scatterplot(data=results, x='num_constraints', y='pk_size', hue='model_type')
# sns.scatterplot(data=results, x='logrows', y='pk_size', hue='model_type')

# Obvious result: logrows = log2(num_constraints)
sns.scatterplot(data=results, x='num_constraints', y='logrows', hue='model_type')
# plot line log2(num_constraints) = logrows manually
x = np.linspace(0, 1e7, 100)
y = np.ceil(np.log2(x))
plt.plot(x, y, color='black', linestyle='--')




# Production plots

# Parameter scaling exploration
import matplotlib.gridspec as gridspec
# sns.set_theme()
import scienceplots
plt.style.use('science')


def fig_parameter_scaling():
    # fig, axes = plt.subplots(1,3, figsize=(8,5))
    fig = plt.figure(figsize=(8, 5))

    # Define the grid layout
    spec = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    ax1 = fig.add_subplot(spec[0, 0]) # First plot (top left)
    ax2 = fig.add_subplot(spec[0, 1]) # Second plot (top right)
    ax3 = fig.add_subplot(spec[1, :]) # Third plot (bottom, spanning full width)


    sns.scatterplot(data=results, x='num_constraints', y='logrows', hue='model_type', ax=ax3)
    ax3.set_xlabel('Number of Constraints (ncon)')
    ax3.set_ylabel('SRS Logrows')
    # plot line log2(num_constraints) = logrows manually
    x = np.linspace(1e5, results['num_constraints'].max(), 100)
    y = np.ceil(np.log2(x))
    ax3.plot(x, y, color='black', linestyle='--', zorder=-1, label='\ceil{log_2(ncon)}')
    ax3.legend(title='Model Type')



    # Make two subplots link yaxis for the two plots
    ax1.get_shared_y_axes().join(ax1, ax2)
    # Hide tick labels on the second plot
    ax2.set(yticks=[])

    sns.scatterplot(data=results, x='param_count', y='macs', hue='model_type', ax=ax1)
    ax1.set_xlabel('Model Parameter Count')
    ax1.set_ylabel('MACs (proxy for FLOPs)')
    ax1.get_legend().remove()


    sns.scatterplot(data=results, x='num_constraints', y='macs', hue='model_type', ax=ax2)
    ax2.set_xlabel('Number of Constraints')
    # ax2.set_ylabel('MACs (proxy for FLOPs)')
    ax2.get_legend().remove()
    plt.tight_layout()

fig_parameter_scaling()

