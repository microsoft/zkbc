import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl, seaborn as sns
import math, string, re, pickle, json, os, sys, datetime, itertools
from collections import Counter
from tqdm import tqdm

# Parameter scaling exploration
import matplotlib.gridspec as gridspec
# sns.set_theme()
import scienceplots
plt.style.use('science')
plt.rcParams.update({'text.usetex': True})

results = pd.read_csv('model_size_results_compiled_Jan8.csv')

results['setup_time'] = results['pk_time'] + results['vk_time']


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
plt.show()




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
sns.scatterplot(data=result
s, x='num_constraints', y='proof_size', hue='model_type')

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




def fig_parameter_scaling(fig, save=True):
    # Define the grid layout
    spec = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    ax1 = fig.add_subplot(spec[0, 0]) # First plot (top left)
    ax2 = fig.add_subplot(spec[0, 1]) # Second plot (top right)
    ax3 = fig.add_subplot(spec[1, :]) # Third plot (bottom, spanning full width)


    sns.scatterplot(data=results, x='num_constraints', y='logrows', hue='model_type', ax=ax3)
    ax3.set_xlabel(r'Number of Constraints $(n_{{con}})$')
    ax3.set_ylabel('SRS Logrows')
    # plot line log2(num_constraints) = logrows manually
    x = np.linspace(1e5, results['num_constraints'].max(), 100)
    y = np.ceil(np.log2(x))
    ax3.plot(x, y, color='black', linestyle='--', zorder=-1, label=r'$\lceil \log_2(n_{{con}}) \rceil$')
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
    if save:
        plt.savefig('figs/model_size_parameter_scaling.png', dpi=500, bbox_inches='tight')
        plt.show()

# with plt.rc_context({'text.usetex': False}):
fig = plt.figure(figsize=(3.5*2, 2.625*2))
fig_parameter_scaling(fig)


## Small versions
fig, ax = plt.subplots(1,1, figsize=(3.5, 2.625))
sns.scatterplot(data=results, x='param_count', y='macs', hue='model_type', ax=ax)
ax.set_xlabel('Model Parameter Count')
ax.set_ylabel('MACs (proxy for FLOPs)')
ax.legend(title='Model Type')
plt.tight_layout()
plt.savefig('figs/param_count_vs_macs.png', dpi=500, bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(1,1, figsize=(3.5, 2.625))
sns.scatterplot(data=results, y='num_constraints', x='macs', hue='model_type', ax=ax)
ax.set_ylabel('Number of Constraints $(n_{{con}})$')
ax.set_xlabel('MACs (proxy for FLOPs)')
ax.legend(title='Model Type')
plt.tight_layout()
plt.savefig('figs/macs_vs_constraints.png', dpi=500, bbox_inches='tight')
plt.show()





# # Key result: keygen time scales with num_constraints
# sns.scatterplot(data=results, x='num_constraints', y='time_to_setup', hue='model_type')
# plt.show()

# # Key result: proving time & setup time scale with logrows
# sns.scatterplot(data=results, x='logrows', y='time_to_prove', hue='model_type')

# plt.show()
# # Set yaxis log


with plt.rc_context({'text.usetex': False}):
    fig, [ax1,ax2] = plt.subplots(1,2, figsize=(3.5*2, 2.625), sharey=True)
    sns.scatterplot(data=results, x='num_constraints', y='time_to_prove', hue='model_type', ax=ax1)
    plt.yscale('log')
    ax1.get_legend().remove()
    ax1.set_xlabel('Number of Constraints $(n_{{con}})$')
    ax1.set_ylabel('Proving Time (s)')

    x = np.linspace(results['num_constraints'].min(), results['num_constraints'].max(), 100)
    y = x
    scale_factor = results['time_to_prove'].max() / y.max()
    scientific_str = r"${:.1e}".format(scale_factor).replace('e', r'\times 10^{')+r'}'
    ax1.plot(x, scale_factor*y, color='black', linestyle='--', zorder=-1, label=scientific_str+r'n_{{con}}$')

    sns.scatterplot(data=results, x='logrows', y='time_to_prove', hue='model_type', ax=ax2)
    x = np.linspace(results['logrows'].min(), results['logrows'].max(), 100)
    y = np.exp2(x)
    scale_factor = results['time_to_prove'].max() / y.max()
    scientific_str = r"${:.1e}".format(scale_factor).replace('e', r'\times 10^{')+r'}'
    ax2.plot(x, scale_factor*y, color='black', linestyle='--', zorder=-1, label=scientific_str+r'n_{{con}}$')
    ax2.set_xlabel('SRS Logrows')
    plt.legend(title='Model Type')
    plt.tight_layout()
    plt.savefig('figs/nconst_proving_time_scaling.png', dpi=500, bbox_inches='tight')
    plt.show()










colors = sns.color_palette()[3:6]
plt.scatter(results['logrows'], results['pk_size'], label='pk', c=colors[0])
plt.scatter(results['logrows'], results['vk_size'], label='vk', c=colors[1])
plt.scatter(results['logrows'], results['proof_size'], label='Proof File $(\pi)$', c=colors[2])
plt.yscale('log')
plt.xlabel('SRS Logrows')
plt.ylabel('File Size')
# Set custom y-ticks in gigabytes
yticks_in_bytes = [10000, 10**6, 10**9, 10**11]
yticks_lab = ['10 KB', '1 MB', '1 GB', '100 GB']
plt.yticks(yticks_in_bytes, yticks_lab)
plt.legend(['Proving Key $(pk)$', 'Verification Key $(vk)$', 'Proof $(\pi)$'], bbox_to_anchor=(1.01, 1), loc='upper left')
plt.show()



# Option 2
colors = sns.color_palette()[3:6]
def proof_size_figure(fig, save=True):
    ax = fig.add_subplot()    
    ax.scatter(results['num_constraints'], results['pk_size'], label='Proving Key $(pk)$', c=colors[0])
    ax.scatter(results['num_constraints'], results['vk_size'], label='Verification Key $(vk)$', c=colors[1])
    ax.scatter(results['num_constraints'], results['proof_size'], label='Proof $(\pi)$', c=colors[2])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r'Number of Constraints $(n_{{con}})$')
    ax.set_ylabel('File Size')

    # Set custom y-ticks in gigabytes
    yticks_in_bytes = [10000, 10**6, 10**9, 10**11]
    yticks_lab = ['10 KB', '1 MB', '1 GB', '100 GB']
    ax.set_yticks(yticks_in_bytes, yticks_lab)

    # Add three lines of best fit for each of the data and add to legend
    x = np.linspace(results['num_constraints'].min(), results['num_constraints'].max(), 100)
    scale_factor = results['pk_size'].max() / x.max()
    scientific_str = r"${:.1e}".format(scale_factor).replace('e+0','e').replace('e', r'\times 10^{')+r'}'
    ax.plot(x, scale_factor*x, color=colors[0], linestyle='--', zorder=-1, label=scientific_str+r'n_{{con}}$')

    scale_factor = results['vk_size'].max() / x.max()
    # scientific_str = r"${:.1e}".format(scale_factor).replace('e', r'\times 10^{')+r'}'
    scientific_str = r"${:0.1f}".format(scale_factor).replace('e', r'\times 10^{')+r''
    ax.plot(x, scale_factor*x, color=colors[1], linestyle='--', zorder=-1, label=scientific_str+r'n_{{con}}$')
    
    # The third linear fit need a different x-axis
    xy = results[['num_constraints','proof_size']].dropna()
    fit = np.polyfit(xy['num_constraints'], xy['proof_size'], 1)
    scientific_str = r"${:.1e}".format(fit[0]).replace('.0','').replace('e-0','e-').replace('e', r'\times 10^{')+r'}'
    scientific_str_axis = r"{:.0e}".format(fit[1]).replace('e+0','e').replace('e', r'\times 10^{')+r'}$'
    ax.plot(x, fit[0]*x+fit[1], color=colors[2], linestyle='--', zorder=-1, label=scientific_str+r'n_{{con}} + '+scientific_str_axis)

    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    # plt.tight_layout()

    if save:
        # plt.savefig('figs/nconst_file_size_scaling.png', dpi=500, bbox_inches='tight')
        plt.show()


# # with plt.rc_context({'text.usetex': False}):
# fig = plt.figure(figsize=(3.5*2, 2.625*2))
# proof_size_figure(fig)




#  Absolutely huge megaplot
fig = plt.figure(layout='constrained', figsize=(3.5*2, 2.625*2))
f_left, f_right = fig.subfigures(1, 2)
fig_parameter_scaling(f_left, save=False)
f_topright, f_bottomright = f_right.subfigures(2, 1, hspace=0)
proof_size_figure(f_bottomright, save=False)

# plt.tight_layout()
plt.show()
# 





# %% Final paper plot


fig = plt.figure(figsize=(3.5*2, 2.625*2))

# Top row plots
gs = gridspec.GridSpec(2, 6, figure=fig, wspace=1.8, hspace=0.3)
ax_a = fig.add_subplot(gs[0, :2])
ax_b = fig.add_subplot(gs[0, 2:4])

# Bottom row plots
ax_c = fig.add_subplot(gs[1, :3])
ax_d = fig.add_subplot(gs[1, 3:])

# Plot (a): Model Parameter Count vs MACs
sns.scatterplot(data=results, x='param_count', y='macs', hue='model_type', ax=ax_a)
ax_a.set_xlabel('Model Parameters')
ax_a.set_ylabel('MACs (proxy for FLOPs)')
ax_a.get_legend().remove()
offset_text = ax_a.xaxis.get_offset_text()
offset_text.set_ha('left') # Horizontally align to the right
offset_text.set_va('bottom') # Vertically align to the bottom
# offset_text.set_position((1.1, 0)) # Set the position to the far right of the axis
# plt.show()




# Plot (b): MACs vs Number of Constraints
sns.scatterplot(data=results, x='macs', y='num_constraints', hue='model_type', ax=ax_b)
ax_b.set_xlabel('MACs')
ax_b.set_ylabel('Number of Constraints')
# ax_b.get_legend().remove()
offset_text = ax_b.xaxis.get_offset_text()
offset_text.set_ha('left') # Horizontally align to the right
offset_text.set_va('bottom') # Vertically align to the bottom

handles_models, labels__models = ax_b.get_legend_handles_labels()
ax_b.get_legend().remove()
# fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.821, 0.86))

# Bottom Row Plots
# Plot (c): Number of Constraints vs Proof Time
sns.scatterplot(data=results, x='num_constraints', y='time_to_prove', hue='model_type', ax=ax_c, legend=False)
ax_c.set_xlabel('Number of Constraints $(n_{{con}})$')
ax_c.set_ylabel('Proof Time (s)')
ax_c.set_yscale('log')

x = np.linspace(results['num_constraints'].min(), results['num_constraints'].max(), 100)
y = x
scale_factor = results['time_to_prove'].max() / y.max()
scientific_str = r"${:.1e}".format(scale_factor).replace('e+0','e').replace('e', r'\times 10^{')+r'}'
ax_c.plot(x, scale_factor*y, color='black', linestyle='--', zorder=-1, label=scientific_str+r'n_{{con}}$')
ax_c.legend()






# Plot (d): File Size vs Number of Constraints
colors = sns.color_palette()[3:6]
ax_d.scatter(results['num_constraints'], results['pk_size'], label='Proving Key $(pk)$', c=colors[0],  marker='s', edgecolors='white')
ax_d.scatter(results['num_constraints'], results['vk_size'], label='Verification Key $(vk)$', c=colors[1], marker='s', edgecolors='white')
ax_d.scatter(results['num_constraints'], results['proof_size'], label='Proof File $(\pi)$', c=colors[2], marker='s', edgecolors='white')
ax_d.set_xlabel(r'Number of Constraints $(n_{{con}})$')
ax_d.set_ylabel('File Size', labelpad=-4)
ax_d.set_yscale('log')

offset_text = ax_c.xaxis.get_offset_text()
offset_text.set_position((1.05, 0))
offset_text = ax_d.xaxis.get_offset_text()
offset_text.set_position((1.05, 0))

 # Set custom y-ticks in gigabytes
yticks_in_bytes = [10000, 10**6, 10**9, 10**11]
yticks_lab = ['10 KB', '1 MB', '1 GB', '100 GB']
ax_d.set_yticks(yticks_in_bytes, yticks_lab)

 # Add three lines of best fit for each of the data and add to legend
x = np.linspace(results['num_constraints'].min(), results['num_constraints'].max(), 100)
scale_factor = results['pk_size'].max() / x.max()
scientific_str = r"${:.1e}".format(scale_factor).replace('e+0','e').replace('e', r'\times 10^{')+r'}'
ax_d.plot(x, scale_factor*x, color=colors[0], linestyle='--', zorder=-1, label=scientific_str+r'n_{{con}}$')

scale_factor = results['vk_size'].max() / x.max()
# scientific_str = r"${:.1e}".format(scale_factor).replace('e', r'\times 10^{')+r'}'
scientific_str = r"${:0.1f}".format(scale_factor).replace('e', r'\times 10^{')+r''
ax_d.plot(x, scale_factor*x, color=colors[1], linestyle='--', zorder=-1, label=scientific_str+r'n_{{con}}$')

# The third linear fit need a different x-axis
xy = results[['num_constraints','proof_size']].dropna()
fit = np.polyfit(xy['num_constraints'], xy['proof_size'], 1)
scientific_str = r"${:.1e}".format(fit[0]).replace('.0','').replace('e-0','e-').replace('e', r'\times 10^{')+r'}'
scientific_str_axis = r"{:.0e}".format(fit[1]).replace('e+0','e').replace('e', r'\times 10^{')+r'}$'
ax_d.plot(x, fit[0]*x+fit[1], color=colors[2], linestyle='--', zorder=-1, label=scientific_str+r'n_{{con}} + '+scientific_str_axis)


# Create Legend in blank space (c)
handles, labels = ax_d.get_legend_handles_labels()
fig.legend(handles_models+handles, labels__models+labels, loc='upper right', bbox_to_anchor=(0.93, 0.9))

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('figs/model_size_final.png', dpi=500, bbox_inches='tight')
plt.savefig('figs/model_size_final.pdf', bbox_inches='tight')
plt.show()
