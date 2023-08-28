# This file contains some toolkits for measures the differences (errors) between the data passed into a proof and the data that comes out of a proof. It also contains some tools for visualizing the errors across standard statistical checks.

# The bottom of the file includes real results analysis.

import json, os, numpy as np
from glob import glob # the best package for finding files
import ezkl
import matplotlib.pyplot as plt
from tqdm import tqdm

def proof_file_to_io(prooffilename: str, scale: int) -> (np.array, np.array):
    """
    This will take a proof file and return the input and output values as numpy arrays.
    """

    with open(prooffilename, 'r') as f:
        proof = json.load(f)

    # Process the proof instances
    proof_instances = proof['instances']
    proof_input = proof_instances[0]
    proof_output = proof_instances[1]

    # Convert field elements to ints
    proof_input = np.array([ezkl.vecu64_to_float(byte_array, scale) for byte_array in proof_input]).flatten()
    proof_output = np.array([ezkl.vecu64_to_float(byte_array, scale) for byte_array in proof_output]).flatten()

    return proof_input, proof_output

def input_file_to_io(inputfilename: str) -> (np.array, np.array):
    """This will take an input.json file and return the input and output values."""
    with open(inputfilename, 'r') as f:
        input_file = json.load(f)
    return np.array(input_file['input_data']).flatten(), np.array(input_file['output_data']).flatten() if 'output_data' in input_file else None


def calculate_differences(real_values: np.array, comparison_values: np.array, sigma: int = 4):
    """
    This will look at the differences between the real values and the output values based on MSE (mean squared error). It will return overall performance and can flag any points that are 4 standard deviations away from the mean std of error.

    :param real_values: the real values
    :param comparison_values: the output values
    :param sigma: the number of standard deviations to use as a cutoff

    :return: (data_mse, high_sigma_points, std_of_fsp)
    """
    se = np.square(real_values - comparison_values)
    data_mse = np.mean(se)
    high_sigma_points = se > sigma*data_mse
    std_of_fsp = se[high_sigma_points] / data_mse

    return data_mse, len(high_sigma_points), se, high_sigma_points, std_of_fsp


def accuracy_analysis(real_values: np.array, comparison_values: np.array, plot_statistical_checks:bool = True, sigma: int = 4):
    """
    This will look at the differences between the real values and the output values based on MSE (mean squared error). It will return overall performance and can will show analysis tools to examine errors that deviate significantly. 

    :param real_values: the original values
    :param comparison_values: the proof values
    :plot_statistical_checks: whether to plot the statistical checks (turn off for fast calculation of data_mse)
    :param sigma: the number of standard deviations to use as a cutoff

    :return: (data_mse, max_error, max_error_as_a_percent_of_range)

    """

    data_mse, high_sigma_count, se, high_sigma_points, std_of_fsp = calculate_differences(real_values, comparison_values, sigma=sigma)

    max_error = np.sqrt(np.max(se))

    max_error_as_a_percent_of_range = max_error / (np.max(real_values) - np.min(real_values))

    data_mse_as_a_percent_of_range = data_mse / (np.max(real_values) - np.min(real_values))

    median_value_of_high_sigma = np.median(real_values[high_sigma_points])

    print(f"Data MSE: {data_mse}")
    print(f"Number of points with error > {sigma} standard deviations: {high_sigma_count}")
    print(f"Max error as a percent of range: {max_error_as_a_percent_of_range*100}%")
    print("Run with plot_statistical_checks = True for detailed analysis")

    if plot_statistical_checks:
        plt.rcParams['axes.formatter.useoffset'] = False
        plt.rcParams['axes.formatter.limits'] = (-3, 3)

        fig, axes = plt.subplots(2,2)

        axes[0][0].scatter(real_values, comparison_values, s=3)
        axes[0][0].scatter(real_values[high_sigma_points], comparison_values[high_sigma_points], s=3, c='r', label=f'{sigma}$\sigma$ points')
        plt.legend()
        axes[0][0].set_xlabel('Original Values')
        axes[0][0].set_ylabel('Proof Values')
        axes[0][0].set_title('Actual Value Comparison')
        axes[0][0].legend()

        axes[1][0].scatter(real_values, se, s=4)
        axes[1][0].scatter(real_values[high_sigma_points], se[high_sigma_points], s=4, c = 'r')
        axes[1][0].set_xlabel('Original Values')
        axes[1][0].set_ylabel('Squared Error (se)')
        axes[1][0].set_title('Error Homoscedasticity')

        counts, bins, _ = axes[0][1].hist(se, bins=20, label='All points')
        axes[0][1].hist(se[high_sigma_points],color='r', bins=bins, label=f'{sigma}$\sigma$ points')
        # Plot sigma level of variance and mean
        axes[0][1].axvline(data_mse, color='k', linestyle='dashed', linewidth=1, label="MSE")
        # axes[0][1].axvline(data_mse*sigma, color='k', linestyle='dashed', linewidth=1)
        axes[0][1].set_xlabel('Squared Error (se)')
        axes[0][1].set_ylabel('Count')
        axes[0][1].set_title('Histogram of Squared Errors')
        axes[0][1].set_yscale('log')
        axes[0][1].legend()

        # The final subplot will just be a table of the values
        axes[1][1].axis('off')
        tbl = axes[1][1].table(cellText=[[f"{data_mse:.2e}"], 
                                         [f"{max_error:.2e}"],
                                         [f"{high_sigma_count}"], 
                                        [f"{max_error_as_a_percent_of_range*100:.2f}%"]],
                            rowLabels=['MSE', 'Max error',
                                       f'{sigma}$\sigma$ Count', 'Max error\n% of range'],
                            loc='center')

        # Adjust the bounding box. This example moves the table slightly to give space for labels. 
        # You may need to adjust these numbers to get the desired look.
        
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.auto_set_column_width(col=list(range(len(['Value']))))
        tbl.scale(1.2, 2)
        tbl.auto_set_font_size(False)
        tbl.auto_set_column_width([0])
        axes[1][1].set_title('Summary Statistics')

        plt.tight_layout()
        plt.show()

    return data_mse, max_error, max_error_as_a_percent_of_range, data_mse_as_a_percent_of_range


def accuracy_results(prooffilename: str, inputfilename: str, settingsfilename:str = None, scale: int | None = None, plot_statistical_checks:bool = True, sigma: int = 4):

    if settingsfilename is not None:
        with open(settingsfilename, 'r') as f:
            settings = json.load(f)
        scale = settings['run_args']['scale']
    elif scale is None:
        raise ValueError("Must provide either settingsfilename or scale")
    

    proof_input, proof_output = proof_file_to_io(prooffilename, scale)
    input_input, input_output = input_file_to_io(inputfilename)

    accuracy_analysis(input_input, proof_input, plot_statistical_checks=plot_statistical_checks, sigma=sigma)
    output_accuracy = accuracy_analysis(input_output, proof_output, plot_statistical_checks=plot_statistical_checks, sigma=sigma)
    return output_accuracy


# Example with CNN with random inputs
prooffilename = 'example_files/proof.proof'
inputfilename = 'example_files/input.json'
settingsfilename = 'example_files/settings.json'

accuracy_results(prooffilename, inputfilename, settingsfilename)


# Example on pretrained GPT2



# Example with MNIST full dataset and simple CNN
# Make sure you first run the MNIST_example.py script to generate the data and proofs.
proof_files = glob("../../MNIST/data/ezkl_proofs/*.proof")

all_proof_inputs, all_proof_output, all_input_inputs, all_input_output = [], [], [], []
for prooffilename in tqdm(proof_files):
    proof_input, proof_output = proof_file_to_io(prooffilename, 4)
    inputfilename = prooffilename.replace(".proof", ".json").replace("ezkl_proofs", "ezkl_inputs").replace("/MLP", "/input_")
    input_input, input_output = input_file_to_io(inputfilename)

    all_proof_inputs.append(proof_input)
    all_proof_output.append(proof_output)
    all_input_inputs.append(input_input)
    all_input_output.append(input_output)

all_proof_inputs = np.concatenate(all_proof_inputs).flatten()
all_proof_output = np.concatenate(all_proof_output).flatten()
all_input_inputs = np.concatenate(all_input_inputs).flatten()
all_input_output = np.concatenate(all_input_output).flatten()

# Example from a singal proof
accuracy_analysis(input_input, proof_input)
accuracy_analysis(proof_output, input_output)

accuracy_analysis(all_input_inputs, all_proof_inputs)
accuracy_analysis(all_input_output, all_proof_output)