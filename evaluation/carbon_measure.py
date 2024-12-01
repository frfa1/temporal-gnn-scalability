import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from carbontracker import parser

save_path = "../figs/"

MODELS = [
    "A3TGCNModel", "ChebGRU", "GCNConv", "GCNConv_Edge", "SimpleGRU"
]

def get_carbon(folder):

    all_carbon_dict = {}
    for subset in range(1,4):
        all_carbon_dict[subset] = {}
        subset_dir = folder + "subset_" + str(subset) + "/"
        #group_labels = ["gpu", "cpu"]
        #all_gpu_carbon_lists = []
        #all_cpu_carbon_lists = []
        for model in MODELS:
            try:
                #model_gpu_carbon_list = []
                #model_cpu_carbon_list = []
                model_carbon_list = []
                logs = parser.parse_all_logs(log_dir=subset_dir + model)
                #total_energy, total_co2eq, total_equivalents = parser.aggregate_consumption(log_dir=folder + model)
                #print(total_energy, total_co2eq, total_equivalents)
                for log in logs:
                    model_carbon_list.append(log["actual"]["co2eq (g)"])
                    #model_gpu_carbon_list.append(np.sum(log["components"]["gpu"]["avg_power_usages (W)"]))
                    #model_cpu_carbon_list.append(np.sum(log["components"]["cpu"]["avg_power_usages (W)"]))
                all_carbon_dict[subset][model] = model_carbon_list
            except:
                print("Skipping:", model, subset)
                continue
        #all_gpu_carbon_lists.append(model_gpu_carbon_list)
        #all_cpu_carbon_lists.append(model_cpu_carbon_list)

    return all_carbon_dict

def viz_carbon(data, saveas="carbon_bars"):
    # Mapping of original model names to new names
    model_name_mapping = {
        'SimpleGRU': 'Turbine-level GRU',
        'GCNConv': 'GCNConv-GRU',
        'ChebGRU': 'ChebConv-GRU',
        'GCNConv_Edge': 'EdgeConv-GCNConv-GRU',
        'A3TGCNModel': 'A3TGCN'
    }
    # Extract subsets and models
    subsets = sorted(data.keys())
    models_set = set()
    for subset_data in data.values():
        models_set.update(subset_data.keys())
    models = sorted(models_set)

    # Initialize a dictionary to hold the summed values per model per subset
    values = {model: {} for model in models}

    # Compute the sum of values for each model in each subset
    for model in models:
        for subset in subsets:
            if model in data.get(subset, {}):
                summed_value = sum(data[subset][model])
                values[model][subset] = summed_value
            else:
                values[model][subset] = None  # Indicates missing data

    # Map model names to the new names for labeling
    mapped_models = [model_name_mapping.get(model, model) for model in models]

    # Prepare data for plotting
    N = len(models)
    M = len(subsets)
    ind = np.arange(N)  # The x locations for the models
    width = 0.8 / M     # The width of each bar

    # Set up the plot style
    sns.set_style('whitegrid')
    palette = sns.color_palette('Set2', M)  # Use a color palette with M colors

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot each subset's data
    for i, subset in enumerate(subsets):
        subset_values = []
        for model in models:
            value = values[model][subset]
            if value is not None:
                subset_values.append(value)
            else:
                subset_values.append(0)  # Plot zero-height bar for missing data
        positions = ind + i * width  # Adjust positions for each subset
        ax.bar(positions, subset_values, width, label=subset, color=palette[i])

    # Customize the plot
    ax.set_xlabel('Models', fontsize=16)
    ax.set_ylabel('Summed co2eq (g)', fontsize=16)
    #ax.set_title('Summed co2eq (g)', fontsize=16)
    ax.set_xticks(ind + width * (M - 1) / 2)
    ax.set_xticklabels(mapped_models, fontsize=12)
    ax.legend(title='Subsets', fontsize=12, title_fontsize=12)
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)

    # Display the plot
    plt.tight_layout()

    plt.savefig(save_path + saveas)
    plt.close()

def main():
    hyp_dir = "../train_evaluate/out/carbontracker/hyps/"
    final_train_dir = "../train_evaluate/out/carbontracker/final_train/"
    final_test_dir = "../train_evaluate/out/carbontracker/final_test/"
    data = get_carbon(final_test_dir)
    viz_carbon(data, saveas="carbon_bars_test.png")


if __name__ == "__main__":
    main()
