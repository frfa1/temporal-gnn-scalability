import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pickle

save_path = "../figs/"

MODELS = [
    "A3TGCNModel", "ChebGRU", "GCNConv", "GCNConv_Edge", "SimpleGRU"
]
def get_results(folder):
    for subset in range(1,4):
        for model in MODELS:
            model_dir = folder + "subset_" + str(subset) + "/" + model
            results = model_dir + ".pkl"
            with open(results, 'rb') as file:
                results_dict = pickle.load(file)
                print(results)
                continue
model_name_mapping = {
    'SimpleGRU': 'Turbine-level GRU',
    'GCNConv': 'GCNConv-GRU',
    'ChebGRU': 'ChebConv-GRU',
    'GCNConv_Edge': 'EdgeConv-GCNConv-GRU',
    'A3TGCNModel': 'A3TGCN',
    'mean': 'mean',
    'persistence': 'persistence',
}

def plot_time_rmse(folder, saveas="results_time_ahead.png"):
    """
    Plots the RMSE over time for different models and subsets.
    
    Parameters:
    - folder (str): The base directory containing the results.
    - saveas (str): The filename to save the plot as.
    """
    MODELS = [
        "A3TGCNModel", "ChebGRU", "GCNConv", "GCNConv_Edge", "SimpleGRU"
    ]
    MODELS += ["mean", "persistence"]
    subsets = [1, 2, 3]
    
    # **Assign a unique color to each model**
    num_models = len(MODELS)
    color_map = cm.get_cmap('tab10', num_models)  # 'tab10' provides up to 10 distinct colors
    model_colors = {model: color_map(i) for i, model in enumerate(MODELS)}
    
    # **Create subplots stacked vertically with shared x-axis**
    fig, axes = plt.subplots(
        nrows=len(subsets),
        ncols=1,
        figsize=(12, 5 * len(subsets)),  # Adjust height based on number of subsets
        sharex=True,
        sharey=True
    )
    
    # If there's only one subset, axes is not a list; make it a list for consistency
    if len(subsets) == 1:
        axes = [axes]
    
    # Initialize a list to collect handles and labels for the legend
    legend_handles = []
    legend_labels = []
    
    # Iterate over each subset to create individual plots
    for idx, subset in enumerate(subsets):
        ax = axes[idx]
        for model in MODELS:
            # Construct the path to the results file
            model_dir = os.path.join(folder, f"subset_{subset}", model)
            results_file = f"{model_dir}.pkl"
            
            # Check if the results file exists to prevent errors
            if not os.path.isfile(results_file):
                logging.warning(f"{results_file} does not exist. Skipping model '{model}' for subset {subset}.")
                continue
            
            # Load the results dictionary
            try:
                with open(results_file, 'rb') as file:
                    results_dict = pickle.load(file)
            except Exception as e:
                logging.warning(f"Failed to load {results_file}: {e}. Skipping model '{model}' for subset {subset}.")
                continue
            
            # Extract RMSE values
            time_rmse = results_dict.get('time_rmse', [])
            if not time_rmse:
                logging.warning(f"'time_rmse' not found in {results_file}. Skipping model '{model}' for subset {subset}.")
                continue
            x = range(1, len(time_rmse) + 1)  # Assuming each element corresponds to an hour ahead
            
            # Translate model name
            mapped_name = model_name_mapping.get(model, model)
            
            # Plot RMSE
            line, = ax.plot(
                x, 
                time_rmse, 
                marker='o', 
                linestyle='-', 
                label=mapped_name,
                color=model_colors[model]
            )
            
            # Collect handles and labels for the legend only from the first subplot
            if idx == 0:
                legend_handles.append(line)
                legend_labels.append(mapped_name)
        
        # **Set Y-Axis to Start at 0**
        ax.set_ylim(bottom=0)
        
        # **Set Subplot Title for Each Subset**
        ax.set_title(f"Subset {subset}", fontsize=16)
        
        # **Enable Grid**
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        ax.set_ylabel("Hourly RMSE", fontsize=14)
    
    # **Set X-Axis Label for the Bottom Subplot**
    axes[-1].set_xlabel("Hours Forecasted Ahead", fontsize=14)
    
    # **Add the Legend to the Bottom Right of the First Subplot**
    if legend_handles and legend_labels:
        # Place the legend inside the first subplot
        axes[0].legend(
            legend_handles, 
            legend_labels, 
            loc='lower right',
            fontsize=12,
            title="Models",
            title_fontsize=14,
            frameon=True,
            bbox_to_anchor=(1, 0)  # Adjust the position as needed
        )

    plt.tight_layout()

    # **Save the Figure to the Specified Path**
    full_save_path = os.path.join(save_path, saveas)
    plt.savefig(full_save_path)
    plt.close()

def main():
    folder = "/home/frfa/thesis/train_evaluate/out/final_scores/"
    plot_time_rmse(folder)
    #get_results(folder)


if __name__ == "__main__":
    main()
