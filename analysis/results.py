import pickle
import numpy as np
import matplotlib.pyplot as plt

save_path = "../figs/"

def get_results(models, subset="1"):
    results = []
    for model in models:
        print(model)
        with open('../train_evaluate/out/final_scores/subset_' + str(subset) + '/' + model + '.pkl', 'rb') as handle:
            b = pickle.load(handle)
            results.append(b)

            for metric in b:
                print(metric)
                print("Mean:", np.mean(b[metric]))
                print("STD:", np.std(b[metric]))
        print("--")

def plot_bar_means_with_std(models, subset="1", saveas="grouped_results_subset_"):
    # Load results
    results = []
    for model in models:
        with open('../train_evaluate/out/final_scores/subset_' + str(subset) + '/' + model + '.pkl', 'rb') as handle:
            b = pickle.load(handle)
            results.append(b)
    
    # Get metrics (assume all models have the same metrics)
    metrics = list(results[0].keys())  # Dynamically get the number of metrics
    metrics = [k for k in results[0].keys() if k != "area_mae" and k != "area_rmse"]

    # Number of models and metrics
    num_models = len(results)
    num_metrics = len(metrics)  # Automatically adapts to the number of metrics
    
    # Calculate means and standard deviations for each model and metric
    means = []
    std_devs = []
    
    for result in results:
        model_means = [np.mean(result[metric]) for metric in metrics]
        model_stds = [np.std(result[metric]) for metric in metrics]
        means.append(model_means)
        std_devs.append(model_stds)
    
    means = np.array(means)
    std_devs = np.array(std_devs)
    
    # X locations for the groups
    bar_width = 0.2
    spacing = 0.3  # Extra spacing between groups
    index = np.arange(num_models)  # Now group by models instead of metrics
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use a vibrant color palette (e.g., tab10 or Paired)
    colors = plt.get_cmap('Paired')(np.linspace(0, 1, num_metrics))  # Get dynamic colors based on num_metrics
    
    # Iterate over metrics to plot each as a bar for all models
    for i, metric in enumerate(metrics):
        ax.bar(index + i * (bar_width + spacing), means[:, i], bar_width, yerr=std_devs[:, i], 
               label=metric, capsize=5, color=colors[i])

    # Add labels and title
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Across Different Metrics (Subset ' + str(subset) + ')')
    
    # Set ticks for x-axis and label them with model names
    ax.set_xticks(index + (num_metrics - 1) * (bar_width + spacing) / 2)
    ax.set_xticklabels([f'Model {i + 1}' for i in range(num_models)])

    # Add legend
    ax.legend(title="Metrics")
    
    # Add some padding for better layout
    plt.tight_layout()
    
    # Save the plot if a filename is provided
    if saveas:
        plt.savefig(save_path + saveas + str(subset) + ".png")
        print(f"Plot saved as {save_path + saveas + str(subset) + '.png'}")

    # Show the plot
    plt.show()


def main():
    MODELS = [
        "A3TGCNModel", "ChebGRU", "GCNConv", "GCNConv_Edge", "SimpleGRU"
    ]
    #plot_bar_means_with_std(MODELS, saveas="grouped_results_subset_")

    get_results(MODELS)

if __name__ == "__main__":
    main()