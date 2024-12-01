import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
#from loader.wpf_data import ENWPFDataset
from data_splits import time_series_cross_validation
from loader.wpf_data import ENWPFDataset


save_path = "../figs/"

def plot_metrics_from_epochs(input_dict, saveas):
    keys = list(input_dict.keys())
    
    # Check if both 'train_loss' and 'val_loss' are present
    has_train_val_loss = 'train_loss' in keys and 'val_loss' in keys
    
    # Number of plots needed
    num_plots = len(keys) - 1 if has_train_val_loss else len(keys)
    
    fig, axs = plt.subplots(num_plots, 1, figsize=(8, num_plots * 4))
    
    # If there is only one plot, axs is not a list but a single Axes object
    if num_plots == 1:
        axs = [axs]
    
    plot_idx = 0
    
    # Plot 'train_loss' and 'val_loss' together if they both exist
    if has_train_val_loss:
        axs[plot_idx].plot(input_dict['train_loss'], marker='o', label='train_loss')
        axs[plot_idx].plot(input_dict['val_loss'], marker='o', label='val_loss')
        axs[plot_idx].set_title('Plot for train_loss and val_loss')
        axs[plot_idx].set_xlabel('Epoch')
        axs[plot_idx].set_ylabel('Loss')
        axs[plot_idx].legend()
        axs[plot_idx].grid(True)
        plot_idx += 1
    
    # Plot the rest of the keys
    for key in keys:
        if key in ['train_loss', 'val_loss']:
            continue
        axs[plot_idx].plot(input_dict[key], marker='o')
        axs[plot_idx].set_title(f'Plot for {key}')
        axs[plot_idx].set_xlabel('Epoch')
        axs[plot_idx].set_ylabel(key)
        axs[plot_idx].grid(True)
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(save_path + saveas)

def plot_time_series_splits(data, saveas="time_series_splits.png"):
    """
        Data: Array of shape (n_cv_splits, 2), where the latter dimension is train and validation size
    """
    n_folds = data.shape[0]
    # Splits on the y-axis
    splits = np.arange(1, n_folds+1)
    # Colors for each segment of the bar
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    # Create the horizontal, stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_height = 0.3  # Thinner bars
    for i in range(len(data[0])):
        # Get the values for each segment
        segment_values = [d[i] for d in data]
        # Determine the left position for each segment
        left_values = np.sum([data[j][:i] for j in range(len(data))], axis=1)
        # Plot each segment
        if i == 0:
            segment = "Train"
        else:
            segment = "Val"
        ax.barh(splits, segment_values, left=left_values, color=colors[i], edgecolor='white', height=bar_height, label=segment)
    # Adding labels and title
    ax.set_xlabel('Sample timestamps', fontsize=14)
    ax.set_ylabel('CV iteration', fontsize=14)
    ax.set_yticks(splits)
    ax.set_yticklabels([n_folds+1-i for i in splits], fontsize=12)
    ax.set_title('Time Series Cross Validation', fontsize=16)
    ax.legend(title='Splits', title_fontsize=12, fontsize=10)
    # Grid and layout adjustments
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    # Saving the figure as a .png file
    plt.savefig(save_path + saveas)
    plt.close() # Close


def main():

    # Just for testing the vizualisation
    """epoch_val_metrics = {
        'area_mae': [1000, 900, 950, 800, 750, 500],
        'area_rmse': [2300, 1600, 950, 1300, 750, 500],
        'area_score': [1500, 1200, 1000, 900, 800, 500], 
        'avg_mae': [100, 90, 95, 80, 75, 50],
        'avg_rmse': [100, 90, 95, 80, 75, 50],
        'avg_score': [100, 90, 95, 80, 75, 50],
        'train_loss': [100, 90, 50, 30, 20, 7], 
        'val_loss': [100, 90, 50, 30, 15, 25], 
    }
    plot_metrics_from_epochs(epoch_val_metrics, saveas="epoch_metrics.png")"""

    parameters = [
            "mean_wind_speed",
            "mean_wind_dir",
            "mean_temp",
            #"min_temp",
            "mean_relative_hum",
            #"max_wind_speed_10min",
            #"max_wind_speed_3sec",
            "mean_pressure",
            "mean_cloud_cover",
            ]
    data_years = [2018, 2019]
    subset = [
        [530000, 601000], # X
        [6310000, 6410000] # Y
        ]
    #subset = [
    #    [566000, 572000], # X
    #    [6330000, 6338000] # Y
    #    ]
    subset_turb = None
    # Processing options
    options = {
        "handle_degrees": True
    }
    optionsFold = { # Per fold preprocessing
        "remove_nan": True,
        "standardize": True,
        #"handle_outliers": {
        #        "n_neighbors": 20,
        #        "contamination": 0.1,
        #        "interpolation": "KNN",
        #        "knn_n": 2
        #    },
        #"handle_degrees": True
    }
    train_data = ENWPFDataset(parameters=parameters, flag='train', subset=subset, subset_turb=subset_turb, options=options)

    n_folds = 3
    data = np.zeros((n_folds, 2)) # (n_folds, 2 for train and val size)
    idx = 0
    for train_indices, val_indices, _, _, _, _ in time_series_cross_validation(train_data, num_folds=n_folds, return_indices=True):
        data[idx, 0] = len(train_indices)
        data[idx, 1] = len(val_indices)
        idx += 1
    plot_time_series_splits(data, saveas="time_series_splits.png")
    sorted_data = data[data[:, 0].argsort()[::-1]]
    plot_time_series_splits(sorted_data, saveas="time_series_splits_transpose.png")

if __name__ == "__main__":
    main()