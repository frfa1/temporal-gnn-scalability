import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

save_path = "../figs/"

def open_hyp(directory="out/hyps", get_losses=False):
    """
        hyp dict:
            param_number {

            }
    """
    n_folds = 3
    count = 0
    all_losses = {}
    for subset in os.listdir(directory):
        all_losses[str(subset)] = {}
        subset_directory = os.path.join(directory, subset)
        print(subset_directory)
        for filename in os.listdir(subset_directory):
            f = os.path.join(subset_directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                print(f)
                with open(f, 'rb') as fp:
                    hyp_dict = pickle.load(fp)

                    current_best_val = 999
                    
                    for param_number in hyp_dict:
                        current_param = hyp_dict[param_number]

                        best_train_loss = 0
                        best_val_loss = 0
                        best_epoch = 0
                        
                        list_losses = []

                        for fold in range(n_folds):
                            current_fold = current_param[fold]
                            best_train = min(current_fold["train_loss"])
                            best_val = min(current_fold["val_loss"])

                            # Get epoch with best val error
                            best_val_epoch = np.argmin(current_fold["val_loss"])

                            best_train_loss += best_train
                            best_val_loss += best_val
                            best_epoch += best_val_epoch

                            list_losses.append(
                                [current_fold["train_loss"], current_fold["val_loss"]]
                                )

                        best_train_loss = best_train_loss / n_folds
                        best_val_loss = best_val_loss / n_folds
                        best_epoch = best_epoch / n_folds

                        #print(current_param["params"])
                        #print("Train, val losses:", best_train_loss, best_val_loss)
                        #print("--")

                        if best_val_loss < current_best_val:
                            best_param = current_param["params"]

                            current_best_val = best_val_loss
                            param_best_epoch = best_epoch

                            all_losses[str(subset)][str(filename)] = list_losses

                    print("BEST PARAMs:", best_param, "|", current_best_val, "Epochs:", param_best_epoch)
                    print("--")
                    
                count += 1
            #if count == 1:
            #    break
    if get_losses:
        return all_losses
    
def clean_subset_name(name):
    return "Subset " + name.split('_')[-1]

def viz_hyp(saveas="hyp_viz.png"):
    all_losses = open_hyp(get_losses=True)

    # Mapping of model file names to display names
    model_name_mapping = {
        "GRU_hyp.p": "GRU",
        "GCNConv_hyp.p": "GCNConv-GRU",
        "ChebGRU_hyp.p": "ChebConv-GRU",
        "GCNConv_Edge_hyp.p": "EdgeConv-GCNConv-GRU",
        "A3TGCNModel_hyp.p": "A3TGCN"
    }

    # Ordered list of model file names
    ordered_model_names = [
        "GRU_hyp.p",
        "GCNConv_hyp.p",
        "ChebGRU_hyp.p",
        "GCNConv_Edge_hyp.p",
        "A3TGCNModel_hyp.p"
    ]

    # Extract subsets
    subsets = sorted(all_losses.keys())

    # Determine grid size
    n_rows = len(ordered_model_names)
    n_cols = len(subsets)

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3), squeeze=False)

    # Plot data
    for i, model in enumerate(ordered_model_names):
        for j, subset in enumerate(subsets):
            ax = axes[i][j]
            data_list = all_losses.get(subset, {}).get(model, [])
            if data_list:
                last_losses = data_list[-1]  # Select the last set of losses
                if last_losses and len(last_losses) == 2:
                    train_losses, val_losses = last_losses
                    epochs = range(1, len(train_losses) + 1)
                    ax.plot(epochs, train_losses, label='Train Loss')
                    ax.plot(epochs, val_losses, label='Validation Loss')
                    # Set subset titles only on the top row
                    if i == 0:
                        ax.set_title(clean_subset_name(subset), fontsize=14)
                    # Set x-labels only on the bottom row
                    # Add grid
                    ax.grid(True)
                else:
                    ax.axis('off')
            else:
                ax.axis('off')
            # Remove axis labels for cleaner look
            if j != 0:
                ax.set_ylabel('')
            # Set y-labels only on the leftmost column
            if j == 0:
                ax.set_ylabel('')
            if (i == n_rows - 1) or ((i == n_rows - 2) and (j == n_cols - 1)):
                ax.set_xlabel('Epoch', fontsize=12)

    # Adjust layout to have space on the left
    plt.tight_layout(rect=[0.08, 0.05, 1, 0.95])

    # Add model names as row labels on the left
    for i, model in enumerate(ordered_model_names):
        model_display_name = model_name_mapping.get(model, model)
        # Calculate the y-position of the text
        #y_pos = (n_rows - i - 0.5) / n_rows
        #fig.text(0.02, y_pos, model_display_name, fontsize=14, rotation=90, va='center', ha='center')
        #fig.text(0.04, y_pos, model_display_name, fontsize=14, rotation=90, va='center', ha='center')
        ax = axes[i][0]
        pos = ax.get_position()
        # Calculate the y-position of the center of the subplot
        y_pos = pos.y0 + pos.height / 2
        # Place the model name at this y-position
        fig.text(0.04, y_pos, model_display_name, fontsize=14, rotation=90, va='center', ha='center')

    # Add a single y-axis label for the entire figure
    fig.text(0.06, 0.5, 'Losses', fontsize=12, rotation='vertical', va='center', ha='center')

    # Add a single legend for the entire figure
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=12)

    plt.savefig(save_path + saveas)
    plt.close()
    

def main():
    all_losses = open_hyp(get_losses=True)
    #viz_hyp()

if __name__ == "__main__":
    main()