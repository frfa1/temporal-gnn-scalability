import sys
import pprint
import pickle
sys.path.append("..")
import copy
from loader.wpf_data import ENWPFDataset
from models.simple_GRU import GRUModel, SingleGRU
import evaluation.metrics as metrics
from train_evaluate.training_viz import plot_metrics_from_epochs
from preprocessing.feature_processing import standardize_batch, normalize_batch, scale_batch, scale_batch_wrapper
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from data_splits import time_series_cross_validation
from sklearn.model_selection import ParameterGrid
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from carbontracker.tracker import CarbonTracker

def tune_hyperparams(model_name, subset=3):

    print("TUNING HYPs")

    ### Hyperparameters ###
    # Data - hyp
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
    all_params = [parameters]
    #all_params = [parameters[:i] for i in range(2, len(parameters)+1)]
    #all_params = [parameters[:2]]
    input_seq_lengths = [48] #, 168] # 336 = 14 days, 168 = 7 days # Lookback window. Number of input features. Weather parameters + historical power

    # Model - hyp
    embedding_dims = [4, 8]
    hidden_sizes = [16, 32, 64] #, 128] #, 64]  # Number of hidden units in GRU # 32, 64, 256
    #hidden_sizes = [8, 16, 32] #, 32, 64, 128, 256]
    num_layers = [1] #,3]#, 2] #], 2]  # Number of GRU layers # 1, 2, 3

    # Training - hyp
    num_epochs = 30
    criterion = nn.MSELoss() # Instantiate loss function - re-used for all models as it doesn't hold a state
    lrs = [0.001, 0.0005]

    # util - hyp
    num_folds = 3
    num_workers = 2 # 0 for cuda initialization error

    ## Load data ##
    output_sizes = [24]  # Forecast window. Prediction ahead, 24 hours default

    if subset == "1":
        geo_subset = [
            [566000, 572000], # X
            [6330000, 6338000] # Y
            ] # ~15 turbines
        batch_size = 20
    elif subset == "2":
        geo_subset = [
            [560000, 580000], # X
            [6310000, 6340000] # Y
            ] # ~67 turbines
        batch_size = 7
    else:
        geo_subset = [
            [530000, 601000], # X
            [6310000, 6410000] # Y
            ] # 348 turbines
        batch_size = 2 # 2
    subset_turb = None # Number of turbines to randomly subset after coordinate subset
    options = { # Preprocessing for whole train set
        "remove_nan": True,
        "handle_degrees": True,
        "handle_outliers": {
                "n_neighbors": 20,
                "contamination": 0.1,
                "interpolation": "KNN",
                "knn_n": 3
            },
    }

    hyp_dict = {}
    param_grid = {
        "input_seq_length": input_seq_lengths,
        "output_size": output_sizes,
        "embedding_dim": embedding_dims,
        "hidden_size": hidden_sizes,
        "num_layers": num_layers,
        "parameters": all_params,
        "lr": lrs
    }
    param_number = 0
    for param in list(ParameterGrid(param_grid)):
        print("params:")
        print(param)

        # Loading data
        train_data = ENWPFDataset(parameters=param["parameters"], flag='train', subset=geo_subset, subset_turb=subset_turb, subset_forecast=int(subset), options=options)
        print("Loaded data")
        input_size = train_data.array_data.shape[-1] # number of features for one time sample
        forecast_size = train_data.forecast_array.shape[-1]

        input_seq_length = param["input_seq_length"]
        output_size = param["output_size"]
        lr = param["lr"]
        train_data.update_input_seq_length(input_seq_length) # Update input sequence length of train data
        train_data.update_output_seq_length(output_size) # Update input sequence length of train data
        if model_name == "GRU":
            embedding_dim = param["embedding_dim"]
            hidden_size = param["hidden_size"]
            num_layers = param["num_layers"]
        elif model_name == "GNN":
            # TO DO
            continue
            
        splits_dict = {"params": param}
        for fold_i, (train_subset, val_subset, train_stats, train_stats_forecast) in enumerate(time_series_cross_validation(train_data, num_folds)):
            print("STARTING FOLD:", fold_i)

            for k in train_stats:
                train_stats[k] = train_stats[k].unsqueeze(0) # (1, N, M) -> (1,1,N,M)

            if model_name == "GRU":
                #model = EfficientGRU(train_data.n_turbines, input_size, hidden_size, output_size, num_layers)
                #model = GRUModel(train_data.n_turbines, input_size, hidden_size, output_size, num_layers) # (n_turbines, input_size, hidden_size, output_size, num_layers=1)
                #model = SingleGRU(train_data.n_turbines, input_size, hidden_size, output_size, num_layers, bidirectional=True, dropout=0.3)
                model = SingleGRU(train_data.n_turbines, input_size, forecast_size, hidden_size, output_size, embedding_dim=embedding_dim, num_layers=num_layers, bidirectional=True, dropout=0.3)
                #model = SingleGRU2(train_data.n_turbines, input_size, hidden_size, output_size, num_layers=num_layers, bidirectional=True, dropout=0.3)
            elif model_name == "GNN":
                pass
            model = model.to(device) # To GPU if possible
            print(model)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) # Small LR finetune
            epoch_val_metrics = train_evaluate(
                subset,
                model_name,
                optimizer,
                param_number,
                fold_i,
                train_subset,
                val_subset,
                train_stats,
                train_stats_forecast,
                train_data.data_columns,
                train_data.data_columns_forecast,
                model,
                criterion,
                num_epochs,
                batch_size,
                num_workers
                )

            splits_dict[fold_i] = copy.deepcopy(epoch_val_metrics)

            #if fold_i > 2:
            #    break

        hyp_dict[param_number] = copy.deepcopy(splits_dict)
        param_number += 1

    return hyp_dict

def train_evaluate(
    subset, model_name, optimizer, param_number, fold_i, 
    train_subset, val_subset, train_stats, train_stats_forecast,
    data_columns, data_columns_forecast,
    model, criterion, num_epochs, batch_size, num_workers):

    print("STARTING TRAINING")
    print("IS MODEL GPU?")
    print(next(model.parameters()).is_cuda)

    epoch_val_metrics = {
        #'area_mae': [],
        #'area_rmse': [],
        #'area_score': [], 
        #'avg_mae': [],
        #'avg_rmse': [],
        #'avg_score': [],
        'train_loss': [], 
        'val_loss': [], 
    }

    #Initialize Variables for EarlyStopping
    best_loss = float('inf')
    best_model_weights = None
    patience = 5

    tracker = CarbonTracker(epochs=1, epochs_before_pred=0, monitor_epochs=-1, verbose=3,
        log_dir="/home/frfa/thesis/train_evaluate/out/carbontracker/subset_" + subset + "/" + model_name, log_file_prefix="param" + str(param_number) + "_fold" + str(fold_i)
        )
    tracker.epoch_start()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=False, # Order is important for time-series
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
        )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
        )

    for epoch in range(num_epochs):
        #print("* EPOCH:", epoch)

        # Training phase
        avg_train_loss = train_model(
            model,
            epoch,
            train_loader,
            train_stats,
            train_stats_forecast,
            data_columns,
            data_columns_forecast,
            optimizer,
            criterion)
        epoch_val_metrics["train_loss"].append(avg_train_loss) # Add train loss of current CV split to current epoch

        #print("EPOCH VAL METRICS BEFORE VALIDATION")
        #print(epoch_val_metrics)

        # Validation phase - Get validation metrics and add each of them to the metrics dict. Idx is epoch
        val_metrics = evaluate(
            model,
            val_loader,
            train_stats,
            train_stats_forecast,
            data_columns,
            data_columns_forecast,
            criterion)
        for key in val_metrics:
            epoch_val_metrics[key].append(val_metrics[key])

        #print("Avg. train loss:", avg_train_loss, "| Avg. val loss:", val_metrics["val_loss"])
        #print("*****************************************")
        #print(f"Epoch validation metrics for epoch {epoch+1}/{num_epochs}:")
        #print(epoch_val_metrics)
        #print("_________________________________________")

        #if epoch > 0:
        #    break

        # Early stopping
        if epoch > 15: 
            if val_metrics["val_loss"] < best_loss:
                best_loss = val_metrics["val_loss"]
                best_model_weights = copy.deepcopy(model.state_dict())  # Deep copy here      
                patience = 5  # Reset patience counter
            else:
                patience -= 1
                if patience == 0:
                    break

    tracker.epoch_end()
    tracker.stop()

    return epoch_val_metrics

def train_model(
    model, epoch, dataloader,
    train_stats, train_stats_forecast,
    data_columns, data_columns_forecast,
    optimizer, criterion):
    model.train()
    total_loss = 0.0
    testing = 0
    for inputs, targets, forecasts in dataloader:

        inputs = scale_batch_wrapper(inputs, train_stats, data_columns).to(device)
        targets = scale_batch_wrapper(targets, train_stats, data_columns).to(device)
        forecasts = scale_batch_wrapper(forecasts, train_stats_forecast, data_columns_forecast, forecast=True).to(device)

        # Standardize batch
        #inputs, targets = inputs.float(), targets.float()
        #inputs, targets = inputs.to(device), targets.to(device)
        #inputs = ((inputs - train_stats["train_mean"]) / (train_stats["train_std"] + 1e-7)).to(device)
        #targets = ((targets - train_stats["train_mean"]) / (train_stats["train_std"] + 1e-7)).to(device)
        #inputs = scale_batch(inputs, train_stats, data_columns).to(device)
        #targets = scale_batch(targets, train_stats, data_columns).to(device)

        # Shuffle turbines
        num_turbines = inputs.shape[2]  # turbine dimension
        #permutation = np.random.permutation(num_turbines)
        #inputs = inputs[:, :, permutation, :]
        #targets = targets[:, :, permutation, :]
        #turb_ids = torch.from_numpy(permutation).to(device)
        turb_ids = torch.from_numpy(np.arange(num_turbines)).to(device)

        # Get just response variable
        reshaped_targets = targets[:,:,:,-1] # (B, T, N, M) -> (B, T, N) (B*N, T)
        outputs = model(inputs, forecasts, turb_ids)
        loss = criterion(outputs, reshaped_targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(dataloader) # Average train loss for each sample
    return avg_train_loss

def evaluate(
    model, val_loader,
    train_stats, train_stats_forecast,
    data_columns, data_columns_forecast,
    criterion):
    """
        Evaluates on validation data in one time series cross validation.
        Returns dictionary val_metrics containing list of validation scores:
            mae, rmse, score, farm_mae, farm_rmse, farm_score, loss
        Each call to evaluate() appends another set of scores to the lists.
    """
    model.eval()
    step = 0
    pred_batch = []
    gold_batch = []
    input_batch = []
    losses = []
    total_val_loss = 0.0

    # y_scale, y_means = torch.unsqueeze(data_scale[:, :, -1], 0), torch.unsqueeze(data_mean[:, :, -1], 0) Used for other metrics than loss

    with torch.no_grad():
        for inputs, targets, forecasts in val_loader:
            # Standardizing batch
            inputs = scale_batch_wrapper(inputs, train_stats, data_columns).to(device)
            targets = scale_batch_wrapper(targets, train_stats, data_columns).to(device)
            forecasts = scale_batch_wrapper(forecasts, train_stats_forecast, data_columns_forecast, forecast=True).to(device)

            # Turbine IDs (no shuffle val)
            num_turbines = inputs.shape[2]  # turbine dimension
            turb_ids = torch.from_numpy(np.arange(num_turbines)).to(device)

            # Get just response variable
            reshaped_targets = targets[:,:,:,-1] # (B, T, N, M) -> (B, T, N) # Only selects the predictor
            outputs = model(inputs, forecasts, turb_ids) # (B, T, N) (batch_size, time_steps, n_turbines)
            val_loss = criterion(outputs, reshaped_targets) # Loss for the batch
            total_val_loss += val_loss.item()

            """ For other metrics than loss
            # Get re-scaled outputs, relu to change negative power values to zero
            outputs = F.relu(outputs * (y_scale + 1e-7) + y_means)
            outputs = outputs.cpu().numpy()

            # Scaling targets back to original (after standardization) - for metrics
            unscaled_targets = reshaped_targets * (y_scale + 1e-7) + y_means
            unscaled_targets = unscaled_targets.cpu().numpy()

            # Unscaled gold targets and re-scaled model outputs
            pred_batch.append(outputs)
            gold_batch.append(unscaled_targets)
            """
    model.train()

    # Val metrics
    val_metrics = {}
    avg_val_loss = total_val_loss / len(val_loader) # Average validation loss for given time series CV fold

    val_metrics["val_loss"] = avg_val_loss

    ## Evaluation metrics (beside loss)
    """
    pred_batch = np.concatenate(pred_batch, axis=0)  # (batch_size, time_steps, n_turbines)
    gold_batch = np.concatenate(gold_batch, axis=0)

    area_mae, area_rmse  = metrics.area_regressor_scores(pred_batch, gold_batch) # area mae, area rmse
    area_score = (area_mae + area_rmse) / 2
    avg_mae, avg_rmse = metrics.regressor_scores(pred_batch, gold_batch) # Average mae and rmse for turbines
    avg_score = (avg_mae + avg_rmse) / 2

    val_metrics["area_mae"] = area_mae
    val_metrics["area_rmse"] = area_rmse
    val_metrics["area_score"] = area_score

    val_metrics["avg_mae"] = avg_mae
    val_metrics["avg_rmse"] = avg_rmse
    val_metrics["avg_score"] = avg_score
    val_metrics["val_loss"] = avg_val_loss"""

    return val_metrics

def main():
    model_name = "GRU"
    subset = str(sys.argv[1])

    hyp_dict = tune_hyperparams(model_name=model_name, subset = subset) 

    #for split in splits_dict:
    #    plot_metrics_from_epochs(splits_dict[split], saveas="turbine_gru_tscv_" + str(split))

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(hyp_dict)
    with open('/home/frfa/thesis/train_evaluate/out/hyps/subset_' + subset + "/" + model_name + '_hyp.p', 'wb') as fp:
        pickle.dump(hyp_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()


