import sys
import pprint
sys.path.append("..")
import copy
import pickle
from loader.wpf_data import ENWPFDataset
from loader.wpfTemporalGraph import ENWPFDatasetWrapper
from models.PGeomTemp import RecurrentGCN, A3TGCNModel, A3TGCN2Model
from models.simple_GRU import GCN_GRUModel, EfficientGCNGRU
import evaluation.metrics as metrics
from train_evaluate.training_viz import plot_metrics_from_epochs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from data_splits import time_series_cross_validation
from sklearn.model_selection import ParameterGrid
from preprocessing.feature_processing import standardize_batch, normalize_batch, scale_batch, scale_batch_wrapper
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from carbontracker.tracker import CarbonTracker
from torch.profiler import profile, record_function, ProfilerActivity

def tune_hyperparams(model_name, subset=3, distance=500):

    print("TUNING HYPs")

    print("Torcch # CPU threads:", torch.get_num_threads())

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
    input_seq_lengths = [48] #, 48, 168] # 336 = 14 days, 168 = 7 days # Lookback window. Number of input features. Weather parameters + historical power

    # Model - hyp
    embedding_dims = [4, 8]
    hidden_sizes = [16, 32, 64] #, 128]
    #filter_sizes = [12, 24] #, 64] #, 64]  # Number of hidden units in GRU # 32, 64, 256
    num_layers = [1] #, 2, 3] #], 2]  # Number of GRU layers # 1, 2, 3
    ks = [1, 2]#,2,3]

    # Training - hyp
    num_epochs = 30
    criterion = nn.MSELoss() # Instantiate loss function - re-used for all models as it doesn't hold a state
    lrs = [0.001, 0.0005]

    # util - hyp
    num_folds = 3
    num_workers = 2 # 0 for cuda initialization error. 2 should be safe

    ## Load data ##
    output_sizes = [24] # [1, 12, 24]  # Forecast window. Prediction ahead, 24 hours default

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
        batch_size = 2
    if model_name == "A3TGCNModel":
        batch_size = 200
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
    print("BATCH SIZE:", batch_size)
    train_data = ENWPFDataset(parameters=parameters, flag='train', subset=geo_subset, subset_turb=subset_turb, subset_forecast=int(subset), options=options)
    print("Loaded data")
    input_size = train_data.array_data.shape[-1] # number of features for one time sample
    forecast_size = train_data.forecast_array.shape[-1]
    n_turbines = train_data.array_data.shape[1] # (T, N, M)

    # To graph Pytorch Geometric Graph data
    loader = ENWPFDatasetWrapper(distance_method="exponential_decay_weight", threshold=distance) # Threshold should be meters between turbines

    hyp_dict = {}
    param_grid = {
        "input_seq_length": input_seq_lengths,
        "output_size": output_sizes,
        "filter_size": hidden_sizes,
        "lr": lrs
        #"k": ks,
    }
    if model_name != "A3TGCNModel":
        param_grid["embedding_dim"] = embedding_dims
    if (model_name == "RecurrentGCN") or (model_name == "ChebGRU"):
        param_grid["k"] = ks
    if (model_name == "ChebGRU") or (model_name == "GCNConv") or (model_name == "GCNConv_Edge"):
        param_grid["gcn_channels"] = [8, 16]
    if (model_name == "GCNConv_Edge"):
        param_grid["edge_conv_out_channels"] = [8, 16]

    param_number = 0
    for param in list(ParameterGrid(param_grid)):
        print("params:")
        print(param)
        input_seq_length = param["input_seq_length"]
        output_size = param["output_size"]
        lr = param["lr"]
        train_data.update_input_seq_length(input_seq_length) # Update input sequence length of train data
        train_data.update_output_seq_length(output_size) # Update input sequence length of train data
        dataset = loader.get_dataset(wpf_object=train_data)

        filter_size = param["filter_size"]
        #k = param["k"]
            
        splits_dict = {"params": param}
        for fold_i, (train_indices, val_indices, train_stats, train_stats_forecast) in enumerate(time_series_cross_validation(train_data, num_folds, return_indices=True)):
            print("STARTING FOLD:", fold_i)

            for k in train_stats:
                train_stats[k] = train_stats[k].unsqueeze(0)

            ## Batching the data
            train_input = np.array(dataset.features)[train_indices]
            train_forecasts = np.array(dataset.forecasts)[train_indices]
            train_target = np.array(dataset.targets)[train_indices]
            train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor)#.to(device)
            train_forecasts_tensor = torch.from_numpy(train_forecasts).type(torch.FloatTensor)#.to(device)
            train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor)#.to(device)
            train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor, train_forecasts_tensor)
            train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=batch_size, shuffle=False, drop_last=True)

            val_input = np.array(dataset.features)[val_indices]
            val_forecasts = np.array(dataset.forecasts)[val_indices]
            val_target = np.array(dataset.targets)[val_indices]
            val_x_tensor = torch.from_numpy(val_input).type(torch.FloatTensor)#.to(device)  # (B, N, F, T)
            val_forecasts_tensor = torch.from_numpy(val_forecasts).type(torch.FloatTensor)#.to(device)
            val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor)#.to(device)  # (B, N, T)
            val_dataset_new = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor, val_forecasts_tensor)
            val_loader = torch.utils.data.DataLoader(val_dataset_new, batch_size=batch_size, shuffle=False, drop_last=True)

            # Load static graph once
            for snapshot in dataset:
                static_edge_index = snapshot.edge_index.to(device)
                static_edge_weight = snapshot.edge_attr.to(device)
                break

            if model_name == "RecurrentGCN":
                model = RecurrentGCN(node_features=input_seq_length, filters=filter_size, k=k, output_size=output_size)
            elif model_name == "A3TGCNModel":
                model = A3TGCNModel(node_features=input_size, forecast_size=forecast_size, hidden_size=filter_size, periods=input_seq_length, output_size=output_size, batch_size=batch_size)
            elif model_name == "ChebGRU":
                model = EfficientGCNGRU(n_turbines=n_turbines, input_size=input_size, forecast_size=forecast_size, gcn_channels=param["gcn_channels"], hidden_size=filter_size, output_size=output_size, gcn_layer="ChebConv", k=param["k"], embedding_dim=param["embedding_dim"], num_layers=1, dropout=0.3)
                #model = GCN_GRUModel(n_turbines=n_turbines, input_size=input_size, gcn_channels=param["gcn_channels"], hidden_size=filter_size, output_size=output_size, gcn_layer="ChebConv", k=param["k"])
            elif model_name == "GCNConv":
                model = EfficientGCNGRU(n_turbines=n_turbines, input_size=input_size, forecast_size=forecast_size, gcn_channels=param["gcn_channels"], hidden_size=filter_size, output_size=output_size, gcn_layer="GCNConv", embedding_dim=param["embedding_dim"], num_layers=1, dropout=0.3)          
            elif model_name == "GCNConv_Edge":
                model = EfficientGCNGRU(n_turbines=n_turbines, input_size=input_size, forecast_size=forecast_size, gcn_channels=param["gcn_channels"], hidden_size=filter_size, output_size=output_size, edge_conv_out_channels=param["edge_conv_out_channels"], gcn_layer="GCNConv", embedding_dim=param["embedding_dim"], num_layers=1, dropout=0.3)

            # To GPU if possible
            model = model.to(device)
            print(model)

            data_dict = {
                "train_loader": train_loader,
                "val_loader": val_loader,
                "static_edge_index": static_edge_index,
                "static_edge_weight": static_edge_weight
            }

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) # Small LR finetune
            epoch_val_metrics = train_evaluate(
                subset,
                model_name,
                optimizer,
                param_number,
                fold_i,
                data_dict,
                train_stats,
                train_stats_forecast,
                train_data.data_columns,
                train_data.data_columns_forecast,
                model,
                criterion,
                num_epochs,
                num_workers
                )
            splits_dict[fold_i] = copy.deepcopy(epoch_val_metrics)

            #if fold_i > 2:
            #    break

        hyp_dict[param_number] = copy.deepcopy(splits_dict)
        param_number += 1

    return hyp_dict

def train_evaluate(
    subset, 
    model_name, 
    optimizer, 
    param_number, 
    fold_i, 
    data_dict, 
    train_stats,
    train_stats_forecast,
    data_columns,
    data_columns_forecast,
    model, 
    criterion, 
    num_epochs, 
    num_workers
    ):
    print("STARTING TRAINING")
    print("IS MODEL GPU?")
    print(next(model.parameters()).is_cuda)

    #train_mean = train_mean[0,:,-1].unsqueeze(-1).float() # Just output -> (N, 1)
    #train_std = train_std[0,:,-1].unsqueeze(-1).float()
    #train_min = train_min[0,:,-1].unsqueeze(-1).float()
    #train_max = train_max[0,:,-1].unsqueeze(-1).float()

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
    innit_patience = 5
    patience = innit_patience

    tracker = CarbonTracker(epochs=1, epochs_before_pred=0, monitor_epochs=-1, verbose=3,
        log_dir="/home/frfa/thesis/train_evaluate/out/carbontracker/hyps/subset_" + subset + "/" + model_name, log_file_prefix="param" + str(param_number) + "_fold" + str(fold_i)
        )
    tracker.epoch_start()

    for epoch in range(num_epochs):
        print("* EPOCH:", epoch)

        # Training phase
        avg_train_loss = train_model(
            model_name,
            data_dict,
            model,
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
            model_name,
            data_dict,
            model,
            train_stats,
            train_stats_forecast,
            data_columns,
            data_columns_forecast,
            criterion)
        for key in val_metrics:
            epoch_val_metrics[key].append(val_metrics[key])

        print("Avg. train loss:", avg_train_loss, "| Avg. val loss:", val_metrics["val_loss"])
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
                patience = innit_patience  # Reset patience counter
            else:
                patience -= 1
                if patience == 0:
                    break

    tracker.epoch_end()
    tracker.stop()
    return epoch_val_metrics

def train_model(
    model_name,
    data_dict, 
    model,
    train_stats,
    train_stats_forecast,
    data_columns,
    data_columns_forecast,
    optimizer, 
    criterion
    ):
    model.train()
    loss_list = []
    train_count = 0
    batch_loss = 0
    train_loader = data_dict["train_loader"]
    static_edge_index = data_dict["static_edge_index"]
    static_edge_weight = data_dict["static_edge_weight"]

    """with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=2), 
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as p:"""

    for inputs, targets, forecasts in train_loader:
        inputs = inputs.permute(0, 3, 1, 2) # (B, N, M, T) -> (B, T, N, M)
        forecasts = forecasts.permute(0, 3, 1, 2)
        targets = targets.permute(0, 3, 1, 2)
        inputs = scale_batch_wrapper(inputs, train_stats, data_columns).to(device)
        targets = scale_batch_wrapper(targets, train_stats, data_columns).to(device)
        forecasts = scale_batch_wrapper(forecasts, train_stats_forecast, data_columns_forecast, forecast=True).to(device)
        
        # Shuffle turbines
        num_turbines = inputs.shape[2]  # turbine dimension
        #permutation = np.random.permutation(num_turbines)
        #inputs = inputs[:, :, permutation, :]
        #targets = targets[:, :, permutation, :]
        #turb_ids = torch.from_numpy(permutation).to(device)
        turb_ids = torch.from_numpy(np.arange(num_turbines)).to(device)
        
        reshaped_targets = targets[:,:,:,-1]
        outputs = model(inputs, forecasts, turb_ids, static_edge_index, static_edge_weight)         # Get model predictions
        loss = criterion(outputs, reshaped_targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss.item())
        
        #mean_tensor.permute(1,2,0)
        #norm_snapshot_x = (snapshot.x - train_min) / (train_max - train_min + 1e-7)
        #norm_snapshot_y = (snapshot.y - train_min) / (train_max - train_min + 1e-7)

        #p.step()
        
    #print(p.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15))

    avg_train_loss = sum(loss_list) / len(loss_list) # Average train loss for each sample
    return avg_train_loss

def evaluate(
    model_name, 
    data_dict, 
    model, 
    train_stats,
    train_stats_forecast,
    data_columns,
    data_columns_forecast,
    criterion
    ):
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
    loss_list = []
    total_val_loss = 0.0

    val_loader = data_dict["val_loader"]
    static_edge_index = data_dict["static_edge_index"]
    static_edge_weight = data_dict["static_edge_weight"]
    with torch.no_grad():
        for inputs, targets, forecasts in val_loader:
            inputs = inputs.permute(0, 3, 1, 2) # (B, N, M, T) -> (B, T, N, M)
            forecasts = forecasts.permute(0, 3, 1, 2)
            targets = targets.permute(0, 3, 1, 2)
            inputs = scale_batch_wrapper(inputs, train_stats, data_columns).to(device)
            targets = scale_batch_wrapper(targets, train_stats, data_columns).to(device)
            forecasts = scale_batch_wrapper(forecasts, train_stats_forecast, data_columns_forecast, forecast=True).to(device)
            
            reshaped_targets = targets[:,:,:,-1]

            # Turbine IDs (no shuffle in val)
            num_turbines = inputs.shape[2]  # turbine dimension
            turb_ids = torch.from_numpy(np.arange(num_turbines)).to(device)

            outputs = model(inputs, forecasts, turb_ids, static_edge_index, static_edge_weight) # Get model predictions
            loss = criterion(outputs, reshaped_targets)
            loss_list.append(loss.item())

    model.train()

    # Val metrics
    val_metrics = {}
    avg_val_loss = sum(loss_list) / len(loss_list) # Average validation loss for given time series CV fold
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
    model_name = sys.argv[1]
    subset = str(sys.argv[2]) # 1: Small, 2: Medium, 3: All
    distance = int(sys.argv[3])

    print("Training model:", model_name)
    hyp_dict = tune_hyperparams(model_name = model_name, subset = subset, distance = distance) 

    #for split in splits_dict:
    #    plot_metrics_from_epochs(splits_dict[split], saveas="turbine_gru_tscv_" + str(split))

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(hyp_dict)
    with open('/home/frfa/thesis/train_evaluate/out/hyps/subset_' + subset + "/" + model_name + '_hyp.p', 'wb') as fp:
        pickle.dump(hyp_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main()


