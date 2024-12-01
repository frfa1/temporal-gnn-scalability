import sys
import pprint
import pickle
sys.path.append("..")
import copy
from loader.wpf_data import ENWPFDataset
from loader.wpfTemporalGraph import ENWPFDatasetWrapper
from models.simple_GRU import SingleGRU, EfficientGCNGRU, GRUModel, GCN_GRUModel
from models.PGeomTemp import RecurrentGCN, A3TGCNModel, A3TGCN2Model
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
from torch.profiler import profile, record_function, ProfilerActivity

def load_data(subset="3", distance=500):
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
    if subset == "1":
        geo_subset = [
            [566000, 572000], # X
            [6330000, 6338000] # Y
            ] # ~15 turbines
    elif subset == "2":
        geo_subset = [
            [560000, 580000], # X
            [6310000, 6340000] # Y
            ] # ~67 turbines
    else:
        geo_subset = [
            [530000, 601000], # X
            [6310000, 6410000] # Y
            ] # 348 turbines
    options_train = { # Preprocessing for train
        "remove_nan": True,
        "handle_degrees": True,
        "handle_outliers": {
                "n_neighbors": 20,
                "contamination": 0.1,
                "interpolation": "KNN",
                "knn_n": 3
            },
    }
    train_data = ENWPFDataset(parameters=parameters, flag='train', subset=geo_subset, subset_forecast=int(subset), options=options_train)
    print("Loaded train")

    options_test = train_data.options
    options_test.pop("handle_outliers") # Don't handle outliers on test
    """options_test = { # Preprocessing for test
        "train_stats": {
            "means": torch.from_numpy(train_data.means),
            "stds": torch.from_numpy(train_data.stds),
            "mins": torch.from_numpy(train_data.mins),
            "maxs": torch.from_numpy(train_data.maxs),  
        },
        "train_stats_forecast": {
            "means": torch.from_numpy(train_data.means_forecast),
            "stds": torch.from_numpy(train_data.stds_forecast),
            "mins": torch.from_numpy(train_data.mins_forecast),
            "maxs": torch.from_numpy(train_data.maxs_forecast),  
        },
        "remove_nan": True,
        "handle_degrees": True
    }"""
    test_data = ENWPFDataset(parameters=parameters, flag='test', subset=geo_subset, subset_forecast=int(subset), options=options_test)
    print("Loaded test")

    train_stats = {
        "means": train_data.means,
        "stds": train_data.stds,
        "mins": train_data.mins,
        "maxs": train_data.maxs,
    }
    train_stats_forecast = {
        "means": train_data.means_forecast,
        "stds": train_data.stds_forecast,
        "mins": train_data.mins_forecast,
        "maxs": train_data.maxs_forecast,
    }
    for k in train_stats:
        train_stats[k] = torch.from_numpy(train_stats[k]).unsqueeze(0)
    for k in train_stats_forecast:
        train_stats_forecast[k] = torch.from_numpy(train_stats_forecast[k]).unsqueeze(0).unsqueeze(0)

    return train_data, test_data, train_stats, train_stats_forecast

def train(model_name, test=False, subset="3", distance=500):
    subset = str(subset)

    train_data, test_data, train_stats, train_stats_forecast = load_data(subset, distance) # Load both sets

    input_size = train_data.array_data.shape[-1] # number of features for one time sample
    forecast_size = train_data.forecast_array.shape[-1]
    input_seq_length = 48

    ### Hyperparameters ###
    output_size = 24  # Forecast window. Prediction ahead, 24 hours default

    # Training - hyp
    num_epochs = 20
    if subset == "1":
        batch_size = 24
    elif subset == "2":
        batch_size = 7
    else:
        batch_size = 2
    if model_name == "A3TGCNModel":
        batch_size = 100
    criterion = nn.MSELoss() # Instantiate loss function - re-used for all models as it doesn't hold a state

    print("BATCH SIZE:", batch_size)

    # util - hyp
    num_workers = 0 # 0 for cuda initialization error

    n_turbines = train_data.n_turbines
    data_columns = train_data.data_columns
    data_columns_forecast = train_data.data_columns_forecast
            
    if model_name == "SimpleGRU":
        if subset == "1":
            embedding_dim = 8
            hidden_size = 32
            lr = 0.0005
        elif subset == "2":
            embedding_dim = 8
            hidden_size = 64
            lr = 0.0005
        else:
            embedding_dim = 8
            hidden_size = 64
            lr = 0.0005
        model = SingleGRU(n_turbines, input_size, forecast_size, hidden_size, output_size, embedding_dim=embedding_dim, num_layers=1, bidirectional=True, dropout=0.3)
    #elif model_name == "RecurrentGCN":
    #    model = RecurrentGCN(node_features=input_size, filters=filter_size, k=k, output_size=output_size)
    elif model_name == "A3TGCNModel":
        if subset == "1":
            hidden_size = 32
            lr = 0.001
        elif subset == "2":
            hidden_size = 16
            lr = 0.001
        else:
            hidden_size = 64
            lr = 0.001
        model = A3TGCNModel(node_features=input_size, forecast_size=forecast_size, hidden_size=hidden_size, periods=input_seq_length, output_size=output_size, batch_size=batch_size)
    elif model_name == "ChebGRU":
        if subset == "1":
            embedding_dim = 8
            gcn_channels = 8
            hidden_size = 16
            k = 2
            lr = 0.0005
        elif subset == "2":
            embedding_dim = 4
            gcn_channels = 8
            hidden_size = 16
            k = 2
            lr = 0.0005
        else:
            embedding_dim = 4
            gcn_channels = 16
            hidden_size = 64
            k = 1
            lr = 0.001
        model = EfficientGCNGRU(n_turbines=n_turbines, input_size=input_size, forecast_size=forecast_size, gcn_channels=gcn_channels, hidden_size=hidden_size, output_size=output_size, gcn_layer="ChebConv", k=k, embedding_dim=embedding_dim, num_layers=1, dropout=0.3)
    elif model_name == "GCNConv":
        if subset == "1":
            embedding_dim = 8
            gcn_channels = 8
            hidden_size = 32
            lr = 0.0005
        elif subset == "2":
            embedding_dim = 8
            gcn_channels = 8
            hidden_size = 16
            lr = 0.0005
        else:
            embedding_dim = 8
            gcn_channels = 16
            hidden_size = 16
            lr = 0.001
        model = EfficientGCNGRU(n_turbines=n_turbines, input_size=input_size, forecast_size=forecast_size, gcn_channels=gcn_channels, hidden_size=hidden_size, output_size=output_size, gcn_layer="GCNConv", embedding_dim=embedding_dim, num_layers=1, dropout=0.3)
    elif model_name == "GCNConv_Edge":
        if subset == "1":
            embedding_dim = 8
            edge_conv = 8
            gcn_channels = 16
            hidden_size = 16
            lr = 0.001
        elif subset == "2":
            embedding_dim = 8
            edge_conv = 8
            gcn_channels = 16
            hidden_size = 16
            lr = 0.0005
        else:
            embedding_dim = 4
            edge_conv = 8
            gcn_channels = 16
            hidden_size = 16
            lr = 0.001
        model = EfficientGCNGRU(n_turbines=n_turbines, input_size=input_size, forecast_size=forecast_size, gcn_channels=gcn_channels, hidden_size=hidden_size, output_size=output_size, edge_conv_out_channels=edge_conv, gcn_layer="GCNConv", embedding_dim=embedding_dim, num_layers=1, dropout=0.3)

    train_data.update_input_seq_length(input_seq_length)
    test_data.update_input_seq_length(input_seq_length)
    train_data.update_output_seq_length(output_size)
    test_data.update_output_seq_length(output_size)

    # To GPU if possible
    model = model.to(device)
    print(model)

    if not (model_name == "SimpleGRU"):
        ### For GNNs

        # Get graph
        loader = ENWPFDatasetWrapper(distance_method="exponential_decay_weight", threshold=distance) # Threshold should be meters between turbines
        train_dataset = loader.get_dataset(wpf_object=train_data)
        test_dataset = loader.get_dataset(wpf_object=test_data)

        ## Batching the data
        train_input = np.array(train_dataset.features)
        train_forecasts = np.array(train_dataset.forecasts)
        train_target = np.array(train_dataset.targets)
        train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor)#.to(device)
        train_forecasts_tensor = torch.from_numpy(train_forecasts).type(torch.FloatTensor)
        train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor)#.to(device)
        train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor, train_forecasts_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=batch_size, shuffle=False, drop_last=True)

        test_input = np.array(test_dataset.features)
        test_forecasts = np.array(test_dataset.forecasts)
        test_target = np.array(test_dataset.targets)
        test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor)#.to(device)  # (B, N, F, T)
        test_forecasts_tensor = torch.from_numpy(test_forecasts).type(torch.FloatTensor)
        test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor)#.to(device)  # (B, N, T)
        test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor, test_forecasts_tensor)
        test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=batch_size, shuffle=False, drop_last=True)

        # Load static graph once
        for snapshot in train_dataset:
            static_edge_index = snapshot.edge_index.to(device)
            static_edge_weight = snapshot.edge_attr.to(device)
            break
        graph = True

    else:
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=False, # Order is important for time-series
            drop_last=True,
            num_workers=num_workers
            )
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False, # Order is important for time-series
            drop_last=True,
            num_workers=num_workers
            )
        static_edge_index = None
        static_edge_weight = None
        graph = False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) # ADAM adapts learning rate, so overkill to finetune

    """#Initialize Variables for EarlyStopping
    best_loss = float('inf')
    best_model_weights = None
    patience = 5"""

    if test == True:
        tracker = CarbonTracker(epochs=1, epochs_before_pred=0, monitor_epochs=-1, verbose=3,
            log_dir="/home/frfa/thesis/train_evaluate/out/carbontracker/final_test/subset_" + subset + "/" + model_name, log_file_prefix="param"
            )
        tracker.epoch_start()
        saved_model_path = "/home/frfa/thesis/train_evaluate/out/final_models/subset_" + subset + "/" + model_name + ".pth"
        model.load_state_dict(torch.load(saved_model_path))
        test_metrics = test_model(
            model,
            test_loader,
            train_stats,
            train_stats_forecast,
            data_columns,
            data_columns_forecast,
            optimizer,
            criterion,
            only_loss=False,
            graph = graph,
            static_edge_index=static_edge_index, # None if simple model
            static_edge_weight=static_edge_weight
            )
        tracker.epoch_end()
        tracker.stop()
        print("TEST METRICS:")
        print(test_metrics)
        return test_metrics
    else:
        epoch_train_losses = []
        tracker = CarbonTracker(epochs=1, epochs_before_pred=0, monitor_epochs=-1, verbose=3,
            log_dir="/home/frfa/thesis/train_evaluate/out/carbontracker/final_train/subset_" + subset + "/" + model_name, log_file_prefix="param"
            )
        tracker.epoch_start()
        for epoch in range(num_epochs):
            print("* EPOCH:", epoch)

            # Training phase
            avg_train_loss = train_model(
                model,
                train_loader,
                train_stats,
                train_stats_forecast,
                data_columns,
                data_columns_forecast,
                optimizer,
                criterion,
                graph = graph,
                static_edge_index=static_edge_index, # None if simple model
                static_edge_weight=static_edge_weight
                )
            print("Avg. train loss:", avg_train_loss)
            epoch_train_losses.append(avg_train_loss)

        # Save model
        torch.save(model.state_dict(), "/home/frfa/thesis/train_evaluate/out/final_models/subset_" + subset + "/" + model_name + ".pth")
        tracker.epoch_end()
        tracker.stop()
        print("TRAIN EPOCH LOSESS:")
        print(epoch_train_losses)
        return epoch_train_losses

def train_model(
    model,
    dataloader,
    train_stats,
    train_stats_forecast,
    data_columns,
    data_columns_forecast,
    optimizer,
    criterion,
    graph=False,
    static_edge_index=None,
    static_edge_weight=None
    ):
    model.train()
    total_loss = 0.0
    loss_list = []
    for inputs, targets, forecasts in dataloader:

        """print("SHAPES:")
        print(inputs.shape, targets.shape, forecasts.shape)
        for k in train_stats:
            print(train_stats[k].shape)
        for k in train_stats_forecast:
            print(train_stats_forecast[k].shape)"""
            
        optimizer.zero_grad()
        if graph == True: # For GNNs
            inputs = inputs.permute(0, 3, 1, 2) # (B, N, M, T) -> (B, T, N, M)
            forecasts = forecasts.permute(0, 3, 1, 2)
            targets = targets.permute(0, 3, 1, 2)
            num_turbines = inputs.shape[2]
            turb_ids = torch.from_numpy(np.arange(num_turbines)).to(device)
            inputs = scale_batch_wrapper(inputs, train_stats, data_columns).to(device)
            targets = scale_batch_wrapper(targets, train_stats, data_columns).to(device)
            forecasts = scale_batch_wrapper(forecasts, train_stats_forecast, data_columns_forecast, forecast=True).to(device)
            outputs = model(inputs, forecasts, turb_ids, static_edge_index, static_edge_weight)
        else: # For simple GRU
            inputs = scale_batch_wrapper(inputs, train_stats, data_columns).to(device)
            targets = scale_batch_wrapper(targets, train_stats, data_columns).to(device)
            forecasts = scale_batch_wrapper(forecasts, train_stats_forecast, data_columns_forecast, forecast=True).to(device)
            num_turbines = inputs.shape[2]
            turb_ids = torch.from_numpy(np.arange(num_turbines)).to(device)
            outputs = model(inputs, forecasts, turb_ids)

        reshaped_targets = targets[:,:,:,-1]
        loss = criterion(outputs, reshaped_targets)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    avg_train_loss = sum(loss_list) / len(loss_list) # Average train loss for each sample
    return avg_train_loss

def test_model(
    model,
    dataloader,
    train_stats,
    train_stats_forecast,
    data_columns,
    data_columns_forecast,
    optimizer,
    criterion,
    only_loss=True,
    graph=False,
    static_edge_index=None,
    static_edge_weight=None
    ):
    model.eval()
    total_loss = 0.0
    if only_loss:
        loss_list = []
    else:
        targets_list = []
        pred_list = []
    for inputs, targets, forecasts in dataloader:

        if graph == True: # For GNNs
            inputs = inputs.permute(0, 3, 1, 2) # (B, N, M, T) -> (B, T, N, M)
            forecasts = forecasts.permute(0, 3, 1, 2)
            targets = targets.permute(0, 3, 1, 2)
            inputs = scale_batch_wrapper(inputs, train_stats, data_columns).to(device)
            targets = scale_batch_wrapper(targets, train_stats, data_columns).to(device)
            forecasts = scale_batch_wrapper(forecasts, train_stats_forecast, data_columns_forecast, forecast=True).to(device)
            num_turbines = inputs.shape[2]
            turb_ids = torch.from_numpy(np.arange(num_turbines)).to(device)
            outputs = model(inputs, forecasts, turb_ids, static_edge_index, static_edge_weight)
        else: # For simple GRU
            inputs = scale_batch_wrapper(inputs, train_stats, data_columns).to(device)
            targets = scale_batch_wrapper(targets, train_stats, data_columns).to(device)
            forecasts = scale_batch_wrapper(forecasts, train_stats_forecast, data_columns_forecast, forecast=True).to(device)
            num_turbines = inputs.shape[2]
            turb_ids = torch.from_numpy(np.arange(num_turbines)).to(device)
            outputs = model(inputs, forecasts, turb_ids)

        reshaped_targets = targets[:,:,:,-1]

        if only_loss: 
            loss = criterion(outputs, reshaped_targets)
            loss_list.append(loss.item())
        else:
            targets_list.append(reshaped_targets.detach().cpu())
            pred_list.append(outputs.detach().cpu()) # (C, B, T, N)

    if only_loss:
        avg_test_loss = sum(loss_list) / len(loss_list) # Average train loss for each sample
        return avg_test_loss
    else:
        targets_list = torch.cat(targets_list, dim=0) # (C, B, T, N) -> (B, T, N)
        pred_list = torch.cat(pred_list, dim=0)

        mins = train_stats["mins"][:,:,:,-1].detach().cpu() # (1, 1, 15, 7) -> (1, 1, 15)
        maxs = train_stats["maxs"][:,:,:,-1].detach().cpu()

        print("SHAPES BEFORE RESCALE")
        print(targets_list.shape, pred_list.shape)
        print(mins.shape, maxs.shape)

        rescaled_targets = targets_list * (maxs - mins + 1e-7) + mins
        rescaled_preds = pred_list * (maxs - mins + 1e-7) + mins
        
        rescaled_targets = rescaled_targets.permute(0,2,1).numpy() # (B, T, N) -> (B,N,T)
        rescaled_preds = rescaled_preds.permute(0,2,1).numpy()
        
        print("EVALUATION DATA:")
        print(rescaled_targets.shape)
        print(rescaled_preds.shape) 
        print(mins.shape)
        print(maxs.shape)

        avg_mae, avg_rmse, avg_score, all_mae, all_rmse, time_mae, time_rmse = metrics.regressor_scores(rescaled_preds, rescaled_targets)
        area_mae, area_rmse = metrics.area_regressor_scores(rescaled_preds, rescaled_targets)
        test_metrics = {
            "avg_mae": avg_mae,
            "avg_rmse": avg_rmse,
            "avg_score": avg_score,
            "area_mae": area_mae,
            "area_rmse": area_rmse,
            "turbines_mae": all_mae,
            "turbines_rmse": all_rmse,
            "time_mae": time_mae,
            "time_rmse": time_rmse,
        }
        return test_metrics

def main():
    run = sys.argv[1]
    model_name = sys.argv[2]
    subset = str(sys.argv[3])
    distance = int(sys.argv[4])

    if run == "train":
        print("Training model:", model_name, "|", "Subset:", subset)
        metrics = train(model_name=model_name, test=False, subset=subset, distance=distance)
    else:
        print("Testing model:", model_name, "|", "Subset:", subset)
        metrics = train(model_name=model_name, test=True, subset=subset, distance=distance)
        with open("/home/frfa/thesis/train_evaluate/out/final_scores/subset_" + subset + "/" + model_name + ".pkl", 'wb') as f:
            pickle.dump(metrics, f)
    print(metrics)

if __name__ == "__main__":
    main()