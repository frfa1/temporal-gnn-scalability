import sys
sys.path.append("..")
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from preprocessing.feature_processing import standardize_batch, normalize_batch, scale_batch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from models.baselines import Baselines
import evaluation.metrics as metrics
from train import load_data


def test_baselines(model_name, test=True, subset=3):
    train_data, test_data, train_stats, train_stats_forecast = load_data(subset) # Load both sets
    input_size = train_data.array_data.shape[-1] # number of features for one time sample
    output_size = 24
    num_workers = 0 # 0 for cuda initialization error
    n_turbines = train_data.n_turbines
    data_columns = train_data.data_columns
    data_columns_forecast = train_data.data_columns_forecast
    batch_size = 24

    model = Baselines(output_length=output_size)

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False, # Order is important for time-series
        drop_last=True,
        num_workers=num_workers
        )
    test_metrics = test_model(model, model_name, test_loader, train_stats, train_stats_forecast, data_columns, data_columns_forecast)
    return test_metrics

def test_model(model, model_name, dataloader, train_stats, train_stats_forecast, data_columns, data_columns_forecast):
    targets_list, pred_list = [], []
    for inputs, labels, forecasts in dataloader:
        inputs, labels, forecasts = inputs.float(), labels.float(), forecasts.float()
        


        inputs = inputs[:,:,:,-1] # (B, T, N, M) -> (B, T, N)
        labels = labels[:,:,:,-1]

        if model_name == "persistence":
            outputs = model.predictPersistence(inputs)
        elif model_name == "mean":
            outputs = model.predictMean(inputs)

        targets_list.append(labels) #.detach().cpu())
        pred_list.append(outputs) #.detach().cpu())

    targets_list = torch.cat(targets_list, dim=0) # (C, B, T, N) -> (B, T, N)
    pred_list = torch.cat(pred_list, dim=0)

    targets_list = targets_list.permute(0,2,1).numpy()
    pred_list = pred_list.permute(0,2,1).numpy()

    print("FINAL SHAPES")
    print(targets_list.shape)
    print(pred_list.shape)

    avg_mae, avg_rmse, avg_score, all_mae, all_rmse, time_mae, time_rmse = metrics.regressor_scores(pred_list, targets_list)
    area_mae, area_rmse = metrics.area_regressor_scores(pred_list, targets_list)
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

    print("Testing model:", model_name, "|", "Subset:", subset)
    metrics = test_baselines(model_name=model_name, test=True, subset=subset)
    print("METRICS:")
    print(metrics)
    with open("/home/frfa/thesis/train_evaluate/out/final_scores/subset_" + subset + "/" + model_name + ".pkl", 'wb') as f:
        pickle.dump(metrics, f)


if __name__ == "__main__":
    main()