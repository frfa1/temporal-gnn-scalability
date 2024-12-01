import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
from logging import getLogger
import numpy as np

def mae(pred, gt):
    """
    Desc:
        Mean Absolute Error
    Args:
        pred:
        gt: ground truth vector
    Returns:
        MAE value
    """
    _mae = 0.
    if len(pred) > 0 and len(gt) > 0:
        _mae = np.mean(np.abs(pred - gt))
    return _mae

def mse(pred, gt):
    """
    Desc:
        Mean Square Error
    Args:
        pred:
        gt: ground truth vector
    Returns:
        MSE value
    """
    _mse = 0.
    if len(pred) > 0 and len(gt) > 0:
        _mse = np.mean((pred - gt)**2)
    return _mse

def rmse(pred, gt):
    """
    Desc:
        Root Mean Square Error
    Args:
        pred:
        gt: ground truth vector
    Returns:
        RMSE value
    """
    return np.sqrt(mse(pred, gt))

def area_regressor_scores(preds, gts):
    """
        Regression metrics for whole area
        Input: preds and gts of shape (B, N, T)
        Output: Tuple of scores for whole area
    """
    # Sum preds and gts along turbines to get total power for whole area at each timestamp
    pred, gt = np.sum(preds, 1), np.sum(gts, 1) # (B, N, T) -> (B, T). Summing power of all turbines for each (B, T) pair
    _mae = mae(pred, gt)
    _rmse = rmse(pred, gt)
    return _mae, _rmse

def regressor_scores(preds, gts):
    """
        Regression metrics for each turbine and then summed
        Input: preds and gts of shape (B, N, T)
        Output: Tuple of summed score
    """
    all_mae, all_rmse = [], []
    time_mae, time_rmse = [], []
    B, N, T = preds.shape

    # Turbine scores
    for turb_i in range(N):
        pred = preds[:, turb_i, :] # (B, N, T) -> (B, T) for a single turbine
        gt = gts[:, turb_i, :]
        _mae, _rmse = mae(pred, gt), rmse(pred, gt)
        all_mae.append(_mae)
        all_rmse.append(_rmse)

    # Time ahead scores
    for time_t in range(T):
        pred = preds[:, :, time_t]
        gt = gts[:, :, time_t]
        _mae, _rmse = mae(pred, gt), rmse(pred, gt)
        time_mae.append(_mae)
        time_rmse.append(_rmse)
    
    avg_score = sum(np.add(all_mae, all_rmse)) / (len(all_mae)*2) # (MAE + RSME) / 2
    avg_mae = sum(all_mae) / len(all_mae)
    avg_rmse = sum(all_rmse) / len(all_rmse)
    return avg_mae, avg_rmse, avg_score, all_mae, all_rmse, time_mae, time_rmse
