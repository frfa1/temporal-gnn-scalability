from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset


def time_series_cross_validation(dataset, num_folds=5, return_indices=False):
    effective_length = len(dataset)
    #length = dataset.array_data.shape[0] # Actual dataset length
    fold_size = effective_length // (num_folds + 1)
    for i in range(num_folds):
        val_start = (i + 1) * fold_size
        val_end = (i + 2) * fold_size if i < num_folds - 1 else effective_length
        # Adjust the indices for the training and validation splits
        train_indices = list(range(0, val_start))
        val_indices = list(range(val_start, val_end))
        train_tensor = dataset.array_data[train_indices] # (T, N, M)
        train_forecast_tensor = dataset.forecast_array[train_indices] # (T, O, N, M)

        print("FOLD SHAPES")
        print(train_tensor.shape)
        print(train_forecast_tensor.shape)

        train_mean, train_std = train_tensor.mean(dim=0, keepdim=True), train_tensor.std(dim=0, keepdim=True) # (1, N, M)
        train_min, train_max = train_tensor.amin(dim=0, keepdim=True), train_tensor.amax(dim=0, keepdim=True)
        
        train_mean2, train_std2 = train_forecast_tensor.mean(dim=(0,1), keepdim=True), train_forecast_tensor.std(dim=(0,1), keepdim=True) # (1, 1, N, M)
        train_min2, train_max2 = train_forecast_tensor.amin(dim=(0,1), keepdim=True), train_forecast_tensor.amax(dim=(0,1), keepdim=True)
        
        train_stats = {
            "means": train_mean,
            "stds": train_std,
            "mins": train_min,
            "maxs": train_max,
        }
        train_stats_forecast = {
            "means": train_mean2,
            "stds": train_std2,
            "mins": train_min2,
            "maxs": train_max2
        }
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        if return_indices:
            # For visualizing split
            yield train_indices, val_indices, train_stats, train_stats_forecast
        else:
            #yield train_tensor, val_subset, train_mean, train_std, train_min, train_max
            yield train_subset, val_subset, train_stats, train_stats_forecast
