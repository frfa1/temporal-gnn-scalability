from sklearn.neighbors import LocalOutlierFactor

# Interpolations
#from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from sklearn.impute import KNNImputer

import numpy as np
import pandas as pd
import torch

def remove_nan(array, means):
    """
        Remove nans of each feature in the data for each turbine, with the following methods:
        New: Replace nans with the feature means from the feature-turbine mean array of train data
        /* Old: First, attempts to use the most recent of the previous 10 values of the feature/turbine.
        Secondly, if nan values still exists, use the mean of all the values for the feature/turbine
        Both methods uses data from train
    """
    # Quick solution for tensors
    return torch.where(torch.isnan(array), means.expand_as(array), array)

    new_array = np.copy(array) # (T, N, M) means: (1, N, M)
    timesteps, n_turbines, n_features = new_array.shape

    # Step 0: Use the means of the train
    # for the particular turbine
    for turbine in range(n_turbines):
        for feature in range(n_features):
            current_mean = means[0, turbine, feature]
            nan_mask = np.isnan(new_array[:, turbine, feature]) # Get nan mask for train
            new_array[:, turbine, feature][nan_mask] = current_mean # Update train nan vals
    return new_array

    """# Step 1: Replace NaN values with the most recent of the previous 10 values
    for turbine in range(n_turbines):
        for feature in range(n_features):
            series = pd.Series(new_array[:, turbine, feature])
            filled_series = series.fillna(method='ffill', limit=10)
            new_array[:, turbine, feature] = filled_series.values
    # Step 2: Replace any remaining NaN values with the mean of all values of the feature/turbine
    for turbine in range(n_turbines):
        for feature in range(n_features):
            current_mean = np.nanmean(new_array[:, turbine, feature])
            nan_mask = np.isnan(new_array[:, turbine, feature])
            new_array[:, turbine, feature][nan_mask] = current_mean
    return (new_array, False)"""

def degrees_to_sin_cos(array, feature_index, data_columns):
    """Convert wind direction degrees feature to two new features: sine and cosine"""
    #new_array = np.copy(array)
    degrees = array[:,:,feature_index]
    radians = torch.deg2rad(degrees) # np.deg2rad(degrees)  # Convert degrees to radians
    sin_values = torch.sin(radians) # np.sin(radians)    # Compute sine
    cos_values = torch.cos(radians) # np.cos(radians)    # Compute cosine
    # Create a new array with additional space for sine and cosine features
    new_features = array.shape[2] + 1  # One original feature replaced by two new features (sine and cosine)
    new_data = torch.zeros((array.shape[0], array.shape[1], new_features)) # np.zeros((array.shape[0], array.shape[1], new_features))
    # Copy over the original features up to the feature_index
    new_data[:, :, :feature_index] = array[:, :, :feature_index]
    # Add the sine values in place of the original feature
    new_data[:, :, feature_index] = sin_values
    new_data[:, :, feature_index + 1] = cos_values
    # Copy over the remaining original features after the feature_indices
    new_data[:, :, feature_index + 2:] = array[:, :, feature_index + 1:]
    new_data_columns = data_columns[:feature_index]
    new_data_columns.append("sin_values")
    new_data_columns.append("cos_values")
    new_data_columns = new_data_columns + data_columns[feature_index + 1:]
    return new_data, new_data_columns

def standardize_batch(batch, means, stds, feature_indices):
    """Standardize the batch array to have mean 0 and standard deviation 1 at feature indices"""
    std_batch = batch.clone()
    std_batch[:,:,:,feature_indices] = (std_batch[:,:,:,feature_indices] - means[:,:,:,feature_indices]) / (stds[:,:,:,feature_indices] + 1e-7) # Epsilon helps with numerical stability
    return std_batch

def normalize_batch(batch, mins, maxs, feature_indices):
    """Normalize batch between 0 and 1 with min and max"""
    nrm_batch = batch.clone()
    # Bug fixing
    if mins == None:
        nrm_batch[:,:,:,feature_indices] = nrm_batch[:,:,:,feature_indices] / 100 # Percentage features
    else:
        nrm_batch[:,:,:,feature_indices] = (nrm_batch[:,:,:,feature_indices] - mins[:,:,:,feature_indices]) / (maxs[:,:,:,feature_indices] - mins[:,:,:,feature_indices] + 1e-7)
    return nrm_batch

def scale_batch(new_batch, train_stats, data_columns, std_indices=[], nrm_indices=[], pct_indices=[]):
    means, stds = train_stats["means"], train_stats["stds"]
    mins, maxs = train_stats["mins"], train_stats["maxs"]
    #new_batch = batch.clone()
    new_batch = standardize_batch(new_batch, means, stds, std_indices)
    new_batch = normalize_batch(new_batch, mins, maxs, nrm_indices)
    new_batch = normalize_batch(new_batch, None, None, pct_indices)
    return new_batch

def scale_batch_wrapper(array, train_stats, data_columns, forecast=False):
    if forecast:
        std_indices = [index for index, value in enumerate(data_columns) if value in ["temperatur_2m", "wind_speed_100m", "wind_speed_10m"]]
        nrm_indices = [index for index, value in enumerate(data_columns) if value in ["None"]]
        pct_indices = [index for index, value in enumerate(data_columns) if value in ["None"]]
    else:
        std_indices = [index for index, value in enumerate(data_columns) if value in ["mean_wind_speed", "mean_temp", "min_temp", "max_wind_speed_10min", "max_wind_speed_3sec", "mean_pressure"]]
        nrm_indices = [index for index, value in enumerate(data_columns) if value in ["VAERDI"]]
        pct_indices = [index for index, value in enumerate(data_columns) if value in ["mean_relative_hum", "mean_cloud_cover"]]
    array = scale_batch(
        array,
        train_stats,
        data_columns,
        std_indices,
        nrm_indices,
        pct_indices
        )
    return array

def standardize(array, feature_indices):
    """Standardize the array to have mean 0 and standard deviation 1 at feature indices"""
    #means = array[:,:,feature_indices].mean(dim=0, keepdim=True) # pytorch
    #stds = array[:,:,feature_indices].std(dim=0, keepdim=True)
    means = np.nanmean(array[:,:,feature_indices], axis=0, keepdims=True)
    stds = np.nanstd(array[:,:,feature_indices], axis=0, keepdims=True)
    std_array = array.copy()
    std_array[:,:,feature_indices] = (array[:,:,feature_indices] - means) / (stds + 1e-7) # Epsilon helps with numerical stability
    return std_array

def standardize_fold(train_array, val_array=None, feature_indices=[]):
    """
    Standardize the array to have mean 0 and standard deviation 1 at feature indices
        Input is a list of a train_array and a val_array, where the mean and stds comes from the train_array
        and both are standarded using this. 
        The purpose is to use solely train information on a given split to standardize - avoiding data leakage.
        Returns tuples of standardized train and val arrays
    """
    if val_array is None: # Singular array standardization
        return (standardize(array, feature_indices), False)
    #means = array[:,:,feature_indices].mean(dim=0, keepdim=True) # pytorch
    #stds = array[:,:,feature_indices].std(dim=0, keepdim=True)
    means = np.nanmean(train_array[:,:,feature_indices], axis=0, keepdims=True)
    stds = np.nanstd(train_array[:,:,feature_indices], axis=0, keepdims=True)
    std_train_array = train_array.copy()
    std_val_array = val_array.copy()
    std_train_array[:,:,feature_indices] = (train_array[:,:,feature_indices] - means) / (stds + 1e-7) # Epsilon helps with numerical stability
    std_val_array[:,:,feature_indices] = (val_array[:,:,feature_indices] - means) / (stds + 1e-7) # Epsilon helps with numerical stability
    print(std_train_array[:,0,-1])
    print(std_val_array[:,0,-1])
    return (std_train_array, std_val_array)

def local_outlier_factor(array, feature_indices, n_neighbors=20, contamination="auto"):
    """
        Input: Numpy array
        Returns a list of list of LOF outlier scores for each point for each wind turbine
    """
    numpy_array = array[:, :, feature_indices]
    list_of_lofs = []
    for i in range(numpy_array.shape[1]):
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        wind_turbine_data = numpy_array[:, i, :]
        is_inlier = lof.fit_predict(wind_turbine_data)
        list_of_lofs.append(is_inlier)
    return list_of_lofs

def local_outlier_factor_fold(train_array, val_array, feature_indices, n_neighbors=20, contamination="auto"):
    """
        Input is a list of a train_array and a val_array
        Returns a tuples with list of list of LOF outlier scores for each point for each wind turbine,
        for train, val respectively
    """
    train_array = train_array[:, :, feature_indices]
    val_array = val_array[:, :, feature_indices]
    list_of_lofs_train = []
    list_of_lofs_val = []
    for i in range(train_array.shape[1]):
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        wind_turbine_data_train = train_array[:, i, :]
        wind_turbine_data_val = val_array[:, i, :]
        lof.fit(wind_turbine_data_train) # Fit just on train
        is_inlier_train = lof.predict(wind_turbine_data_train)
        is_inlier_val = lof.predict(wind_turbine_data_val)
        list_of_lofs.append(is_inlier)
    return (list_of_lofs_train, list_of_lofs_val)

def handle_outliers(array, feature_indices, params, select_turbine=None):
    """
        Outlier handling using Local Outlier Factor (LOF)
        First, outliers are found using local_outlier_factor().

        params: Dictionary containing params for outlier detection and impution method
    
        Returns array with handled outliers, i.e. where power output (last index of feature_indices) is imputed
    """
    array_copy = np.copy(array.numpy())
    list_of_lofs = local_outlier_factor(array_copy, feature_indices, params["n_neighbors"], params["contamination"])

    # array: (T, N, M)
    for turb_idx in range(array_copy.shape[1]):
        if select_turbine: # Only select the input turbine
            if turb_idx != select_turbine:
                continue

        x, y = array_copy[:, turb_idx, feature_indices[0]], array_copy[:, turb_idx, feature_indices[1]] # x and y of (T, N, M)
        turb_lofs = list_of_lofs[turb_idx]

        if params["interpolation"] == "KNN":
            y_short = [y[idx] if turb_lofs[idx] > 0 else np.nan for idx, _ in enumerate(y)] # Replacing outliers in y with np.nan
            df_xy = pd.DataFrame({'x': x, 'y': y_short})
            imputer = KNNImputer(n_neighbors = params["knn_n"])
            imputer.fit(df_xy)
            df_imputed = imputer.transform(df_xy) # Update y (power) with KNN imputations
            x_imputed = df_imputed[:, 0].tolist()
            y_imputed = df_imputed[:, 1].tolist()
            y_imputed = np.clip(y_imputed, 0, np.max(y_short))
            array_copy[:, turb_idx, feature_indices[1]] = y_imputed
        elif params["interpolation"] == "Cubic":
            x_short = np.array([idx for idx, _ in enumerate(turb_lofs) if turb_lofs[idx] > 0]) # A strictly increasing count of timesteps, skipping outliers
            y_short = y[x_short]
            cs = CubicSpline(x_short, y_short)
            x_full = np.arange(len(turb_lofs))
            y_imputed = cs(x_full) # Update y (power) with Cubic imputations            
            y_imputed = np.clip(y_imputed, 0, np.max(y_short))
            new_array[:, turb_idx, feature_indices[1]] = y_imputed

    return torch.from_numpy(array_copy), list_of_lofs # Imputed array and list of outliers
 
def preprocess_factory(
    array,
    options = {},
    data_columns = [],
    train_stats = {}, # means, stds, mins, maxs
    forecast = False,
):
    """
        Apply preprocessing methods based on the provided options dictionary. Option includes statistics to normalize/standardize with
        # Normalisation/Standardisation/Scaling
        # Correlation coeffictions - colinear features
        # Feature conversion - PCA, Autoencoder NN, CNN
        # Outlier detections
        # Differencing - 
        # Signal Processing
        # Feature extraction - PACF, Mutual Information
        # Feature conversion - PCA, Autoencoder NN, CNN

        Input:
            array: (T, N, M)
            options: Dictionary with boolean values
            data_columns: List of column names for data. Used to index specific features

        Output:
            array: (T, N, M)
    """
    if options.get("remove_nan", False):
        array = remove_nan(array, train_stats["means"])
    if options.get("handle_outliers", False) and not (forecast):
        feature_indices = [index for index, value in enumerate(data_columns) if value in ["mean_wind_speed", "VAERDI"]]
        array, _ = handle_outliers(array, feature_indices, params=options["handle_outliers"])
    if options.get("standardize", False):
        if array.dim() == 3: # (T,N,M) -> (1,T,N,M) For batch methods
            array = array.unsqueeze(0)
            for k in train_stats:
                train_stats[k] = train_stats[k].unsqueeze(0)
        array = scale_batch_wrapper(array, train_stats, data_columns, forecast=forecast)
        array = array.squeeze(0)
        for k in train_stats:
            train_stats[k] = train_stats[k].squeeze(0)
    if options.get("handle_degrees", False):
        if forecast:
            # ["temperatur_2m", "wind_direction_100m", "wind_direction_10m", "wind_speed_100m", "wind_speed_10m"]
            feature_indices = [index for index, value in enumerate(data_columns) if value in ["wind_direction_100m", "wind_direction_10m"]]
        else:
            # ["mean_wind_speed", "mean_wind_dir", "mean_temp", #"min_temp", "mean_relative_hum", #"max_wind_speed_10min", #"max_wind_speed_3sec", "mean_pressure", "mean_cloud_cover", "VAERDI"]
            feature_indices = [index for index, value in enumerate(data_columns) if value in ["mean_wind_dir"]]
        for feature_index in feature_indices:
            array, data_columns = degrees_to_sin_cos(array, feature_index, data_columns)
    return array, data_columns

def main():
    pass

if __name__ == "__main__":
    main()