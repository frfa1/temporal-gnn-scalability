import sys
sys.path.append("..")
import pandas as pd
import geopandas as gpd
from pandas import json_normalize
import os
import json
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

from shapely import Polygon, Point
import ast
import copy

from preprocessing.feature_processing import preprocess_factory

# Carbontracker
from carbontracker.tracker import CarbonTracker

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Timestamp Fast Mapping
hourly_time_dict = {
    '01:00': 0, '02:00': 1, '03:00': 2, '04:00': 3, '05:00': 4,
    '06:00': 5, '07:00': 6, '08:00': 7, '09:00': 8, '10:00': 9, '11:00': 10,
    '12:00': 11, '13:00': 12, '14:00': 13, '15:00': 14, '16:00': 15, '17:00': 16,
    '18:00': 17, '19:00': 18, '20:00': 19, '21:00': 20, '22:00': 21, '23:00': 22, '00:00': 23, 
}

class ENWPFDataset(Dataset):
    """
        Custom PyTorch dataloader class with data from 2018 and 2019. 
        Data has been initially preprocessed, such that each wind turbine in train is in test and vice versa.
        By default, setting the param flag to "train" loads year 2018, and "test" loads year 2019. 
    """
    def __init__(
            self,
            data_path = "/home/data_shares/energinet/energinet/frederik",
            master_filename = 'final_master.csv',
            subset = None, # [[X1, X2], [Y1, Y2]] Coordinate borders to subset wind turbines
            subset_turb = None, # Number of turbines to randomly subset - after coordinate subset
            subset_forecast = 3, # 1-3 of standard geo defined forecasts 
            parameters = ["mean_wind_speed", "mean_wind_dir"],
            input_seq_length = 336,
            output_seq_length = 24,
            flag = 'train',
            with_forecasts = True,
            options = {}, # Preprocessing options
            num_folds = 1, # For TS CV. Standard is 5 splits
        ):

        super().__init__()

        # initialization
        self.parameters = parameters
        self.data_path = data_path
        self.master_filename = master_filename
        self.subset = subset
        self.subset_turb = subset_turb
        self.subset_forecast = subset_forecast
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag] # Converts train/val/test to number - used to index into data
        self.flag = flag
        self.with_forecasts = with_forecasts
        self.options = options
        self.num_folds = num_folds

        self.__read_data__()

    def __read_data__(self):

        if self.flag == "train":
            df_data = pd.read_csv(
                os.path.join(self.data_path, "final_train.csv")
                )
            self.forecast_array = torch.load(
                os.path.join(self.data_path, "train_forecasts_subset_" + str(self.subset_forecast) + ".pt")
                ).numpy()
            #self.df_forecasts = pd.read_parquet(
            #    os.path.join(self.data_path, "train_forecasts.parquet")
            #    )
        else:
            df_data = pd.read_csv(
                os.path.join(self.data_path, "final_test.csv")
                )
            self.forecast_array = torch.load(
                os.path.join(self.data_path, "test_forecasts_subset_" + str(self.subset_forecast) + ".pt")
                ).numpy()
            #self.df_forecasts = pd.read_parquet(
            #    os.path.join(self.data_path, "test_forecasts.parquet")
            #    )

        # Load master data
        master_df = pd.read_csv(os.path.join(self.data_path, self.master_filename))

        # Convert GSRN columns to numeric
        master_df["GSRN"] = pd.to_numeric(master_df["GSRN"])
        df_data["GSRN"] = pd.to_numeric(df_data["GSRN"])
 
        # Subset wind turbines by coordinates ranges
        if self.subset:
            utm_x, utm_y = self.subset[0], self.subset[1]
            master_df = master_df[
                (master_df["UTM_x"] > utm_x[0]) & (master_df["UTM_x"] < utm_x[1]) & 
                (master_df["UTM_y"] > utm_y[0]) & (master_df["UTM_y"] < utm_y[1])]
            print("Length of df_data before check", len(df_data))
            df_data = df_data[df_data["GSRN"].isin(master_df["GSRN"])]
            print("Length of df_data after check", len(df_data))
        master_df = master_df[master_df["GSRN"].isin(df_data["GSRN"].unique())]
        # Randomly subset X amount of turbines
        if self.subset_turb:
            master_df = master_df.sample(n=self.subset_turb)
            df_data = df_data[df_data["GSRN"].isin(master_df["GSRN"])]
        self.master_df = master_df

        # Sort timedata and master data
        df_data.sort_values(by=["TIME_CET", "GSRN"], inplace=True)
        self.master_df.sort_values(by=["GSRN"], inplace=True)

        # Turbine counts, time series length
        self.turbine_list = df_data["GSRN"].unique()
        self.n_turbines = len(self.turbine_list)
        self.total_size = int(len(df_data) / self.n_turbines)

        # One-hot encode turbine IDs
        #one_hot = pd.get_dummies(df_data["GSRN"])
        #one_hot_columns = list(one_hot.columns)
        #one_hot = one_hot.values

        # Get integer IDs of turbines for embedding encoding
        #df_data["int_ID"] = pd.factorize(df_data["GSRN"])[0]

        self.data_columns_forecast = ["temperatur_2m", "wind_direction_100m", "wind_direction_10m", "wind_speed_100m", "wind_speed_10m"]
        self.data_columns = self.parameters.copy()
        self.data_columns.append("VAERDI")
        #self.data_columns = ["int_ID"] + self.data_columns # Prepend int ID
        #self.data_columns.append("GSRN") # Bug finding
        self.array_data = df_data[self.data_columns].values # Get numpy array of data
        
        # Add one-hot
        #self.array_data = np.append(one_hot, self.array_data, axis=1)
        #self.data_columns = one_hot_columns + self.data_columns

        self.array_data = np.reshape(self.array_data, [self.total_size, self.n_turbines, len(self.data_columns)]) # (total length, n turbines, n features)
        self.array_data = torch.from_numpy(self.array_data) #.to(device) # Convert to tensor
        self.set_length()

        # TESTING
        #self.array_data[:] = 100

        # Replace "empty" values of forecast with nan for better statistics
        #forecast_mask = (self.forecast_array >= -100000) & (self.forecast_array <= 100000)
        #self.forecast_array = np.where(forecast_mask, self.forecast_array, np.nan)

        # Data stats for all variables, for all turbines. For test, this should be feed in the "options" dictionary for preprocessing
        if self.flag == "train":
            self.array_data, self.forecast_array = self.array_data[:len(self)], self.forecast_array[:len(self)]
            self.means, self.stds = np.nanmean(self.array_data, axis=0, keepdims=True), np.nanstd(self.array_data, axis=0, keepdims=True) 
            self.mins, self.maxs = np.nanmin(self.array_data, axis=0, keepdims=True), np.nanmax(self.array_data, axis=0, keepdims=True)
            self.means_forecast, self.stds_forecast = np.nanmean(self.forecast_array, axis=(0,2)), np.nanstd(self.forecast_array, axis=(0,2)) 
            self.mins_forecast, self.maxs_forecast = np.nanmin(self.forecast_array, axis=(0,2)), np.nanmax(self.forecast_array, axis=(0,2))
            #self.means_forecast, self.stds_forecast = train_forecast.mean(dim=(0,2)), train_forecast.std(dim=(0,2)) # (1, N, M)
            #self.mins_forecast, self.maxs_forecast = train_forecast.min(dim=(0,2)).values, train_forecast.max(dim=(0,2)).values
            self.options["train_stats"] = {
                "means": torch.from_numpy(self.means),
                "stds": torch.from_numpy(self.stds),
                "mins": torch.from_numpy(self.mins),
                "maxs": torch.from_numpy(self.maxs)
            }
            self.options["train_stats_forecast"] = {
                "means": torch.from_numpy(self.means_forecast),
                "stds": torch.from_numpy(self.stds_forecast),
                "mins": torch.from_numpy(self.mins_forecast),
                "maxs": torch.from_numpy(self.maxs_forecast)
            }
            for k in self.options["train_stats_forecast"]:
                self.options["train_stats_forecast"][k] = self.options["train_stats_forecast"][k].unsqueeze(0)

        # Reshape weather forecasts to allow same preprocess as historical variables
        self.forecast_array = torch.from_numpy(self.forecast_array)
        self.forecast_array = self.forecast_array.permute(0, 2, 1, 3) # (T, N, O, M) -> (T, O, N, M)
        T, O, N, M = self.forecast_array.shape
        self.forecast_array = self.forecast_array.reshape(T*O, N, M) 

        # Applies preprocessing based on options dictionary on whole array. If empty dict, no preprocessing is done.
        self.array_data, self.data_columns = preprocess_factory(self.array_data, self.options, self.data_columns, self.options["train_stats"])
        self.forecast_array, self.data_columns_forecast = preprocess_factory(self.forecast_array, self.options, self.data_columns_forecast, self.options["train_stats_forecast"], forecast=True)

        # Shape forecasts back to original
        self.forecast_array = self.forecast_array.reshape(T, O, N, -1) # Feature dim changed after preprocessing
        #self.forecast_array = self.forecast_array.permute(0, 2, 1, 3) # (T, N, O, M)

        if self.flag == "train":
            self.array_data, self.forecast_array = self.array_data[:len(self)], self.forecast_array[:len(self)]
            self.means, self.stds = np.nanmean(self.array_data, axis=0, keepdims=True), np.nanstd(self.array_data, axis=0, keepdims=True) 
            self.mins, self.maxs = np.nanmin(self.array_data, axis=0, keepdims=True), np.nanmax(self.array_data, axis=0, keepdims=True)
            self.means_forecast, self.stds_forecast = np.nanmean(self.forecast_array, axis=(0,1)), np.nanstd(self.forecast_array, axis=(0,1)) 
            self.mins_forecast, self.maxs_forecast = np.nanmin(self.forecast_array, axis=(0,1)), np.nanmax(self.forecast_array, axis=(0,1))

        # Bug fixing
        #sequence_remainder = self.array_data % self.input_seq_length
        #self.array_data = self.array_data[:-23,:,:]

        self.df_data = df_data # Dataframe without preprocessing


        print("Data shape:", self.array_data.shape)
        print("Forecast shape:", self.forecast_array.shape)

    def split_preprocess(self, train_indices, val_indices, optionsFold):
        """
            Callable method that makes/updates an array in the class for current train-val split of the data
            based on pairwise (train,val) indices of the data. Includes pairwise preprocesing.
            Input:
                train_indices, val_indices: Lists of indices the the split
                options: Preprocessing steps to fit on train and apply on both sets
        """
        train_array_new = self.old_array[train_indices,:,:].copy()
        val_array_new = self.old_array[val_indices,:,:].copy()
        train_array, val_array, self.data_columns = preprocess_factory(train_array_new, val_array_new, optionsFold, self.data_columns) # Preprocessing on this split only
        train_array, val_array = torch.from_numpy(train_array).to(device), torch.from_numpy(val_array).to(device)
        self.array_data[train_indices,:,:] = train_array.copy()
        self.array_data[val_indices,:,:] = val_array.copy()
        #self.array_data = torch.from_numpy(self.array_data).to(device) # To gpu tensor

    def set_length(self):
        self.effective_length = self.array_data.shape[0] - self.input_seq_length - self.output_seq_length + 1

    def update_input_seq_length(self, new_input_seq_length):
        """ Setter method for changing input sequence length without reloading data """
        self.input_seq_length = new_input_seq_length
        self.set_length()

    def update_output_seq_length(self, new_output_seq_length):
        """ Setter method for changing input sequence length without reloading data """
        self.output_seq_length = new_output_seq_length
        self.set_length()

    def __len__(self):
        return self.effective_length

    def get_raw_df(self):
        return self.raw_df

    def __getitem__(self, idx):
        # Sliding window with the size of input_seq_length + output_seq_length
        # (total length, n turbines, n features)
        s_begin = idx
        s_end = s_begin + self.input_seq_length
        r_begin = s_end
        r_end = r_begin + self.output_seq_length
        seq_x = self.array_data[s_begin:s_end, :, :] # (T, N, M)
        seq_y = self.array_data[r_begin:r_end, :, :]
        seq_forecast = self.forecast_array[r_begin, :, :, :] # (T, O, N, M) -> (O, N, M)

        return seq_x.float(), seq_y.float(), seq_forecast.float()
        #return seq_x.astype('float32'), seq_y.astype('float32')


def main():
    print("Loading data")
    parameters = [
        "mean_wind_speed",
        "mean_wind_dir",
        "mean_temp",
        "min_temp",
        "mean_relative_hum",
        "max_wind_speed_10min",
        "max_wind_speed_3sec",
        "mean_pressure",
        "mean_cloud_cover",
        ]
    subset = [
        [530000, 601000], # X
        [6310000, 6410000] # Y
        ]
    subset_turb = 10
    options = {
        "standardize": True,
        "handle_outliers": {
                "n_neighbors": 20,
                "contamination": 0.1,
                "interpolation": "KNN",
                "knn_n": 2
            },
        "handle_degrees": True
    }
    train_data = ENWPFDataset(parameters=parameters, flag='train', subset=subset, subset_turb=subset_turb, options=options)

    # Carbon tracker
    """tracker = CarbonTracker(epochs=1, epochs_before_pred=-1, monitor_epochs=-1, verbose=3)
    for epoch in range(1):
        tracker.epoch_start()
        train_data = ENWPFDataset(data_path = folder, weather_path = weather_path, capacity=1000, data_years = [2018, 2019], flag='train', subset=[[530000, 601000], [6310000, 6410000]])
        tracker.epoch_end()
    tracker.stop()"""

    #print(train_data.weather_data)

    # Pytorch
    """train_data_loader = DataLoader(
        train_data,
        batch_size=1000, #config.batch_size
        shuffle=True,
        drop_last=True,
        num_workers=2) # config.num_workers"""

if __name__ == "__main__":
    main()

#master_df = pd.read_parquet(folder + "masterdatawind.parquet")
#settlement_df = pd.read_parquet(folder + "settlement/" + "2019.parquet")
#settlement2018_df = pd.read_parquet(folder + "settlement/" + "2018.parquet")

"""settlement_df = settlement_df[:100000]

joined_df = master_df.merge(settlement_df, how="inner", on="GSRN")

print(settlement_df)
print(master_df)

print(joined_df)
print(joined_df.columns)"""

"""joined_df = master_df.merge(settlement_df, how="inner", on="GSRN")

print(joined_df)
print(joined_df.columns)"""