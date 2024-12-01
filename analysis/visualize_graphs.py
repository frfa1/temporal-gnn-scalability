import sys
sys.path.append("..")
from loader.wpf_data import ENWPFDataset
from torch_geometric_temporal.signal import temporal_signal_split
from wpfTemporalGraph import ENWPFDatasetWrapper

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import contextily as cx
import numpy as np
import networkx as nx

from matplotlib.colors import Normalize
import matplotlib.colors as colors
from matplotlib import cm

# Interpolations
#from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from sklearn.impute import KNNImputer

save_path = "../figs/"

def plot_graph():


def main():
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
    subset = [
        [530000, 601000], # X
        [6310000, 6410000] # Y
        ]
    options = {
        "handle_degrees": True
    }
    train_data = ENWPFDataset(parameters=parameters, flag='train', subset=subset, options=options)
    loader = ENWPFDatasetWrapper(distance_method="exponential_decay_weight", threshold=500)
    dataset = loader.get_dataset(wpf_object=train_data)
    for time, snapshot in enumerate(dataset):
        edge_index = snapshot.edge_index
        edge_weight = snapshot.edge_attr
        break
    plot_graph(edge_index)

if __name__ == "__main__":
    main()