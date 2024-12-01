import sys
sys.path.append("..")
from loader.wpf_data import ENWPFDataset
from preprocessing.feature_processing import local_outlier_factor, handle_outliers
from evaluation.carbon_measure import get_carbon

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import contextily as cx
import numpy as np
import shapely
from shapely.geometry import Polygon, LineString, Point

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.colors as colors
from matplotlib import cm

# Interpolations
#from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from sklearn.impute import KNNImputer

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

save_path = "../figs/"

def plot_map_simple(master_gdf, title, markersize, saveas, figsize=(16, 10)):#, zoom=None: list):
    turbine_type_mapping = {
        'M': 'Turbine in a park',
        'W': 'Single turbine',
        'H': 'Household turbine',
        'P': 'Turbine park',
    }
    master_gdf['Turbine_type'] = master_gdf['Turbine_type'].map(turbine_type_mapping)
    master_gdf['Turbine_type'] = master_gdf['Turbine_type'].astype('category')
    category_colors = {
        'Turbine in a park': 'olivedrab',
        'Single turbine': 'purple',
        'Household turbine': 'red',
        'Turbine park': 'royalblue',
    }
    colors = master_gdf['Turbine_type'].map(category_colors)
    # Create a color map for the categorical variable
    fig, ax = plt.subplots(figsize=figsize)
    # Plot the GeoDataFrame, coloring by 'Turbine_type'
    master_gdf[["geometry", "Turbine_type"]].plot(
        ax=ax,
        #column='Turbine_type',
        #categorical=True,
        legend=True,
        color=colors,
        #legend_kwds={'title': 'Turbine Type'},  # Add legend title
        edgecolor='k',  # Edge color
        alpha=0.5,
        markersize=markersize, 
        linewidth=0.1
    )
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=key, markersize=10, markerfacecolor=color, markeredgewidth=0.5, markeredgecolor='k') 
            for key, color in category_colors.items()]
    ax.legend(handles=handles, title='Turbine Type')
    cx.add_basemap(ax, crs=master_gdf.crs, source=cx.providers.CartoDB.Positron)
    #cbar = plt.colorbar(ax.get_children()[0], ax=ax, orientation='vertical', pad=0.02)
    #cbar.set_label("Capacity kW", fontsize=24)
    # Set font size for axes labels
    ax.set_xlabel("X", fontsize=20)
    ax.set_ylabel("Y", fontsize=20)
    plt.tight_layout()
    # Save fig
    plt.savefig(save_path + saveas)
    plt.close()

"""master_df = pd.read_parquet("/home/data_shares/energinet/energinet/masterdatawind.parquet")
master_df = master_df[
    (master_df["UTM_x"] > 400000) & (master_df["UTM_x"] < 1200000) & 
    (master_df["UTM_y"] > 5900000) & (master_df["UTM_y"] < 8000000)]
master_gdf = gpd.GeoDataFrame(master_df, geometry=gpd.points_from_xy(master_df['UTM_x'], master_df['UTM_y']), crs="EPSG:25832")
plot_map_simple(master_gdf, title="All turbines", markersize=10, saveas="all_turbines_map.png", figsize=(16, 10))"""

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import geopandas as gpd
import contextily as cx
import numpy as np

def plot_map_three(master_gdf_list, titles, markersize, saveas, figsize=(24, 8)):
    """
    Plots three GeoDataFrames side by side with a shared colorbar.

    Parameters:
        master_gdf_list (list): List of three GeoDataFrames to plot.
        titles (list): List of titles for each subplot.
        markersize (float): Marker size for points.
        saveas (str): File name to save the plot.
        figsize (tuple): Size of the overall figure.
    """
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)

    # Get the shared colormap and normalization
    all_capacities = np.concatenate([gdf["Capacity_kw"] for gdf in master_gdf_list])
    vmin, vmax = all_capacities.min(), all_capacities.max()
    cmap = 'viridis'
    
    for idx, (gdf, title, ax) in enumerate(zip(master_gdf_list, titles, axes)):
        gdf.plot(
            ax=ax,
            column="Capacity_kw",
            cmap=cmap,
            edgecolor='k',
            alpha=0.5,
            markersize=markersize,
            vmin=vmin,
            vmax=vmax,
            legend=False  # Disable individual legends
        )
        cx.add_basemap(ax, crs=gdf.crs, source=cx.providers.CartoDB.Positron)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)

    # Add a shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label("Capacity kW", fontsize=14)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path + saveas)
    plt.close()


def plot_map(master_gdf, title, markersize, saveas, figsize=(16, 10), graph=False, train_data=None, squares=[]):#, zoom=None: list):
    if graph != True:
        vmin = master_gdf["Capacity_kw"].min()
        vmax = master_gdf["Capacity_kw"].max()
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = 'viridis'

        ax = master_gdf[["geometry", "Capacity_kw"]].plot(
            figsize=figsize,
            edgecolor='k', #edgecolor='none',
            aspect=1,
            c=master_gdf["Capacity_kw"],
            cmap=cmap,  # Choose a colormap (e.g., 'viridis', 'plasma', 'coolwarm', etc.)
            legend=False, #True,
            #legend_kwds={'label': "Capacity_kw"},  # Add legend label
            #vmin=0, #master_gdf["Capacity_kw"].min(),  # Set the minimum value for the colormap
            #vmax=3600, #master_gdf["Capacity_kw"].max(),  # Set the maximum value for the colormap,
            alpha = 0.5,
            markersize=markersize,
            zorder=2
        )
        # Explicitly create a ScalarMappable and add a colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Dummy data for ScalarMappable
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, fraction=0.05, shrink=1.0)
        cbar.set_label("Capacity kW", fontsize=24)
        #cbar = plt.colorbar(ax.get_children()[0], ax=ax, orientation='vertical', pad=0.02)
        #cbar.set_label("Capacity kW", fontsize=24)

        for square in squares:
            x_min, x_max = square[0][0], square[0][1]
            y_min, y_max = square[1][0], square[1][1]
            square = Polygon([
                (x_min, y_min),  # Bottom-left
                (x_max, y_min),  # Bottom-right
                (x_max, y_max),  # Top-right
                (x_min, y_max),  # Top-left
                (x_min, y_min)   # Close the loop
            ])
            square_gdf = gpd.GeoDataFrame({'geometry': [square]}, crs="EPSG:25832")
            square_gdf.boundary.plot(ax=ax, alpha=1, edgecolor="red", linewidth=1)

    elif graph == True:
        ax = master_gdf["geometry"].plot(
            figsize=figsize,
            color='white',   # Set points to white
            edgecolor='k',   # Black border for the points
            aspect=1,
            markersize=markersize,
            zorder=2  # Points are on top of the lines
        )
        from loader.wpfTemporalGraph import ENWPFDatasetWrapper
        import networkx as nx
        loader = ENWPFDatasetWrapper(distance_method="exponential_decay_weight", threshold=10000)
        dataset = loader.get_dataset(wpf_object=train_data)

        gdf_lines = gpd.GeoDataFrame(
            [(line, weight) for line, weight in loader.line_strings], 
            columns=['geometry', 'weight']
        )
        # Normalize edge weights for line widths
        min_width = 0.5
        max_width = 5
        edge_weights_list = np.array(list(gdf_lines["weight"]))
        min_weight = edge_weights_list.min()
        max_weight = edge_weights_list.max()
        def normalize(weight):
            return min_width + (weight - min_weight) * (max_width - min_width) / (max_weight - min_weight)
        gdf_lines['linewidth'] = gdf_lines['weight'].apply(normalize)
        norm = Normalize(vmin=gdf_lines['weight'].min(), vmax=gdf_lines['weight'].max())
        cmap = plt.cm.magma  # Choose a colormap for the lines
        #gdf_lines['line_color'] = [cmap(norm(weight)) for weight in gdf_lines['weight']]
        gdf_lines['line_color'] = [cmap(weight) for weight in gdf_lines['weight']]
        gdf_lines.plot(ax=ax, color=gdf_lines['line_color'], linewidth=gdf_lines['weight'], zorder=3)
        #for idx, row in gdf_lines.iterrows():
        #    gdf_lines.iloc[[idx]].plot(ax=ax, color='blue', linewidth=normalize(row['weight'])) #, zorder=1)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Trick to add the colorbar
        cbar_lines = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
        cbar_lines.set_label("Line Weight", fontsize=24)

    # Map
    cx.add_basemap(ax, crs=master_gdf.crs, source=cx.providers.CartoDB.Positron)

    # Set font size for axes labels
    #ax.set_xlabel("X", fontsize=24)
    #ax.set_ylabel("Y", fontsize=24)

    plt.tight_layout()

    # Save fig
    plt.savefig(save_path + saveas)


    plt.close()

def plot_wind_grids(t, saveas="wind_grids.png"):
    """
        Input t: Weather geopandas dataframe
    """
    fig = plt.figure(figsize=(18, 14))

    ax = t["geometry"].boundary.plot(
        edgecolor="k", aspect=1, linewidth=0.5, figsize=(18, 14),
        #vmin=t["mean_wind_speed"].min(), vmax=t["mean_wind_speed"].max(), # Color map
    )

    V = np.array(t["mean_wind_speed"])
    cmap = plt.cm.Blues
    cNorm  = colors.Normalize(vmin=np.min(V), vmax=np.max(V))
    scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cmap)

    #im = ax.imshow(V, cmap=cmap, norm=cNorm)

    color_list = []        
    for idx in range(0,len(V)):
            colorVal = scalarMap.to_rgba(V[idx])
            r,g,b,a = colorVal
            color_list.append((r,g,b,a))

    idx_count = 0
    for index, row in t.iterrows():
        polygon, angle = row["geometry"], row["mean_wind_dir"]
        centroid = polygon.centroid
        x, y = centroid.x, centroid.y
        dx = np.cos(np.deg2rad(-angle + 90))
        dy = np.sin(np.deg2rad(-angle + 90))
        arrow_length = 7500
        half_arrow_length = arrow_length / 2
        ax.arrow(
            x - dx * half_arrow_length, y - dy * half_arrow_length, # From point
            #dx * half_arrow_length, dy * half_arrow_length, # to point
            dx * half_arrow_length, dy  * half_arrow_length,
            fc=color_list[idx_count], ec=color_list[idx_count], head_width=3000 # Design
            )
        idx_count += 1

        #plt.text(x, y, str(angle), fontsize=5)

    cx.add_basemap(ax, crs=t.crs, source=cx.providers.CartoDB.Positron)

    #cb1 = mpl.colorbar.ColorbarBase(axc, cmap=cmap, norm=cNorm,orientation='vertical')
    cbar = fig.colorbar(mappable=scalarMap, ax=ax)
    cbar.set_label("Hourly mean wind speed (m/s)", fontsize=20)

    ax.set_xlabel("UTM-32 X coordinates", fontsize=18)
    ax.set_ylabel("UTM-32 Y coordinates", fontsize=18)

    # Save fig
    plt.savefig(save_path + saveas)

    # Close
    plt.close()

def scatter_plot(x, y, outlier_scores, interpolation="Raw", capacity=None, saveas="scatter_plot.png"):
    print("Making scatterplot")
    # Creating the scatter plot
    fig, ax = plt.subplots(figsize=(18, 14))
    if interpolation != "Raw":
        outlier_color = "green"
        outlier_label = "Imputed outliers"
    else:
        outlier_color = "red"
        outlier_label = "Outliers"

    inlier_idx = [idx for idx, _ in enumerate(outlier_scores) if outlier_scores[idx] > 0]
    outlier_idx = [idx for idx, _ in enumerate(outlier_scores) if outlier_scores[idx] <= 0]
    inlier_x, inlier_y = x[inlier_idx], y[inlier_idx]
    outlier_x, outlier_y = x[outlier_idx], y[outlier_idx]
    
    ax.scatter(inlier_x, inlier_y, s=10, c="blue", alpha=0.5, label="Inliers") # Inlier plot
    ax.scatter(outlier_x, outlier_y, s=10, c=outlier_color, alpha=1, label=outlier_label) # Outlier plot

    ax.legend(fontsize=18, loc="lower right")

    #plt.title('Wind power relative to wind speed of a single turbine')  # Set the title of the plot
    plt.xlabel('Wind Speed (m/s)', fontsize=18)  # Set the label for the x-axis
    plt.ylabel('Wind Power (kWh)', fontsize=18)  # Set the label for the y-axis
    plt.grid(True)  # Add a grid if needed

    if capacity:
        plt.axhline(y=capacity, color='chocolate', linestyle='-', label="Capacity")
        plt.text(0, capacity + 5, 'Capacity', color='chocolate', fontsize=18)
    # Saving the figure as a .png file
    plt.savefig(save_path + saveas)
    plt.close() # Close

def plot_bar_means_with_std(data_lists, labels, saveas=None):
    """
    Plot a bar chart where each bar represents the mean of a respective list 
    and the error bars represent the standard deviation.

    Args:
    - data_lists (list of lists): A list of lists where each inner list contains numerical values.
    - labels (list of strings): A list of names/labels for each bar.
    - saveas (str, optional): Filename to save the figure. If None, the figure will not be saved.
    """
    
    # Calculate the means and standard deviations for the data lists
    means = [np.mean(data) for data in data_lists]
    std_devs = [np.std(data) for data in data_lists]
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the bar chart
    ax.bar(labels, means, yerr=std_devs, capsize=5, color='skyblue') #, edgecolor='black')
    
    # Add labels and title
    ax.set_ylabel('Mean Carbon Emission (co2eq (g)', fontsize=14)
    #ax.set_title('Mean of Lists with Standard Deviation', fontsize=16)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if a filename is provided
    if saveas:
        plt.savefig(save_path + saveas)

def plot_time_power(array_data, start_time=None, saveas="timely_plot.png"):
    # array_data: (T, N, M)
    last_feature_data = array_data[:, :, -1] # Power
    summed_values = last_feature_data.sum(axis=1)
    T = array_data.shape[0]

    if start_time:
        time_steps = pd.date_range(start=start_time, periods=T, freq='H')
    else:
        time_steps = np.arange(T)

    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, summed_values, linestyle='-')
    #plt.title('Time Series of Summed Last Feature Across Turbines')
    plt.xlabel('Time')
    plt.ylabel('Summed Power [kWh]')
    plt.grid(True)
    plt.savefig(save_path + saveas)
    plt.close

def viz_edge_weights(saveas="edge_weight_graph.png"):
    # Original function setup
    distance = np.linspace(0, 2000, 1000)
    edge_weight = np.exp(-0.001 * distance)
    #edge_weight[distance > 500] = 0  # Apply the cutoff at x = 500

    # Sample data: Three lists of edge weights
    # Replace these lists with your actual data
    edge_weights_subset1 = np.array([0.9, 0.85, 0.8])
    edge_weights_subset2 = np.array([0.7, 0.65, 0.6])
    edge_weights_subset3 = np.array([0.5, 0.45, 0.4])

    # Infer the distances from the edge weights
    def infer_distance(edge_weights):
        # Avoid log of zero or negative numbers
        edge_weights = np.clip(edge_weights, 1e-10, None)
        distances = -1000 * np.log(edge_weights)
        # Apply cutoff: distances beyond 500 are set to NaN
        distances[distances > 500] = np.nan
        return distances

    distances_subset1 = infer_distance(edge_weights_subset1)
    distances_subset2 = infer_distance(edge_weights_subset2)
    distances_subset3 = infer_distance(edge_weights_subset3)

    # Plot the original function
    plt.plot(distance, edge_weight, label=r'$e^{-0.001 \times \text{Distance}}$')

    # Plot the points from the subsets
    #plt.scatter(distances_subset1, edge_weights_subset1, color='blue', label='Subset 1')
    #plt.scatter(distances_subset2, edge_weights_subset2, color='green', label='Subset 2')
    #plt.scatter(distances_subset3, edge_weights_subset3, color='orange', label='Subset 3')

    # Add the dashed cutoff line at x = 500
    plt.axvline(x=500, color='red', linestyle='--', label='Cutoff at x=500')

    # Label the axes
    plt.xlabel('Distance')
    plt.ylabel('Edge weight')
    plt.ylim(bottom=0)

    # Add a legend
    plt.legend()

    plt.savefig(save_path + saveas)

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
    geo_subset1 = [
        [566000, 572000], # X
        [6330000, 6338000] # Y
        ] # ~15 turbines
    geo_subset2 = [
        [560000, 580000], # X
        [6310000, 6340000] # Y
        ] # ~67 turbines
    geo_subset3 = [
        [530000, 601000], # X
        [6310000, 6410000] # Y
        ] # 348 turbines

    subset_forecast = int(subset)
    if subset == "0":
        geo_subset = None
        subset_forecast = 3
    elif subset == "1":
        geo_subset = geo_subset1
    elif subset == "2":
        geo_subset = geo_subset2
    elif subset == "3":
        geo_subset = geo_subset3
    options_train = { # Preprocessing for train
        "remove_nan": True,
        "handle_degrees": True
    }
    train_data = ENWPFDataset(parameters=parameters, flag='train', subset=geo_subset, subset_forecast=subset_forecast, options=options_train)
    print("Loaded train")
    return train_data

def main():

    train_data = load_data(subset="0")
    master_df = train_data.master_df
    try:
        master_df['geometry'] = master_df['geometry'].apply(lambda x: shapely.wkt.loads(x))
    except:
        pass
    master_gdf = gpd.GeoDataFrame(data=master_df, geometry=master_df['geometry'], crs="EPSG:25832")
    print(master_gdf["Capacity_kw"].describe())
    print(master_gdf["Capacity_kw"].unique())

    geo_subset1 = [
        [566000, 572000], # X
        [6330000, 6338000] # Y
        ] # ~15 turbines
    geo_subset2 = [
        [560000, 580000], # X
        [6310000, 6340000] # Y
        ] # ~67 turbines
    squares = [geo_subset1, geo_subset2]

    plot_map(
        master_gdf,
        markersize=40,
        saveas="map.png",
        title="Wind Turbines in Denmark",
        figsize=(16,12),
        #squares=squares,
        )
    """plot_map_three(
        [master_gdf, master_gdf, master_gdf],
        titles=["Subset 1", "Subset 2", "Subset 3"],
        markersize=5,
        saveas="three_subsets_map.png", 
        figsize=(24, 8)
        )"""

    #viz_edge_weights()

    """hyp_dir = "../train_evaluate/out/carbontracker/"
    all_carbon_lists = get_carbon(hyp_dir)
    MODELS = [
        "A3TGCNModel", "ChebGRU", "GCNConv", "GCNConv_Edge", "GRU"
    ]
    plot_bar_means_with_std(all_carbon_lists, MODELS, saveas="carbon_bars_hyp")"""

    """print("Running viz")
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
    data_years = [2018, 2019]
    #geo_subset = [
    #    [566000, 572000], # X
    #    [6330000, 6338000] # Y
    #    ] # ~15 turbines
    geo_subset = [
        [560000, 580000], # X
        [6310000, 6340000] # Y
        ] # ~67 turbines
    #geo_subset  = [
    #    [530000, 601000], # X
    #    [6310000, 6410000] # Y
    #    ] # 348 turbines
    subset_turb = None
    # Processing options
    options = {
        "handle_degrees": True
    }
    train_data = ENWPFDataset(parameters=parameters, flag='train', subset=geo_subset, subset_turb=subset_turb, options=options)

    df = train_data.df_data
    array = train_data.array_data.cpu().numpy() # shape: (T, N, M)
    data_columns = train_data.data_columns

    master_df = train_data.master_df
    try:
        master_df['geometry'] = master_df['geometry'].apply(lambda x: shapely.wkt.loads(x))
    except:
        pass
    gdf = gpd.GeoDataFrame(data=master_df, geometry=master_df['geometry'], crs="EPSG:25832")

    plot_map(gdf, markersize=80, title="Wind Turbines in Denmark", saveas="map_TEST.png", figsize=(16,14), graph=True, train_data=train_data)"""

    ### Scatter Plot: WS / WP
    """turbine_idx = 0 # Select turbine to visualize
    x_idx = data_columns.index("mean_wind_speed")
    y_idx = data_columns.index("VAERDI")
    #sizes_idx = data_columns.index("max_wind_speed_3sec")
    feature_indices = [index for index, value in enumerate(data_columns) if value in ["mean_wind_speed", "VAERDI"]]

    # Detect outliers
    turbine_lofs = local_outlier_factor(array, feature_indices, n_neighbors=20, contamination="auto")[turbine_idx]
    scatter_plot(
        array[:,0,x_idx], array[:,0,y_idx], # x and y
        outlier_scores=turbine_lofs,
        capacity=df.iloc[turbine_idx]["Capacity_kw"],
        saveas="scatter_plot_final" + ".png"
        )"""
    #x, y = array[:,turbine_idx,x_idx], array[:,turbine_idx,y_idx] # x and y of (T, N, M)
    #sizes = array[:,turbine_idx,sizes_idx] # Sizes (based on max wind speed)

    """params = {
        "n_neighbors": 20,
        "contamination": 0.1,
        "interpolation": "KNN",
        "knn_n": 2
    }
    interpolations = ["Raw", "Cubic", "KNN"]
    for interp in interpolations:
        params["interpolation"] = interp
        updated_array, list_of_lofs = handle_outliers(array, feature_indices, params=params, select_turbine=turbine_idx) 
        turbine_lofs = list_of_lofs[turbine_idx]
        x, y = updated_array[:,turbine_idx,x_idx], updated_array[:,turbine_idx,y_idx] # x and y of (T, N, M)
        scatter_plot(
            x, y,
            outlier_scores=turbine_lofs,
            interpolation=params["interpolation"],
            capacity=df.iloc[turbine_idx]["Capacity_kw"],
            saveas="scatter_plot_" + interp + ".png"
            )"""

    """count = 1
    for n in [10, 20, 30]:
        for cont in ["auto", 0.05, 0.1]:
            turbine_lofs = local_outlier_factor(array, feature_indices, n_neighbors=n, contamination=cont)[turbine_idx]
            scatter_plot(
                array[:,0,x_idx], array[:,0,y_idx], # x and y
                outlier_scores=turbine_lofs,
                capacity=df.iloc[turbine_idx]["Capacity_kw"],
                saveas="scatter_plot_" + str(count) + ".png"
                )
            count += 1"""


    """climate_gdf = train_data.load_weather_data()
    climate_gdf = climate_gdf[climate_gdf["to"] == climate_gdf["to"].iloc[1]]
    plot_wind_grids(climate_gdf)"""

    """master_df = pd.read_parquet("/home/data_shares/energinet/energinet/masterdatawind.parquet")
    master_df = master_df[
        (master_df["UTM_x"] > 400000) & (master_df["UTM_x"] < 1200000) & 
        (master_df["UTM_y"] > 5900000) & (master_df["UTM_y"] < 8000000)]
    master_gdf = gpd.GeoDataFrame(master_df, geometry=gpd.points_from_xy(master_df['UTM_x'], master_df['UTM_y']), crs="EPSG:25832")
    plot_map_simple(master_gdf, title="All turbines", markersize=10, saveas="all_turbines_map.png", figsize=(16, 10))"""

if __name__ == "__main__":
    main()


