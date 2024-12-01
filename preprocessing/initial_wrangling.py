import sys
import os
import json
import pandas as pd
import geopandas as gpd
from pandas import json_normalize
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
sys.path.append("..")
from loader.wpf_data import ENWPFDataset

import shapely
from shapely.geometry import Polygon, LineString, Point
import ast

import threading
from carbontracker.tracker import CarbonTracker

month_dict = {
    1: "january", 2: "february", 3: "march", 4: "april", 5: "may", 6: "june",
    7: "july", 8: "august", 9: "september", 10: "october", 11: "november", 12: "december"
}
all_months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]

def weather_to_csv(weather_path, data_years=[2019], month=1):

    """ 
        Method to load and convert initial weather JSON files to DataFrames - and save as monthly csv files

        Arguments:
            weather_path: weather path
            data_years: List of years to consider
            month: Integer, for multithreading. The function will only work with files of given month 
            parameters: parameters of weather to keep

        No return
    """

    # Make a list of all years climate data
    climate_list = []
    for year in data_years:
        print(year)
        year_path = weather_path + str(year) + "grid"
            
        month_dfs = []

        # TIMEIT
        #import time

        for idx, weather_file in enumerate(os.listdir(year_path)):

            # TIMEIT
            #start_time = time.time()

            if weather_file.split(".")[-1] != "txt": # If file is not a txt file (like preprocess month csv files), then skip
                continue

            file_month = int(weather_file.split("-")[1]) 
            if file_month != month:
                continue

            full_path = os.path.join(year_path, weather_file)
            print(full_path)
            df = pd.read_json(full_path, lines=True)
            df_result = pd.concat([df, pd.json_normalize(df["properties"]), pd.json_normalize(df["geometry"])], axis=1).drop(["properties", "geometry", "type", "qcStatus", "calculatedAt"], axis=1)
            print(df_result.columns)
            month_dfs.append(df_result)
            
            # TIMEIT
            #end_time = time.time()
            #iteration_time = end_time - start_time
            #print(f"Iteration {idx+1} took {iteration_time} seconds")

        month_df = pd.concat(month_dfs, ignore_index=True)
        month_df.to_csv(os.path.join(year_path, str(month_dict[int(month)]) + ".csv"), index=False)   # Save

def process_weather_data(weather_path, data_years=[2018,2019], parameters = ["mean_wind_speed", "mean_wind_dir"]):
    """
        Saves full climate geopandas dataframe for the given list of years
    """
    # Converts string dictionary of geometry to polygon
    def string_to_polygon(geom_string):
        coords = ast.literal_eval(geom_string)
        return Polygon(coords[0])

    for year in data_years:
        list_of_climate_dfs = []
        print(year)
        year_path = weather_path + str(year) + "grid"
        for month in all_months:
            full_path = os.path.join(year_path, month + ".csv")
            if full_path.split("/")[-1] in os.listdir(year_path):
                list_of_climate_dfs.append(
                    pd.read_csv(full_path)
                )
        climate_df = pd.concat(list_of_climate_dfs, ignore_index=True)
        climate_df = climate_df[(climate_df["parameterId"].isin(parameters))]
        climate_df = climate_df[climate_df["timeResolution"] == "hour"]

        # Convert parameterIds to unique dataframes with corresponding values
        parameter_dfs = []
        for parameter in parameters:
            print("Going through parameter:", parameter)
            parameter_df = climate_df[climate_df["parameterId"] == parameter]
            #parameter_df.loc[:,parameter] = parameter_df["value"]
            parameter_df.loc[:, parameter] = parameter_df.loc[:, "value"]
            parameter_df.drop(["value"], axis=1, inplace=True)
            parameter_dfs.append(parameter_df)

        # Merge all parameter columns
        climate_df = parameter_dfs[0]
        if len(parameter_dfs) > 1:
            for remaining_parameter_df in parameter_dfs[1:]:
                climate_df = climate_df.merge(remaining_parameter_df[["cellId", "to", remaining_parameter_df.columns[-1]]], on=["cellId", "to"], how="outer")
        
        # Convert string to polygon
        print("Converting to polygon")
        climate_df['geometry'] = climate_df['coordinates'].map(string_to_polygon)

        # Convert into geo dataframe, with "DET DANSKE KVADRATNET" CRS
        climate_gdf = gpd.GeoDataFrame(climate_df, geometry='geometry', crs="EPSG:4326").to_crs("EPSG:25832") #crs="ETRS89") #crs="EPSG:3857")  #crs="EPSG:25832") # 32632
        climate_gdf.to_parquet(os.path.join(year_path, str(year) + "_postprocess.parquet"), index=False)   # Save

        print(climate_gdf)


def map_turbine_weather_cell(weather_gdf, master_gdf):
    """
        Maps wind turbines with cellIds from weather data in master_gdf turbine dataframe
    """
    cellId_polygon_map = {}  # {cellId : polygon} for all cellIds in the weather data
    for index, row in weather_gdf.iterrows():
        if row["cellId"] not in cellId_polygon_map:
            cellId_polygon_map[row["cellId"]] = row["geometry"]
    def locate_cellId(point):
        for cellId, polygon in cellId_polygon_map.items():
            if polygon.contains(point):
                return cellId
        return -1
    master_gdf["cellId"] = master_gdf["geometry"].map(lambda point: locate_cellId(point))
    return master_gdf

def add_weather_to_settlement(weather_path, folder, master_filename, data_years=[2018,2019]):
    all_settlements = []
    all_weather = []
    for year in data_years:
        # Read all weather files
        year_path = weather_path + str(year) + "grid"
        full_path_weather = os.path.join(year_path, str(year) + "_postprocess.parquet")
        all_weather.append(pd.read_parquet(full_path_weather))
    weather_df = pd.concat(all_weather, ignore_index=True)
    weather_df["geometry"] = gpd.GeoSeries.from_wkb(weather_df["geometry"])
    weather_gdf = gpd.GeoDataFrame(weather_df, geometry='geometry', crs="EPSG:25832")
    
    # Map weather cellIds on turbine master data
    master_df = pd.read_parquet(os.path.join(folder, master_filename))

    begin_year, end_year = str(data_years[0]), str(data_years[-1]) # First and last year of period

    # Remove turbines that cancel before period ends
    master_df[master_df["Valid_to"].isna()] = pd.Timestamp("2100-01-01")
    master_df.loc[master_df['Valid_to'] == "9999-01-01 00:00:00", ['Valid_to']] = pd.Timestamp("2100-01-01")
    master_df["Valid_to"] = pd.to_datetime(master_df["Valid_to"])
    master_df = master_df.drop(master_df[master_df["Valid_to"] < pd.Timestamp(end_year + "-01-01")].index)

    # Remove turbines that start after period begins
    master_df["Valid_from"] = pd.to_datetime(master_df["Valid_from"])
    master_df = master_df.drop(master_df[master_df["Valid_from"] > pd.Timestamp(begin_year + "-12-31")].index)

    # Drop turbines without X and Y coordinates
    master_df = master_df.drop(master_df[master_df["UTM_x"].isnull()].index)
    master_df = master_df.drop(master_df[master_df["UTM_y"].isnull()].index)

    # Remove duplicate GSRN
    master_df = master_df.drop(master_df[master_df.duplicated(subset=["GSRN"])].index)

    master_gdf = gpd.GeoDataFrame(master_df, geometry=gpd.points_from_xy(master_df['UTM_x'], master_df['UTM_y']), crs="EPSG:25832")
    master_gdf = map_turbine_weather_cell(weather_gdf, master_gdf) # Map weather cellIds on turbine data
    master_gdf = master_gdf[master_gdf["cellId"] != -1]

    yearly_settlements = []
    for year in data_years:
        # Read processed settlement for year and add weather cellId to the data, and then save
        full_path_settlement = os.path.join(folder, "settlement/" + str(year) + ".parquet")
        settlement_df = pd.read_parquet(full_path_settlement)
        settlement_df = settlement_df.merge(master_gdf[["GSRN", "cellId"]], on="GSRN", how="inner") # Add weather cellIds to data

        print(settlement_df)
        print(settlement_df["cellId"])

        settlement_df["cellId"] = settlement_df["cellId"].astype(str)
        yearly_settlements.append(settlement_df)
        settlement_df.to_parquet(os.path.join(folder, "frederik/settlement" + str(year) + "_postprocessed.parquet"), index=False)   # Save
        settlement_df.to_csv(os.path.join(folder, "frederik/settlement" + str(year) + "_postprocessed.csv"), index=False)

    return yearly_settlements

def load_weather_data(data_years, weather_path):
    """
        Initial preprocessing of weather data is done seperately
        Returns geodataframe with climate data per square before joining with turbine data 
    """
    climate_dfs = []
    for year in data_years:
        year_path = weather_path + str(year) + "grid"
        climate_dfs.append(
            pd.read_parquet(os.path.join(year_path, str(year) + "_postprocess.parquet"))
        )
    climate_df = pd.concat(climate_dfs, ignore_index=True)
    print(climate_df)
    climate_df["geometry"] = gpd.GeoSeries.from_wkb(climate_df["geometry"])
    gdf = gpd.GeoDataFrame(climate_df, geometry='geometry', crs="EPSG:25832")
    return gdf

def data_preprocess(data_path, master_filename, weather_path, data_years=[2018,2019]):

    # Load master data
    master_df = pd.read_parquet(os.path.join(data_path, master_filename))

    # Concatenate all years of settlement data
    settlement_list = []
    for year in data_years:
        settlement_list.append(
            pd.read_parquet(os.path.join(data_path, "frederik/settlement" + str(year) + "_postprocessed.parquet"))
        )

    settlement_df = pd.concat(settlement_list, ignore_index=True)
    print("# Wind turbines in settlement (after initial preprocessing):", len(settlement_df["GSRN"].unique()))

    weather_gdf = load_weather_data(data_years, weather_path) # Load weather data

    begin_year, end_year = str(data_years[0]), str(data_years[-1]) # First and last year of period

    # Remove turbines that cancel before period ends
    master_df[master_df["Valid_to"].isna()] = pd.Timestamp("2100-01-01")
    master_df.loc[master_df['Valid_to'] == "9999-01-01 00:00:00", ['Valid_to']] = pd.Timestamp("2100-01-01")
    master_df["Valid_to"] = pd.to_datetime(master_df["Valid_to"])
    master_df = master_df.drop(master_df[master_df["Valid_to"] < pd.Timestamp(end_year + "-01-01")].index)

    # Remove turbines that start after period begins
    master_df["Valid_from"] = pd.to_datetime(master_df["Valid_from"])
    master_df = master_df.drop(master_df[master_df["Valid_from"] > pd.Timestamp(begin_year + "-12-31")].index)

    # Drop turbines without X and Y coordinates
    master_df = master_df.drop(master_df[master_df["UTM_x"].isnull()].index)
    master_df = master_df.drop(master_df[master_df["UTM_y"].isnull()].index)

    # Remove duplicate GSRN
    master_df = master_df.drop(master_df[master_df.duplicated(subset=["GSRN"])].index)

    # Convert to geopandas with correct CRS
    master_gdf = gpd.GeoDataFrame(master_df, geometry=gpd.points_from_xy(master_df['UTM_x'], master_df['UTM_y']), crs="EPSG:25832")
    #master_gdf = gpd.GeoDataFrame(master_df, geometry=gpd.points_from_xy(master_df['UTM_x'], master_df['UTM_y']))
    #master_gdf.set_crs(epsg=4326, inplace=True)
    #master_gdf[master_gdf["GSRN"] == str(gsrn)]

    # Save final master data
    master_gdf.to_csv(os.path.join(data_path, "frederik/" + "final_master.csv"), index=False)
    #return

    # Convert coordinates to int
    #master_df['UTM_x'] = master_df['UTM_x'].astype(int)
    #master_df['UTM_y'] = master_df['UTM_y'].astype(int)

    # Get settlement data
    settlement_df["TIME_CET"] = pd.to_datetime(settlement_df["TIME_CET"])
    settlement_df["VAERDI"] = settlement_df["VAERDI"].astype(float)

    # Join settlement and master data
    ## note: May skip turbines in a wind farm, as they are are not present in settlement
    joined_df = master_gdf.merge(settlement_df, how="inner", on="GSRN")
    print("# Wind turbines after merging with master data:", len(joined_df["GSRN"].unique()))

    # Select useful columns
    column_names = [
        "GSRN", "TS_ID", "TIME_CET", "Capacity_kw", "Rotor_diameter", "Navhub_height", "cellId"
    ]        # Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv,Patv
    column_names.append("VAERDI")
    joined_df = joined_df[column_names]

    # Remove duplicate (turbine, timestep) pairs
    joined_df.drop_duplicates(subset=["GSRN", "TIME_CET"], inplace=True)
    print("# Wind turbines after dropping GSRN/TIME_CET duplicates:", len(joined_df["GSRN"].unique()))

    # Remove the first sample of the first year in the data, as it likely is the aggregated last hour/15 minutes
    # of the previous year
    """joined_df["minute"] = joined_df["TIME_CET"].dt.minute
    joined_df["hour"] = joined_df["TIME_CET"].dt.hour
    joined_df["dayofyear"] = joined_df["TIME_CET"].dt.dayofyear
    joined_df["year"] = joined_df["TIME_CET"].dt.year
    joined_df = joined_df[
        ~((joined_df["dayofyear"] == 1) & (joined_df["hour"] == 0) & (joined_df["minute"] == 0) & (joined_df["year"] == begin_year))
        ]"""

    # Downsample quarterly settlements to hourly
    # E.g. 00:15, 00:30, 00:45 and 01:00 will be downsampled (summed) to 01:00 - so the preceeding hour
    #quarterly_GSRN = joined_df[joined_df["minute"].isin([15,30,45])]["GSRN"].unique() # All turbines with quarterly values
    #qdf = joined_df[joined_df["GSRN"].isin(quarterly_GSRN)].set_index("TIME_CET")
    joined_df.set_index("TIME_CET", inplace=True)
    resampled_df = joined_df[["GSRN", "VAERDI"]].groupby("GSRN").resample("H", label="right", closed="right")["VAERDI"].sum()
    joined_df.reset_index(inplace=True)
    columns_to_join = list(joined_df.columns)
    columns_to_join.remove("VAERDI")
    merged_df = resampled_df.reset_index().merge(joined_df[columns_to_join], how="left", on=["GSRN", "TIME_CET"])

    # Streamline datetimes to UTC and join with weather data
    merged_df["TIME_CET"] = pd.to_datetime(merged_df["TIME_CET"], utc=True) - pd.Timedelta(hours=1)
    weather_gdf["to"] = pd.to_datetime(weather_gdf["to"])
    print(merged_df)
    print(weather_gdf)

    full_data = merged_df.merge(weather_gdf, left_on=["cellId", "TIME_CET"], right_on=["cellId", "to"], how="inner")
    print("# Wind turbines after weather data merge:", len(full_data["GSRN"].unique()))

    # Filter out wind turbines that don't have a value for all timesteps
    counts = full_data["GSRN"].value_counts()
    full_data = full_data[~full_data["GSRN"].isin(list(counts[counts < counts.max()].index))]
    print("# Wind turbines after removal of low counts:", len(full_data["GSRN"].unique()))
    full_data.sort_values(by=["TIME_CET", "GSRN"], inplace=True)

    # Filter out the date 31/12 as it is inconsistent
    full_data = full_data[~((full_data["TIME_CET"].dt.month == 12) & (full_data["TIME_CET"].dt.day == 31))]

    # Last year as test, else 50% of data
    if len(data_years) > 1:
        train_df = full_data[~(full_data["TIME_CET"].dt.year == data_years[-1])] # Everything but last year
        test_df = full_data[full_data["TIME_CET"].dt.year == data_years[-1]] # Last year

    else:
        train_size = len(train_df) // 2
        train_df = full_data[:train_size]
        test_df = full_data[train_size:]

    train_df.to_csv(os.path.join(data_path, "frederik/" + "final_train.csv"), index=False)
    test_df.to_csv(os.path.join(data_path, "frederik/" + "final_test.csv"), index=False)

def prep_weather_forecasts(
    data_path="/home/data_shares/energinet/energinet/",
    prog="prognosis",
    data_path2 = "/home/data_shares/energinet/energinet/frederik",
    master_filename = 'final_master.csv',
    ):
    grid_path = os.path.join(data_path, prog, "grid_coordinates.csv")
    grid_df = pd.read_csv(grid_path)
    grid_gdf = gpd.GeoDataFrame(grid_df, geometry=gpd.points_from_xy(grid_df["utm_x"], grid_df["utm_y"], crs="EPSG:25832"))

    train_data, test_data = load_data()
    master_df = train_data.master_df
    #master_df = pd.read_csv(os.path.join(data_path2, master_filename))
    master_df['geometry'] = master_df['geometry'].apply(shapely.wkt.loads)
    master_gdf = gpd.GeoDataFrame(data=master_df, geometry=master_df['geometry'], crs="EPSG:25832")

    from shapely.ops import nearest_points
    # unary union of the grid_df geometries 
    pts3 = grid_gdf.geometry.unary_union
    def near(point, pts=pts3):
        # find the nearest point and return the corresponding grid ID value
        nearest = grid_gdf.geometry == nearest_points(point, pts)[1]
        return grid_gdf[nearest].iloc[0]["grid"] #.get_values()[0]
    master_gdf['Forecast_grid'] = master_gdf.apply(lambda row: near(row.geometry), axis=1)

    prog_path = os.path.join(data_path, prog)
    ENetNEA_path = os.path.join(prog_path, "ENetNEA")
    HIR_path = os.path.join(prog_path, "HIR")
    val_id = 0
    for filename in os.listdir(ENetNEA_path):
        f = os.path.join(ENetNEA_path, filename)
        f2 = os.path.join(HIR_path, filename)
        if os.path.isfile(f) and os.path.isfile(f2):
            forecast_df = pd.read_parquet(f)
            forecast_df = forecast_df[(forecast_df.index < "2020-01-01 00:00:00")]
            forecast_df2 = pd.read_parquet(f2)
            forecast_df2 = forecast_df2[(forecast_df2.index > "2017-12-31 23:00:00") & (forecast_df2.index < "2018-02-23 00:00:00")]
            forecast_df = pd.concat([forecast_df2, forecast_df])

            val_name = filename.split(".")[0]
            melted_df = pd.melt(
                forecast_df,
                id_vars=["predicted_ahead"],
                # value_vars=[""], # Using all columns
                var_name="gridID",
                value_name=val_name,
                ignore_index=False,
            )
            melted_df.index = melted_df.index.set_names(["time"])
            melted_df = melted_df.reset_index()
            melted_df["time"] = pd.to_datetime(melted_df["time"])

            if val_id == 0:
                all_df = melted_df
            else:
                all_df = pd.concat([all_df, melted_df[val_name]], axis=1)
                #all_df = pd.merge(all_df, melted_df, on=["time", "predicted_ahead"])          
            val_id += 1

    d = dict.fromkeys(all_df.select_dtypes(np.float64).columns, np.float32)
    all_df = all_df.astype(d)
    all_df["gridID"] = all_df["gridID"].astype('int32')
    master_gdf["Forecast_grid"] = master_gdf["Forecast_grid"].astype('int32')

    train_df_data = train_data.df_data
    test_df_data = test_data.df_data

    all_df["time"] = pd.to_datetime(all_df["time"], utc=True)
    train_df_data["TIME_CET"] = pd.to_datetime(train_df_data["TIME_CET"], utc=True)
    test_df_data["TIME_CET"] = pd.to_datetime(test_df_data["TIME_CET"], utc=True)

    all_df = all_df.merge(master_gdf[["GSRN", "Forecast_grid"]], left_on="gridID", right_on="Forecast_grid")
    all_df = all_df[((all_df["time"].isin(train_df_data["TIME_CET"])) | (all_df["time"].isin(test_df_data["TIME_CET"])))]

    all_df.sort_values(by=["time", "predicted_ahead", "GSRN"])

    train_forecasts = all_df[all_df["time"] < "2019-01-01 00:00:00"]
    test_forecasts = all_df[all_df["time"] >= "2019-01-01 00:00:00"]

    print(train_forecasts)
    print(test_forecasts)

    train_forecasts.to_parquet(os.path.join(data_path, "frederik/" + "train_forecasts.parquet"))
    test_forecasts.to_parquet(os.path.join(data_path, "frederik/" + "test_forecasts.parquet"))
    #train_forecasts.to_csv(os.path.join(data_path, "frederik/" + "train_forecasts.csv"), index=False)
    #test_forecasts.to_csv(os.path.join(data_path, "frederik/" + "test_forecasts.csv"), index=False)

def df_to_tensor(train_data, forecast_df):
    all_df = forecast_df.copy()
    c = train_data.data_columns_forecast
    output_seq_length = train_data.output_seq_length
    t = torch.zeros((train_data.total_size, train_data.n_turbines, output_seq_length, len(c)))
    print(t.shape)
    # Convert time to datetime and sort
    all_df['time'] = pd.to_datetime(all_df['time'])
    times = all_df["time"].unique()
    max_time = times.max()
    # Calculate min_predicted_ahead and t_ahead once
    all_df['min_predicted_ahead'] = all_df.groupby('time')['predicted_ahead'].transform('min')
    all_df['t_ahead'] = all_df['predicted_ahead'] - all_df['min_predicted_ahead']
    # Pre-index data for faster filtering
    all_df.set_index(['time', 'predicted_ahead'], inplace=True)
    for t_idx, time in enumerate(times):
        print(t_idx, time)
        # Generate valid pairs for this time
        starting_df = all_df.loc[(time, slice(None))]
        #print(starting_df)
        min_predicted_ahead = starting_df['min_predicted_ahead'].iloc[0]
        print("min predicted ahead:",min_predicted_ahead)
        # Skip if sequence goes beyond max_time
        if time + pd.Timedelta(hours=output_seq_length) > max_time:
            print("Ending time:", time)
            # Reset index temporarily for filtering
            temp_df = all_df.reset_index()
            for t_ahead in range(output_seq_length):
                if time + pd.Timedelta(hours=t_ahead) > max_time:
                    continue
                # Perform filtering
                df_ahead = temp_df[
                    (temp_df["time"] - pd.Timedelta(hours=t_ahead) == time) &
                    (temp_df["predicted_ahead"] == min_predicted_ahead + t_ahead)
                ]
                df_ahead = df_ahead[c]
                t[t_idx, :, t_ahead, :] = torch.tensor(df_ahead.values)
            continue
        #print(min_predicted_ahead)
        time_list = [time + pd.Timedelta(hours=i) for i in range(output_seq_length)]
        integer_list = list(range(min_predicted_ahead, min_predicted_ahead + output_seq_length))
        pairs = pd.MultiIndex.from_tuples(zip(time_list, integer_list))
        #print(pairs)
        # Filter using the pre-indexed DataFrame
        subset_df = all_df.loc[pairs, c]
        # Reshape and assign directly
        subset_array = subset_df.values.reshape((output_seq_length, train_data.n_turbines, len(c)))
        t[t_idx, :, :, :] = torch.tensor(subset_array).permute(1, 0, 2)
    return t

def save_forecast_tensor(
    subset=3,
    split="train",
    data_path="/home/data_shares/energinet/energinet/"
):
    if split == "train":
        df_forecasts = pd.read_parquet(
            os.path.join(data_path, "frederik", "train_forecasts.parquet")
            )
        save_string = "train_forecasts_subset_"
    elif split == "test":
        df_forecasts = pd.read_parquet(
            os.path.join(data_path, "frederik", "test_forecasts.parquet")
            )
        save_string = "test_forecasts_subset_"
    print(df_forecasts)

    data = load_data(subset=subset, split=split)
    df_forecasts = df_forecasts[df_forecasts["GSRN"].isin(data.master_df["GSRN"])]
    df_forecasts.sort_values(by=["time", "GSRN"], inplace=True)
    print(df_forecasts)

    t = df_to_tensor(data, df_forecasts)
    save_path = os.path.join(data_path, "frederik/" + save_string + str(subset) + ".pt")
    print("Saving to:", save_path)
    torch.save(t, save_path)


def load_data(subset=3, split="all"):
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
        "handle_degrees": True
    }
    if split == "train":
        train_data = ENWPFDataset(parameters=parameters, flag='train', subset=geo_subset, subset_forecast="1", options=options_train)
        return train_data
    elif split == "test":
        test_data = ENWPFDataset(parameters=parameters, flag='test', subset=geo_subset, subset_forecast="1", options=options_train)
        return test_data
    else:
        train_data = ENWPFDataset(parameters=parameters, flag='train', subset=geo_subset, subset_forecast="1", options=options_train)    
        test_data = ENWPFDataset(parameters=parameters, flag='test', subset=geo_subset, subset_forecast="1", options=options_train)
        return train_data, test_data


def main():

    subset = str(sys.argv[1])
    split = str(sys.argv[2])
    save_forecast_tensor(subset=subset, split=split)

    #prep_weather_forecasts()

    #process_weather_data(weather_path, data_years = data_years, parameters = parameters)
    #yearly_settlements = add_weather_to_settlement(weather_path, folder, master_filename, data_years)
    """data_preprocess(folder, master_filename, weather_path, data_years=[2018,2019])"""

    ## CARBONTRACKER ##
    """tracker = CarbonTracker(epochs=1, epochs_before_pred=-1, monitor_epochs=-1, verbose=3)
    tracker.epoch_start()"""

    #weather_to_csv(weather_path, data_years, month) # Load climate data
    # Multi-threading - a thread for each month of the year
    """thread_list = []
    for month in range(1,13):
        thread = threading.Thread(target=weather_to_csv, args=(weather_path, data_years, month))
        thread_list.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in thread_list:
        thread.join()"""

    """tracker.epoch_end()
    tracker.stop()"""
    ## END CARBONTRACKER ##

    #parameters = ["mean_wind_speed", "mean_wind_dir"]

if __name__ == "__main__":
    main()