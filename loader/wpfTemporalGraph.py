import io
import json
import numpy as np
from six.moves import urllib
import shapely
from shapely.geometry import Polygon, LineString, Point
import geopandas as gpd
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, StaticGraphTemporalSignalBatch
from torch_geometric_temporal.signal import temporal_signal_split


class ENWPFDatasetWrapper(object):
    """Wrapper for ENWPFDataset object to get graph data for
       Pytorch Geometric Temporal.
    """
    def __init__(self, distance_method, threshold=float("inf")):
        self.distance_method = distance_method
        self.threshold = threshold

    def _distance_method(self, distance):
        if self.distance_method == "inverse":
            return 1 / (distance + 1e-3) # Avoids zero division
        if self.distance_method == "exponential_decay_weight":
            return np.exp(-0.001 * distance)

    def _get_edges_and_weights(self):
        master_df = self.wpf_object.master_df
        try:
            master_df['geometry'] = master_df['geometry'].apply(lambda x: shapely.wkt.loads(x))
        except:
            pass
        gdf = gpd.GeoDataFrame(data=master_df, geometry=master_df['geometry'], crs="EPSG:25832")
        self.nodes = {} # {id: (x,y) for all nodes} For networkx graph/viz
        self.line_strings = [] # List of tuples of (linestring, edge) for geopandas viz
        srcs, tgts = [], []
        edge_weights = []
        for src, point in enumerate(gdf["geometry"]):
            self.nodes[src] = (point.x, point.y)
            for tgt, point2 in enumerate(gdf["geometry"]):
                distance = point.distance(point2)
                if distance < self.threshold:
                    srcs.append(src)
                    tgts.append(tgt)
                    if src == tgt:
                        w = 1
                    else:
                        w = self._distance_method(distance)
                    edge_weights.append(w)
                    self.line_strings.append((LineString([point, point2]), w))
        self._edges = np.array([srcs, tgts])
        self._edge_weights = np.array(edge_weights)

    def _get_targets_and_features(self):
        array = self.wpf_object.array_data # (T,N,M)
        forecast_array = self.wpf_object.forecast_array # (T,O,N,M)
        self.features = [
            array[i : i + self.lags, :, :].permute(1, 2, 0).cpu().numpy() #.float().cpu() (T, N, M).permute(1, 2, 0) -> (N, M, T)
            for i in range(self.effective_length) 
        ]
        self.forecasts = [
            forecast_array[i, :, :, :].permute(1, 2, 0).cpu().numpy() #.float().cpu() (O, N, M).permute(1, 2, 0) -> (N, M, O)
            for i in range(self.effective_length) 
        ]
        self.targets = [
            array[i + self.lags : i + self.lags + self.output_length, :, :].permute(1, 2, 0).cpu().numpy()
            for i in range(self.effective_length) 
        ]
        
    def get_dataset(self, wpf_object, only_y=True, batch_size=False) -> StaticGraphTemporalSignal:
        """Returning the WPF data iterator. Inherits meta from wpf object

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The WPF dataset.
        """
        self.wpf_object = wpf_object
        self.effective_length = len(wpf_object)
        self.output_length = wpf_object.output_seq_length
        self.only_y = only_y
        self.lags = wpf_object.input_seq_length
        self._get_edges_and_weights()
        self._get_targets_and_features()

        if batch_size:
            dataset = StaticGraphTemporalSignal(
                self._edges, self._edge_weights, self.features, self.targets, forecasts=self.forecasts
            )
        else:
            dataset = StaticGraphTemporalSignal(
                self._edges, self._edge_weights, self.features, self.targets, forecasts=self.forecasts
            )
        return dataset


def main():
    from torch_geometric_temporal.signal import temporal_signal_split
    from wpfTemporalGraph import ENWPFDatasetWrapper
    loader = ENWPFDatasetWrapper(train_data, "inverse")
    dataset = loader.get_dataset(lags=24)
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)
    for time, snapshot in enumerate(train_dataset):
        print(time)
        print(snapshot)
        #print(snapshot.x)
        print(snapshot.edge_index)
        print(snapshot.edge_attr)
        print("--")

if __name__ == "__main__":
    main()
