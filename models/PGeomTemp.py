import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU, A3TGCN, A3TGCN2
from torch_geometric.nn import LayerNorm
from torch.utils.data import DataLoader

class RecurrentGCN(torch.nn.Module):
    """
        Adapted from https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html
    """
    def __init__(self, node_features, filters, k, output_size, dropout=0.5):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GConvGRU(node_features, filters, k)
        self.linear = torch.nn.Linear(filters, output_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity() # manual dropout

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.linear(h)
        return h

class A3TGCNModel(torch.nn.Module):
    """
        Adapted from https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html
    """
    def __init__(self, node_features, forecast_size, hidden_size, periods, output_size, batch_size=False):
        super(A3TGCNModel, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        if batch_size:
            self.encoder_tgnn = A3TGCN2(in_channels=node_features,  out_channels=hidden_size, periods=periods, batch_size=batch_size)
            self.decoder_tgnn = A3TGCN2(in_channels=forecast_size,  out_channels=hidden_size, periods=output_size, batch_size=batch_size)
        else:
            self.encoder_tgnn = A3TGCN(in_channels=node_features, out_channels=hidden_size, periods=periods)
            self.decoder_tgnn = A3TGCN(in_channels=forecast_size, out_channels=hidden_size, periods=output_size)
        #self.tgnn_norm = LayerNorm(hidden_size)
        # Equals single-shot prediction. Concat graph outputs
        self.linear = torch.nn.Linear(2 * hidden_size, output_size)

    def forward(self, x_hist, x_forecast, turb_ids, edge_index, edge_weights):
        """
        x = Node features for T time steps
        model expects shape: (B, N, M, T)
        edge_index = Graph edge indices
        """
        B, T, N, M = x_hist.shape
        B2, T2, N2, M2 = x_forecast.shape

        x_hist = x_hist.permute(0, 2, 3, 1) # (B, N, M, T)
        x_forecast = x_forecast.permute(0, 2, 3, 1)
        h1 = self.encoder_tgnn(x_hist, edge_index) 
        h2 = self.decoder_tgnn(x_forecast, edge_index)
        h = torch.cat((h1, h2), dim=-1)
        h = self.linear(h) # (B, N, T)
        h = h.permute(0, 2, 1) # (B, T, N)
        h = F.relu(h)
        return h

class A3TGCN2Model(torch.nn.Module):
    """
        Adapted from https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html
    """
    def __init__(self, node_features, periods, batch_size):
        super(A3TGCNModel, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features,  out_channels=32, periods=periods, batch_size=batch_size) # node_features=2, periods=12
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)
        h = F.relu(h) 
        h = self.linear(h)
        h = h.permute(0, 2, 1)
        return h

def main():
    pass

if __name__ == "__main__":
    main()