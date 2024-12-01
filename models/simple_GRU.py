import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.func import stack_module_state
from torch.func import functional_call
import copy
from torch import vmap

from torch_geometric.nn import GCNConv, ChebConv, EdgeConv
from torch_geometric.nn import MLP
from torch_geometric.nn import LayerNorm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SingleGRU(nn.Module):
    """
        GRU for single turbine.
        With embeddings
    """
    def __init__(
        self, 
        n_turbines, 
        input_size, 
        forecast_size, 
        hidden_size, 
        output_size, 
        embedding_dim=None,
        num_layers=1, 
        bidirectional=True,
        dropout=0.3
        ):
        super(SingleGRU, self).__init__()
        self.embedding_dim = embedding_dim
        if embedding_dim:
            self.embedding = nn.Embedding(n_turbines, embedding_dim)
            input_size = input_size + embedding_dim
            forecast_size = forecast_size + embedding_dim
            self.emb_input_norm = LayerNorm(input_size)
            self.emb_forecast_norm = LayerNorm(forecast_size)
        self.num_directions = 2 if bidirectional else 1
        self.encoder_gru = nn.GRU(
                input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional
                ) # Share GRU
        self.decoder_gru = nn.GRU(
                forecast_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional
                ) # Share GRU
        self.gru_norm = nn.LayerNorm(hidden_size  * self.num_directions)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity() # manual dropout
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size) # Shared linear layer for turbine GRUs

    def forward(self, x_hist, x_forecast, turb_ids):
        # x_hist: (B, T, N, M)
        # x_forecast: (B, O, N, M)
        # turb_ids: 
        B, T, N, M = x_hist.shape
        B2, T2, N2, M2 = x_forecast.shape
        x_hist = x_hist.permute(0, 2, 1, 3) # (B, N, T, M)
        x_hist = x_hist.reshape(B*N, T, M) # Combining batch and turbines as batches
        x_forecast = x_forecast.permute(0, 2, 1, 3) # (B, N, O, M)
        x_forecast = x_forecast.reshape(B2*N2, T2, M2) # batch*turbine

        if self.embedding_dim:
            # Get emb
            turb_ids = turb_ids.unsqueeze(0).repeat(B, 1).reshape(-1)
            embs = self.embedding(turb_ids) # (B * N) -> (B * N, embedding_dim)
            # Historical emb
            embs_hist = embs.unsqueeze(1).repeat(1, T, 1) # (B * N, T, embedding_dim)
            x_hist = torch.cat((x_hist, embs_hist), dim=2)
            x_hist = self.emb_input_norm(x_hist)
            # Forecast emb
            embs_forecast = embs.unsqueeze(1).repeat(1, T2, 1)
            x_forecast = torch.cat((x_forecast, embs_forecast), dim=2)
            x_forecast = self.emb_forecast_norm(x_forecast)
            #x = self.emb_norm(x)
            #x = F.relu(x)

        _, hidden = self.encoder_gru(x_hist)
        x, _ = self.decoder_gru(x_forecast, hidden)
        x = x[:, -1, :] # (B*N, hidden) # Still batches*turbines as batches
        x = self.gru_norm(x)
        #x = x[:, -1, :] # (B*N, hidden)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.relu(x) # Non-negative outputs
        x = x.reshape(B, N, -1)
        x = x.permute(0, 2, 1)
        return x # (B, T, N)

class EfficientGCNGRU(nn.Module):
    """
        GCN-GRU with shared model for each turbine, including linear layer.
    """
    def __init__(
        self, 
        n_turbines, 
        input_size, 
        forecast_size, 
        gcn_channels, 
        hidden_size, 
        output_size, 
        edge_conv_out_channels=False, 
        gcn_layer="GCNConv", 
        k=1, 
        embedding_dim=None, 
        num_layers=1, 
        bidirectional=True, 
        dropout=0.5
        ):
        super(EfficientGCNGRU, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.edge_conv_out_channels = edge_conv_out_channels
        
        if edge_conv_out_channels:
            # EdgeConv layer - with MLP
            self.edge_conv_mlp = MLP([2 * forecast_size, edge_conv_out_channels])
            self.edge_conv = EdgeConv(self.edge_conv_mlp, aggr='mean', node_dim=1)
            self.ecn_norm = LayerNorm(self.edge_conv_out_channels)
            forecast_size = self.edge_conv_out_channels

        if gcn_layer == "ChebConv":
            self.gcn = ChebConv(forecast_size, gcn_channels, K=k, node_dim=1)
        else:
            self.gcn = GCNConv(forecast_size, gcn_channels, node_dim=1)
        self.gcn_norm = LayerNorm(gcn_channels)
        forecast_size = gcn_channels

        self.gru = SingleGRU(
            n_turbines, input_size, forecast_size, hidden_size, output_size, embedding_dim, num_layers, bidirectional, dropout
        )

    def forward(self, x_hist, x_forecast, turb_ids, edge_index, edge_weight=None):
        # x shape: (batch_size, sequence_length, num_turbines, num_features)
        B, T, N, M = x_hist.shape
        B2, T2, N2, M2 = x_forecast.shape

        # Reshape input to combine batch_size and sequence_length as batch
        x_forecast = x_forecast.reshape(B2 * T2, N2, M2)
        if self.edge_conv_out_channels:
            x_forecast = self.edge_conv(x_forecast, edge_index)
            x_forecast = self.ecn_norm(x_forecast)
            x_forecast = F.relu(x_forecast)

        # Apply GCNConv to all time steps and batches simultaneously
        x_forecast = self.gcn(x_forecast, edge_index)
        x_forecast = self.gcn_norm(x_forecast)
        x_forecast = F.relu(x_forecast)
        # Reshape output to (batch_size * num_turbines, forecast_length, gcn_hidden)
        x_forecast = x_forecast.view(B2, T2, N2, -1)
        # GRU in: (B, T, N, M)
        x = self.gru(x_hist, x_forecast, turb_ids)
        return x


class GRUModel(nn.Module):
    def __init__(self, n_turbines, input_size, hidden_size, output_size, num_layers=1, bidirectional=True, dropout=0.5):
        super(GRUModel, self).__init__()
        self.n_turbines = n_turbines # Number of parallel GRUs to run - one for each turbine
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.gru_list = nn.ModuleList(
            [nn.GRU(
                input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional
                )
            for _ in range(n_turbines)])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity() # manual dropout
        self.fc_list = nn.ModuleList([nn.Linear(hidden_size * self.num_directions, output_size) for _ in range(n_turbines)])
        #self.fc = nn.Linear(hidden_size, output_size) # Shared linear layer for turbine GRUs

    def forward(self, x_list):
        # x_list: (batch_size, sequence_length, n_turbines, m_features)
        outputs = []
        for i in range(self.n_turbines):
            turbine_batch = x_list[:,:,i,:] # Selects data for 1 turbine (B, T, N, M) -> (Batch, Time, M_features)
            out, _ = self.gru_list[i](turbine_batch) # out: (B,T,D∗H) | out[:,-1,:]: (B,D∗H), where B: Batch, T:Time and H:Hidden_size
            out = self.dropout(out)
            out = self.fc_list[i](out[:, -1, :]).unsqueeze(2) # Get the hidden state of the last output 
            outputs.append(out)
        return torch.cat(outputs, dim=2) # (B, T, N)

class GCN_GRUModel(nn.Module):
    def __init__(self, n_turbines, input_size, gcn_channels, hidden_size, output_size, edge_conv_out_channels=False, gcn_layer="GCNConv", k=1, num_layers=1, bidirectional=True, dropout=0.5):
        super(GCN_GRUModel, self).__init__()
        self.n_turbines = n_turbines
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.edge_conv_out_channels = edge_conv_out_channels

        if edge_conv_out_channels:
            # EdgeConv layer - with MLP
            self.edge_conv_mlp = MLP([2 * input_size, edge_conv_out_channels])
            self.edge_conv = EdgeConv(self.edge_conv_mlp, aggr='mean')
            input_size = self.edge_conv_out_channels

        if gcn_layer == "ChebConv":
            self.gcn = ChebConv(input_size, gcn_channels, K=k)
        else:
            self.gcn = GCNConv(input_size, gcn_channels)

        self.gru_list = nn.ModuleList(
            [nn.GRU(
                gcn_channels, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional
                )
            for _ in range(n_turbines)])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity() # manual dropout
        self.fc_list = nn.ModuleList([nn.Linear(hidden_size * self.num_directions, output_size) for _ in range(n_turbines)])

    def forward(self, x, edge_index, edge_weight=None):
        # x has shape [B, N, M, T]
        B, N, M, T = x.size()

        # Initialize list to store GRU outputs for each node
        all_gru_out = []

        if self.edge_conv_out_channels:
            # Apply EdgeConv across the time dimension
            edge_conv_out = []
            for t in range(T):
                edge_conv_out_t = self.edge_conv(x[:, :, :, t].reshape(B * N, M), edge_index)  # Apply EdgeConv
                edge_conv_out.append(edge_conv_out_t.view(B, N, self.edge_conv_out_channels))  # Reshape to [B, N, edge_conv_out_channels]
            # Stack EdgeConv outputs to get shape [B, N, edge_conv_out_channels, T]
            x = torch.stack(edge_conv_out, dim=-1)

        # Apply GCN on each timestep independently
        gcn_out = []
        for t in range(T):
            out = self.gcn(x[:, :, :, t].reshape(B * N, -1), edge_index, edge_weight)  # Flattened shape [B*N, M]
            #print("GCN OUT SHAPE:")
            #print(out.shape)
            gcn_out.append(out.reshape(B, N, -1))  # Reshape to [B, N, gcn_channels]
        # Stack GCN outputs to get shape [B, N, T, gcn_channels]
        gcn_out = torch.stack(gcn_out, dim=2)

        # Process each node with its own GRU
        for n in range(N):
            node_features = gcn_out[:, n, :, :]  # Shape [B, T, gcn_channels] for node n
            out, _ = self.gru_list[n](node_features)  # Process with GRU
            out = self.dropout(out)
            out = self.fc_list[n](out[:, -1, :])  # Get the hidden state of the last output
            all_gru_out.append(out)  # Shape [B, output_size]
 
        # Stack GRU outputs to get shape [B, output_size, N]
        all_gru_out = torch.stack(all_gru_out, dim=1)

        return all_gru_out

def main():

    folder = "/home/data_shares/energinet/energinet/"
    weather_path = "/home/frfa/thesis/data/"
    data_years = [2018, 2019]
    subset = [[530000, 601000], [6310000, 6410000]]

    train_data = ENWPFDataset(
        data_path = folder,
        weather_path = weather_path,
        capacity = capacity,
        data_years = data_years,
        flag = 'train',
        subset = subset
        )
    val_data = ENWPFDataset(
        data_path = folder,
        weather_path = weather_path,
        capacity = capacity,
        data_years = data_years,
        flag = 'val',
        subset = subset
        )

    train_data_loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True,
        drop_last=True,
        num_workers=2)

    val_data_loader = DataLoader(
        val_data,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=2)

    # Example usage
    input_size = train_data.array_data.shape[2] # Number of input features. array_data.shape = (T, N, M)
    hidden_size = 64  # Number of hidden units in GRU
    output_size = 1  # Number of output units
    num_layers = 2  # Number of GRU layers

    model = GRUModel(input_size, hidden_size, output_size, num_layers)
    print(model)

if __name__ == "__main__":
    main()