import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
from logging import getLogger
import numpy as np

class Baselines(nn.Module):
    """
    Simple baselines based on previous timestep
    # Input shape: (B, T, N)
    """
    def __init__(self, output_length):
        super(Baselines, self).__init__()
        self.output_length = output_length

    def predictPersistence(self, tensor):
        #input_y = input_y.to(self.device)  # Make sure input is on the right device
        last_timestep = tensor[:, -1, :]
        return last_timestep.unsqueeze(1).expand(-1, self.output_length, -1)

    def predictMean(self, tensor):
        #input_y = input_y.to(self.device)  # Move input tensor to the right device
        mean_over_T = torch.mean(tensor, dim=1, keepdim=True)
        return mean_over_T.expand(-1, self.output_length, -1)

    """def to(self, device):
        self.device = device  # Store the device
        super(Baselines, self).to(device)  # Call the parent's to() method for proper handling
        return self"""

