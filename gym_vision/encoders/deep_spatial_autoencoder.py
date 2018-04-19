import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class DeepSpatialAutoencoder(nn.Module):
    """input_dim = 224, feature_point_dim = 16, reconstruct_dim=60"""
    def __init__(input_dim, feature_point_dim, reconstruct_dim): 
        super(DeepSpatialAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.feature_point_dim = feature_point_dim # each one will have 2 coordinates representing x,y
        self.reconstruct_dim = reconstruct_dim

        self.alpha = torch.tensor([1], requires_grad=True) # scalar value?

        self.build_conv_layers()

        presoftmax_dim = (input_dim - 6) / 2 - 12 #  hardcoded based on dimensionality of conv layers
        x_mask = np.flatten(np.array([i * np.ones(presoftmax_dim) for i in range(presoftmax_dim)]))
        y_mask = np.flatten(np.array([np.arange(presoftmax_dim) for i in range(presoftmax_dim)])

    def build_conv_layers(self):
        self.conv1 = nn.Conv2D(3, 64, 7, stride=2)
        self.conv2 = nn.Conv2D(64, 32, 5)
        self.conv3 = nn.Conv2D(32, self.feature_point_dim, 5)

    def spatial_softmax(self, x):
        x_flat = x.view(x.shape[0], x.shape[1], -1) # N x channels x (presoftmax_dim)**2
        softmaxed_x = F.softmax(x_flat / self.alpha, dim=2)
        weighted_x = softmaxed_x * x_mask
        weighted_y = softmaxed_x * y_mask

        mean_x = torch.mean(weighted_x, dim=2) # N x channels
        mean_y = torch.mean(weighted_y, dim=2)

        return torch.cat([mean_x, mean_y], dim=1) # N x (2 channels)

    def reconstruct(self, feature_points):
        output_dim = self.reconstruct_dim ** 2
        input_dim = self.feature_point_dim * 2
        reconstruction_net = nn.Linear(input_dim, output_dim)
        reconstructed_output = reconstruction_net(feature_points)
        return reconstructed_output.view(feature_points.shape[0], output_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        feature_points = spatial_softmax(x)
        return reconstruct(feature_points)
"""
    Initialize like this
    dsa = DeepSpatialAutoencoder(224, 16, 56)
"""

def train_step(x, y, dsa, optimizer):
    optimizer.zero_grad()
    loss = nn.MSELoss(dsa(x), x)
    loss.backward()
    optimizer.step()


