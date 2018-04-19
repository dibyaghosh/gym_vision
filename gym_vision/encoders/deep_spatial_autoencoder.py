import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class SpatialConvNetwork(nn.Module):
    def __init__(self,input_dim, feature_point_dim):
        super(SpatialConvNetwork,self).__init__()
        self.input_dim = input_dim
        self.feature_point_dim = feature_point_dim
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2)
        self.conv2 = nn.Conv2d(64, 32, 5)
        self.conv3 = nn.Conv2d(32, self.feature_point_dim, 5)
        presoftmax_dim = int((input_dim - 6) / 2 - 8) #  hardcoded based on dimensionality of conv layers
        self.x_mask = Variable(torch.from_numpy(np.array([i * np.ones(presoftmax_dim) / presoftmax_dim for i in range(presoftmax_dim)]).flatten().astype(np.float32)).cuda())
        self.y_mask = Variable(torch.from_numpy(np.array([np.arange(presoftmax_dim) / presoftmax_dim for i in range(presoftmax_dim)]).flatten().astype(np.float32)).cuda())
        print(self.x_mask.shape)


    def spatial_softmax(self, x):
        x_flat = x.view(x.shape[0], x.shape[1], -1) # N x channels x (presoftmax_dim)**2
        softmaxed_x = F.softmax(x_flat / .001, dim=2)
        
        #softmaxed_x = F.softmax(x_flat / self.alpha, dim=2)
        weighted_x = softmaxed_x * self.x_mask
        weighted_y = softmaxed_x * self.y_mask

        mean_x = torch.sum(weighted_x, dim=2) # N x channels
        mean_y = torch.sum(weighted_y, dim=2)
        #import ipdb; ipdb.set_trace()
        return torch.cat([mean_x, mean_y], dim=1) # N x (2 channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        feature_points = self.spatial_softmax(x)
        return feature_points

class DeepSpatialAutoencoder(nn.Module):
    """input_dim = 224, feature_point_dim = 16, reconstruct_dim=60"""
    def __init__(self, input_dim, feature_point_dim, reconstruct_dim): 
        super(DeepSpatialAutoencoder, self).__init__()


        self.feature_point_dim = feature_point_dim
        self.reconstruct_dim = reconstruct_dim
        output_dim = self.reconstruct_dim ** 2

        self.feature_point_network = SpatialConvNetwork(input_dim,feature_point_dim)

        self.reconstruction_net = nn.Sequential(
            nn.Linear(feature_point_dim * 2, feature_point_dim * 10),
            nn.ReLU(),
            nn.Linear(feature_point_dim * 10 , feature_point_dim * 100),
            nn.ReLU(),
            nn.Linear(feature_point_dim * 100, output_dim),
        )
        
    def forward(self, x):
        feature_points = self.feature_point_network(x)
        reconstructed_output = self.reconstruction_net(feature_points)
        return reconstructed_output.view(feature_points.shape[0],self.reconstruct_dim,self.reconstruct_dim)


def loss(model,image,ds_image,next_image):
    criterion = nn.MSELoss()
    pred_ds = model(image)
    
    #fp1 = model.feature_point_network(image)
    #fp2 = model.feature_point_network(next_image)

    return criterion(pred_ds,ds_image)# + 0.1 * criterion(fp1,fp2)
