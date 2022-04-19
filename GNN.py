import torch

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.graphgym.config import cfg

class GCN(torch.nn.Module):
    def __init__(input_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)

class GAEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index
        # print(x.shape, edge_index.shape)
        x = self.conv1(x, edge_index)
        # print(x.shape)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        # print(x.shape)
        return F.log_softmax(x, dim=1)

class GAEncoder_Decoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)
        self.conv4 = GCNConv(out_dim, hidden_dim)
        self.conv5 = GCNConv(hidden_dim, input_dim)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # print(x.shape)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def decode(self, z, edge_index):
        x = F.relu(z)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv5(x, edge_index)
        return x

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index
        # print(x.shape, edge_index.shape)
        x = self.conv1(x, edge_index)
        # print(x.shape)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.log_softmax(x, dim=1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv5(x, edge_index)
        # print(x.shape)
        return x
    
    def loss(self, res, ori):
        mse_loss = torch.nn.MSELoss(reduction=cfg.model.size_average)
        return mse_loss(res.flatten() , ori.flatten())
if __name__=='__main__':
    assert torch.cuda.is_available()


