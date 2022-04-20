import torch

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.graphgym.config import cfg

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
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
        self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)
        self.lin3 = torch.nn.Linear(hidden_dim, out_dim)

        # self.conv4 = GCNConv(out_dim, hidden_dim)
        self.lin4 = torch.nn.Linear(out_dim, hidden_dim)
        # self.conv5 = GCNConv(hidden_dim, hidden_dim)
        self.lin5 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.conv6 = GCNConv(hidden_dim, input_dim)
        self.lin6 = torch.nn.Linear(hidden_dim, input_dim)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        # print(x.shape)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index) + self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index) + self.lin3(x)
        return F.log_softmax(x, dim=1)
    
    def decode(self, x, edge_index):
        # x = F.relu(z)
        # x = F.dropout(x, training=self.training)
        x = self.lin4(x) #+ self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x =  self.lin5(x) #+ self.conv5(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin6(x) #+ self.conv6(x, edge_index) 
        return x

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index
        # print(x.shape, edge_index.shape)
        z = self.encode(x, edge_index)
        imp = self.decode(z, edge_index)
        # print(x.shape)
        return imp
    
    # def loss(self, res, ori):
    #     mse_loss = torch.nn.MSELoss()
    #     return mse_loss(res.flatten() , ori.flatten())
    def loss(self, res, ori, l2_regularization = 5e-4):
     
        loss = torch.nn.MSELoss()(res.flatten() , ori.flatten())         
        loss

        l2_loss = 0.0
        for param in self.parameters():
            data = param* param
            l2_loss += data.sum()
        loss += 0.2* l2_regularization* l2_loss
        return loss

if __name__=='__main__':
    assert torch.cuda.is_available()


