# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 22:25:08 2023

@author: Nikola
"""
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import numpy as np



class CNN_AE(nn.Module):
    def __init__(self, n_chans1=[8,8,8], k_size = [6,6,6], k_pool_size = [4,4,4], padding_t='same'):
        super().__init__()

        self.chans1 = n_chans1
        self.conv1_e = nn.Conv2d(in_channels=1, out_channels=self.chans1[0], kernel_size=k_size[0],  padding=padding_t)  # add stride=(1,2) to each layer
        self.conv2_e = nn.Conv2d(in_channels=n_chans1[0], out_channels=self.chans1[1], kernel_size=k_size[1], padding=padding_t)
        self.conv3_e = nn.Conv2d(in_channels=self.chans1[1],out_channels=self.chans1[2], kernel_size=k_size[2],  padding=padding_t)


        self.conv1_d = nn.ConvTranspose2d(in_channels = n_chans1[2], out_channels = self.chans1[1], kernel_size = k_pool_size[2], stride=2, output_padding=(0,1))
        self.conv2_d = nn.ConvTranspose2d(in_channels = n_chans1[1], out_channels = self.chans1[0], kernel_size = k_pool_size[1], stride=2, output_padding=(1,0))
        self.conv3_d = nn.ConvTranspose2d(in_channels = n_chans1[0], out_channels = 1,              kernel_size = k_pool_size[0], stride=2, output_padding=(1,0))
        self.dropout = nn.Dropout(0.1)
    def encode(self, x):
        #print(x.size())
        out = torch.relu(self.conv1_e(x))
        #out = self.dropout(out)
        out = F.max_pool2d(out, stride=2, kernel_size=4) # 256x105

        out = torch.relu(self.conv2_e(out))
        #out = self.dropout(out)
        out = F.max_pool2d(out, stride=2, kernel_size=4) # 256x105

        out = torch.relu(self.conv3_e(out))
        #out = self.dropout(out)
        out = F.max_pool2d(out, stride=2, kernel_size=4)
        return out

    def decode(self, x):
        out = torch.relu(self.conv1_d(x))
        #out = self.dropout(out)
        out = torch.relu(self.conv2_d(out))
        #out = self.dropout(out)
        #out = torch.sigmoid(self.conv3_d(out))
        out = self.conv3_d(out)

        return out

    def forward(self,x):

        out = self.encode(x)
        out = self.decode(out)

        return out

    def number_of_params(self):
         print('Numer of network paramteres:')
         print(sum(p.numel() for p in self.parameters()))

class CNN_AE_MEL(nn.Module):
    def __init__(self, n_chans1=[8,8,8], k_size = [3,3,3], k_pool_size = [4,4,4], padding_t='same'):
        super().__init__()

        self.chans1 = n_chans1
        self.conv1_e = nn.Conv2d(in_channels=1, out_channels=self.chans1[0], kernel_size=k_size[0],  padding=padding_t)  # add stride=(1,2) to each layer
        self.conv2_e = nn.Conv2d(in_channels=n_chans1[0], out_channels=self.chans1[1], kernel_size=k_size[1], padding=padding_t)
        self.conv3_e = nn.Conv2d(in_channels=self.chans1[1],out_channels=self.chans1[2], kernel_size=k_size[2],  padding=padding_t)

        self.conv1_d = nn.ConvTranspose2d(in_channels = n_chans1[2], out_channels = self.chans1[1], kernel_size = k_pool_size[2], stride=2, output_padding=(1,0))
        self.conv2_d = nn.ConvTranspose2d(in_channels = n_chans1[1], out_channels = self.chans1[0], kernel_size = k_pool_size[1], stride=2)
        self.conv3_d = nn.ConvTranspose2d(in_channels = n_chans1[0], out_channels = 1,              kernel_size = k_pool_size[0], stride=4, output_padding=(0,1))

        self.bn1 = nn.BatchNorm2d(n_chans1[0])
        self.bn2 = nn.BatchNorm2d(n_chans1[1])
        self.bn3 = nn.BatchNorm2d(n_chans1[2])

        self.bn4 = nn.BatchNorm2d(n_chans1[1])
        self.bn5 = nn.BatchNorm2d(n_chans1[0])
        self.bn6 = nn.BatchNorm2d(1)


    def encode(self, x):
        #print(x.size())
        out = self.conv1_e(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = F.max_pool2d(out, stride=4, kernel_size=4) # 256x105


        out = self.conv2_e(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = F.max_pool2d(out, stride=2, kernel_size=4) # 256x105

        out = self.conv3_e(out)
        out = self.bn3(out)
        out = torch.relu(out)
        out = F.max_pool2d(out, stride=2, kernel_size=4)
        return out

    def decode(self, x):
        out = self.conv1_d(x)
        out = self.bn4(out)
        out = torch.relu(out)

        out = self.conv2_d(out)
        out = self.bn5(out)
        out = torch.relu(out)

        out = self.conv3_d(out)
        #out = torch.sigmoid(out)
        #out = self.bn6(out)

        return out

    def forward(self,x):

        out = self.encode(x)
        out = self.decode(out)

        return out

    def number_of_params(self):
         print('Numer of network paramteres:')
         print(sum(p.numel() for p in self.parameters()))

class DNN_AE(nn.Module):

    def __init__(self, in_shape, n_layers = [256,64,16]):
        super().__init__()

        self.lin1_e = nn.Linear(in_shape, n_layers[0])  # add stride=(1,2) to each layer
        self.lin2_e = nn.Linear(n_layers[0], n_layers[1])
        self.lin3_e = nn.Linear(n_layers[1], n_layers[2])


        self.lin1_d = nn.Linear(n_layers[2],n_layers[1])
        self.lin2_d = nn.Linear(n_layers[1], n_layers[0])
        self.lin3_d = nn.Linear(n_layers[0], in_shape)
        self.dropout = nn.Dropout(0.2)

    def encode(self, x):
        out = torch.relu(self.lin1_e(x))
        #out = self.dropout(out)
        out = torch.relu(self.lin2_e(out))
        #out = self.dropout(out)
        out = torch.relu(self.lin3_e(out))

        return out

    def decode(self, x):
        out = torch.relu(self.lin1_d(x))
        #out = self.dropout(out)
        out = torch.relu(self.lin2_d(out))
        #out = self.dropout(out)
        out = self.lin3_d(out)

        return out

    def forward(self, x):

        out = self.encode(x)
        out = self.decode(out)

        return out


    def number_of_params(self):
         print('Numer of network paramteres:')
         print(sum(p.numel() for p in self.parameters()))


class DNN_AE_ME(nn.Module):

    def __init__(self, in_shape, n_layers = [128,128,128,128,8]):
        super().__init__()

        self.lin1_e = nn.Linear(in_shape, n_layers[0])  # add stride=(1,2) to each layer
        self.lin2_e = nn.Linear(n_layers[0], n_layers[1])
        self.lin3_e = nn.Linear(n_layers[1], n_layers[2])
        self.lin4_e = nn.Linear(n_layers[2], n_layers[3])
        self.lin5_e = nn.Linear(n_layers[3], n_layers[4])


        self.lin1_d = nn.Linear(n_layers[4],n_layers[3])
        self.lin2_d = nn.Linear(n_layers[3], n_layers[2])
        self.lin3_d = nn.Linear(n_layers[2], n_layers[1])
        self.lin4_d = nn.Linear(n_layers[1], n_layers[0])
        self.lin5_d = nn.Linear(n_layers[0], in_shape)
        self.dropout = nn.Dropout(0.4)

        self.bn1 = nn.BatchNorm1d(n_layers[0])
        self.bn2 = nn.BatchNorm1d(n_layers[1])
        self.bn3 = nn.BatchNorm1d(n_layers[2])
        self.bn4 = nn.BatchNorm1d(n_layers[3])
        self.bn5 = nn.BatchNorm1d(n_layers[4])

        self.bn6 = nn.BatchNorm1d(n_layers[3])
        self.bn7 = nn.BatchNorm1d(n_layers[2])
        self.bn8 = nn.BatchNorm1d(n_layers[1])
        self.bn9 = nn.BatchNorm1d(n_layers[0])

    def encode(self, x):
        out = self.lin1_e(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.lin2_e(out)
        out = self.bn2(out)
        out = torch.relu(out)

        out = self.lin3_e(out)
        out = self.bn3(out)
        out = torch.relu(out)

        out = self.lin4_e(out)
        out = self.bn4(out)
        out = torch.relu(out)

        out = self.lin5_e(out)
        out = self.bn5(out)
        out = torch.relu(out)

        return out

    def decode(self, x):

        out = self.lin1_d(x)
        out = self.bn6(out)
        out = torch.relu(out)

        out = self.lin2_d(out)
        out = self.bn7(out)
        out = torch.relu(out)

        out = self.lin3_d(out)
        out = self.bn8(out)
        out = torch.relu(out)

        out = self.lin4_d(out)
        out = self.bn9(out)
        out = torch.relu(out)


        out = self.lin5_d(out)

        return out

    def forward(self, x):

        out = self.encode(x)
        out = self.decode(out)

        return out


    def number_of_params(self):
         print('Numer of network paramteres:')
         print(sum(p.numel() for p in self.parameters()))



class VariationalEncoder(nn.Module):
    def __init__(self, N_input, latent_dims, hidden_dim, device='cuda'):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(N_input, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dims)
        self.linear3 = nn.Linear(hidden_dim, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0
        self.device = device

    def forward(self, x):
        #x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape).to(self.device)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma+np.abs(np.finfo(np.float32).eps)) - 1/2).sum() + np.abs(np.finfo(np.float32).eps)
        return z

class VariationalDecoder(nn.Module):
    def __init__(self, N_output, latent_dims, hidden_dim):
        super(VariationalDecoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, N_output)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        #z = torch.sigmoid(self.linear2(z))
        z = self.linear2(z)
        return z#z.reshape((-1, 1, 28, 28))


class VariationalAutoencoder(nn.Module):
    def __init__(self, N_input, latent_dims, hidden_dim, device = 'cpu'):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(N_input, latent_dims, hidden_dim, device)
        self.decoder = VariationalDecoder(N_input, latent_dims, hidden_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)