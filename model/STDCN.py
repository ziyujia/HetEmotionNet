import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.set_device(0)


class permute(nn.Module):
    def __init__(self):
        super(permute, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)


class permute01(nn.Module):
    def __init__(self):
        super(permute01, self).__init__()

    def forward(self, x):
        return x.permute(1, 2, 0)


class reshape(nn.Module):
    def __init__(self, *args):
        super(reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.contiguous().view(self.shape)


class STDCN_with_GRU(nn.Module):
    def __init__(self, num_f, in_dim_with_c, out_dim_with_c, num_channel):
        super(STDCN_with_GRU, self).__init__()
        # Number of channels before entering GRU
        self.num_f = num_f
        self.in_dim = in_dim_with_c
        self.out_dim = out_dim_with_c
        # Number of graph channels
        self.num_channels = num_channel
        self.per1 = permute()
        self.per2 = permute()
        self.per0 = permute01()
        self.gru = nn.GRU(input_size=self.in_dim, hidden_size=self.out_dim, num_layers=1,
                          batch_first=True, bidirectional=True)
        self.BN = torch.nn.BatchNorm1d(self.num_f)
        # GCN weight
        self.weight = nn.Parameter(torch.Tensor(1, 1))
        self.soft = nn.Parameter(torch.Tensor(self.num_channels))
        self.reset_parameters()

    def norm(self, H, add=True):
        # H shape（b,n,n）
        if add == False:
            H = H * ((torch.eye(H.shape[1]) == 0).type(torch.FloatTensor)).unsqueeze(0)
        else:
            H = H * ((torch.eye(H.shape[1]) == 0).type(torch.FloatTensor)).unsqueeze(0).to(device) + torch.eye(
                H.shape[1]).type(torch.FloatTensor).unsqueeze(0).to(device)
        deg = torch.sum(H, dim=-1)
        # deg shape (b,n)
        deg_inv = deg.pow(-1 / 2)
        deg_inv = deg_inv.view((deg_inv.shape[0], deg_inv.shape[1], 1)) * torch.eye(H.shape[1]).type(
            torch.FloatTensor).unsqueeze(0).to(device)
        # deg_inv shape (b,n,n)
        H = torch.bmm(deg_inv, H)
        H = torch.bmm(H, deg_inv)
        H = torch.tensor(H, dtype=torch.float32).to(device)
        return H

    def reset_parameters(self):
        nn.init.constant_(self.weight, 10)
        nn.init.constant_(self.soft, 1 / self.num_channels)

    def gcn_conv(self, X, H):
        # X shape (B,N,F)
        # H shape (B,N,N)

        for i in range(X.shape[-1]):
            if i == 0:
                out = torch.bmm(H, X.unsqueeze(-2)[:, :, :, i])
                out = torch.matmul(out, self.weight)
            else:
                out = torch.cat((out, torch.matmul(torch.bmm(H, X.unsqueeze(-2)[:, :, :, i]), self.weight)),
                                dim=-1)
        return out

    def forward(self, X, H):
        # Spatial
        # H shape (b,c,n,n)
        # x shape (b,n,in)
        for i in range(self.num_channels):
            if i == 0:
                X_ = F.leaky_relu(self.gcn_conv(X, self.norm(H[:, i, :, :])))
                X_ = X_ * self.soft[i]
            else:
                X_tmp = F.leaky_relu(self.gcn_conv(X, self.norm(H[:, i, :, :])))
                X_tmp = X_tmp * self.soft[i]
                X_ = X_ + X_tmp
        # x shape (b,n,f)
        # temporal
        X_ = self.per1(X_)
        X_, hn = self.gru(X_)
        X_ = self.BN(X_)
        X_ = self.per2(X_)
        # x_ shape (b,NEWn,out_channel)
        return X_
