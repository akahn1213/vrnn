import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 

class VRNN(nn.Module):
  def __init__(self, x_dim, hlv_dim, h_dim, z_dim, n_layers, train_hlvs, bias=False):
    super(VRNN, self).__init__()

    self.x_dim = x_dim
    self.hlv_dim = hlv_dim
    self.h_dim = h_dim
    self.z_dim = z_dim
    self.n_layers = n_layers
    self.train_hlvs = train_hlvs

    #feature-extracting transformations
    self.phi_x = nn.Sequential(
      nn.Linear(x_dim, h_dim),
      nn.ReLU(),
      nn.Linear(h_dim, h_dim),
      nn.ReLU()).cuda()
    self.phi_z = nn.Sequential(
      nn.Linear(z_dim, h_dim),
      nn.ReLU()).cuda()

    #encoder
    if(self.train_hlvs):
      self.enc = nn.Sequential(
        nn.Linear(h_dim + h_dim + hlv_dim, h_dim),
        nn.ReLU(),
        nn.Linear(h_dim, h_dim),
        nn.ReLU()).cuda()
    else:
      self.enc = nn.Sequential(
        nn.Linear(h_dim + h_dim, h_dim),
        nn.ReLU(),
        nn.Linear(h_dim, h_dim),
        nn.ReLU()).cuda()
    self.enc_mean = nn.Linear(h_dim, z_dim).cuda()
    self.enc_std = nn.Sequential(
      nn.Linear(h_dim, z_dim),
      nn.Softplus()).cuda()

    #prior
    self.prior = nn.Sequential(
      nn.Linear(h_dim, h_dim),
      nn.ReLU()).cuda()
    self.prior_mean = nn.Linear(h_dim, z_dim).cuda()
    self.prior_std = nn.Sequential(
      nn.Linear(h_dim, z_dim),
      nn.Softplus()).cuda()

    #decoder
    if(self.train_hlvs):
      self.dec = nn.Sequential(
        nn.Linear(h_dim + h_dim + hlv_dim, h_dim),
        nn.ReLU(),
        nn.Linear(h_dim, h_dim),
        nn.ReLU()).cuda()
    else:
      self.dec = nn.Sequential(
        nn.Linear(h_dim + h_dim, h_dim),
        nn.ReLU(),
        nn.Linear(h_dim, h_dim),
        nn.ReLU()).cuda()
    self.dec_mean = nn.Sequential(
      nn.Linear(h_dim, x_dim)).cuda()
      

    #recurrence
    if(self.train_hlvs):
      self.rnn = nn.GRU(h_dim + h_dim + hlv_dim, h_dim, n_layers, bias).cuda()
    else:
      self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias).cuda()


  def forward(self, x, y, x_avg, kl_weight):

    if x.is_cuda: device='cuda'
    else: device='cpu'


    all_enc_mean, all_enc_std = [], []
    #all_dec_mean, all_dec_std = [], []
    all_dec_mean = []
    all_dec_mean = torch.empty(x.size(0), x.size(1), x.size(2))
    #kld_loss = 0
    kld_loss = torch.zeros([x.size(1)]).to(device)
    nll_loss = 0
    mseloss = torch.nn.MSELoss()
    l1loss = torch.nn.L1Loss()


    #x_avg = torch.mean(x, 1)
  

    h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(device)
    for t in range(x.size(0)):
      
      phi_x_t = self.phi_x(x[t]).to(device)
      #encoder
      if(self.train_hlvs):
        enc_t = self.enc(torch.cat([phi_x_t, h[-1], y], 1)).to(device)
      else:      
        enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1)).to(device)
      
      enc_mean_t = self.enc_mean(enc_t).to(device)
      enc_std_t = self.enc_std(enc_t).to(device)

      #prior
      prior_t = self.prior(h[-1]).to(device)
      prior_mean_t = self.prior_mean(prior_t).to(device)
      prior_std_t = self.prior_std(prior_t).to(device)

      #sampling and reparameterization
      z_t = self.reparam(enc_mean_t, enc_std_t).to(device)
      phi_z_t = self.phi_z(z_t).to(device)

      #decoder
      if(self.train_hlvs):
        dec_t = self.dec(torch.cat([phi_z_t, h[-1], y], 1)).to(device)
      else:
        dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1)).to(device)
      dec_mean_t = self.dec_mean(dec_t).to(device)


      #recurrence
      if(self.train_hlvs):
        _, h = self.rnn(torch.cat([phi_x_t, phi_z_t, y], 1).unsqueeze(0), h)
      else:
        _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

      #computing losses
      kld_loss += self.kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t).to(device)*x_avg[t]
      nll_loss += mseloss(dec_mean_t, x[t]).to(device)

      all_enc_std.append(enc_std_t)
      all_enc_mean.append(enc_mean_t)
      all_dec_mean[t] = dec_mean_t.detach()

    kld_loss /= (x.size(0))
    nll_loss /= x.size(0)
    loss = kl_weight*torch.mean(kld_loss) + nll_loss


    return kld_loss, nll_loss, loss, all_dec_mean

  def reparam(self, mean, std):
    eps = torch.FloatTensor(std.size()).normal_().to(mean.device)
    eps = Variable(eps)
    return eps.mul(std).add_(mean).to(mean.device)


  def kld_gauss(self, mean_1, std_1, mean_2, std_2):
    kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
      (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
      std_2.pow(2) - 1)
    return  0.5 * torch.sum(kld_element, dim=1)

