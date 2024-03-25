import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import copy
#import scanpy as sc
#import seaborn as sns

import sys 
import os
sys.path.append('')
from L0_regularization.l0_layers import L0Dense
from torch.nn.parameter import Parameter

from torchdiffeq import odeint

class initial_position(nn.Module):
    
    def __init__(self, dim, nhidden):
        super(initial_position, self).__init__()
        
    def forward(self, x): 
        
        x0 = torch.mean(x,axis=0)
        for g in range (x.shape[2]):
            zscore = (x[...,g] - x[...,g].mean()) / torch.sqrt(x[...,g].var())
            zscore = torch.where(torch.isnan(zscore), torch.zeros_like(zscore), zscore)
            x0[:,g] = torch.mean(x[...,g][zscore<3])
        return x0  

class ODEBlock(nn.Module):

    def __init__(self, odefunc, dim, tol, method):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.dim = dim
        self.tol = tol
        self.method = method
        
    def set_times(self,times):
        self.integration_times = times
        
    def forward(self, x):
        integrated = odeint(self.odefunc, x, self.integration_times, rtol = self.tol, atol= self.tol, method = self.method)
        out = torch.empty(len(self.integration_times), 1, self.dim)
        for i in range(len(self.integration_times)):
            out[i] = integrated[i][0,:self.dim]
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def array_tensor(X,T,X0):
    tX = []
    tT = []
    tX0 = []
    for i in range(len(X)):
        data = X[i]
        t = T[i]
        x0 = X0[i]
        data_dim = data.shape[1]
        samp_ts = torch.empty(data.shape[0]).float()   
        for i in range (data.shape[0]):
            samp_ts[i]= torch.tensor(float(t[i]))
        z0= torch.empty((x0.shape[0],1,data_dim)).float().to(device)
        z = torch.empty((data.shape[0], 1, data_dim)).float().to(device)
        for j in range(data_dim):
            for i in range(x0.shape[0]):    
                z0[i,0,j] = torch.tensor(float(x0[i,j]))
            for i in range(data.shape[0]):
                z[i,0,j] = torch.tensor(float(data[i, j]))
        tX.append(z)
        tT.append(samp_ts)
        tX0.append(z0)
    return tX,tT,tX0    

def _remove_cells(data,time,idx1,idx2):
    for i in range(len(data)):
        data[i] = data[i][idx1:idx2]
        time[i] = time[i][idx1:idx2]
        assert(len(data[i])==len(time[i]))
    return data,time

def _nan2zero(x):
    return np.where(np.isnan(x), np.zeros_like(x), x)

def _mean(data):
    for i in range(len(data)):
        data[i] = (data[i] - np.min(data[i],axis=0))/ (np.max(data[i],axis=0) - np.min(data[i],axis=0))
        data[i] =_nan2zero(data[i])
    return data


def _initial_cond(data,n):
    init = []
    for i in range(len(data)):
        #print(i)
        #print(data[i][:n,:].shape)
        init.append(data[i][:n,:])
    return init    

device = torch.device('cuda:' + str(args['gpu']) if torch.cuda.is_available() else 'cpu')
filename = 'models/PathReg/'
try:
    os.makedirs('./'+filename)
except FileExistsError:
    pass    

class SoftsignMod(nn.Module):
    def __init__(self):
        super().__init__() # init the base class
        #self.shift = shift

    def forward(self, input):
        shift = 0.5
        shifted_input =(input- shift) #500*
        abs_shifted_input = torch.abs(shifted_input)
        return(shifted_input/(1+abs_shifted_input))   #1/500*
    

class LogShiftedSoftSignMod(nn.Module):
    def __init__(self):
        super().__init__() # init the base class

    def forward(self, input):
        shift = 0.5
        shifted_input =  input - shift
        abs_shifted_input = torch.abs(shifted_input)
        soft_sign_mod = shifted_input/(1+abs_shifted_input)
        return(torch.log1p(soft_sign_mod))      
    
class ExponentialLayer(nn.Module):
    def forward(self, x):
        return torch.exp(x)    
    

class L0_MLP(nn.Module):
    def __init__(self, input_dim, layer_dims=(100, 100), N=50000, beta_ema=0.999,
                 weight_decay=0., my_lambda = 1, local_rep=False, temperature=2./3.):
        super(L0_MLP, self).__init__()
        self.layer_dims = layer_dims
        self.input_dim = input_dim
        self.N = N
        self.beta_ema = beta_ema
        self.weight_decay = self.N * weight_decay
        self.lambdas = my_lambda


        layers_sums = []
        layers_sums += [SoftsignMod()]
        layers_sums+= [L0Dense(input_dim, layer_dims[0], droprate_init=0.2, weight_decay=self.weight_decay,
                               lamba=my_lambda, local_rep=local_rep, temperature=temperature, bias = True)]
        layers_sums+= [L0Dense(layer_dims[0], input_dim, droprate_init=0.5, weight_decay=self.weight_decay,
                               lamba=my_lambda, local_rep=local_rep, temperature=temperature, bias = False)]

        self.output_sums = nn.Sequential(*layers_sums)
        
        layers_prods = []
        layers_prods += [LogShiftedSoftSignMod()]
        layers_prods+= [L0Dense(input_dim, layer_dims[0], droprate_init=0.2, weight_decay=self.weight_decay,
                               lamba=my_lambda, local_rep=local_rep, temperature=temperature,  bias = True)]
        layers_prods += [ExponentialLayer()]
        layers_prods+= [L0Dense(layer_dims[0], input_dim, droprate_init=0.5, weight_decay=self.weight_decay,
                               lamba=my_lambda, local_rep=local_rep, temperature=temperature,  bias = False)]

        self.output_prods = nn.Sequential(*layers_prods)

        self.gene_multipliers = Parameter(torch.rand(1,input_dim), requires_grad= True)

        self.layers = []
        for m in self.modules():
            if isinstance(m, L0Dense):
                self.layers.append(m)
        
        self.nfe = 0
            
        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = copy.deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.
        self.nfe = 0

       
    def forward(self, t, x):  
        self.nfe += 1
        joint = self.output_prods(x) + self.output_sums(x)
        return torch.relu(self.gene_multipliers)* (joint - x) #added minus x
    
    def regularization(self):
        regularization = 0.
        for layer in self.l0_layers:
            regularization += - (1. / self.N) * layer.regularization()
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.l0_layers:
            e_fl, e_l0 = layer.count_expected_flops_and_l0()
            expected_flops += e_fl
            expected_l0 += e_l0
        return expected_flops, expected_l0

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = copy.deepcopy(list(p.data for p in self.parameters()))
        return params
    

def PathReg(pathreg_model):
    Wo_sums = pathreg_model[1].odefunc.output_sums[1].sample_weights_ones()
    alpha_comb_sums = pathreg_model[1].odefunc.output_sums[2].sample_weights_ones()
    Wo_prods = pathreg_model[1].odefunc.output_prods[1].sample_weights_ones()
    alpha_comb_prods = pathreg_model[1].odefunc.output_prods[3].sample_weights_ones()
    
    WM_sums = torch.matmul(torch.abs(Wo_sums), torch.abs(alpha_comb_sums))
    WM_prods = torch.matmul(torch.abs(Wo_prods), torch.abs(alpha_comb_prods))
    
    return torch.mean(torch.abs(WM_sums) + torch.abs(WM_prods))

def L1(pathreg_model):
    Wo_sums = pathreg_model[1].odefunc.output_sums[1].weights
    alpha_comb_sums = pathreg_model[1].odefunc.output_sums[2].weights
    Wo_prods = pathreg_model[1].odefunc.output_prods[1].weights
    alpha_comb_prods = pathreg_model[1].odefunc.output_prods[3].weights
    
    WM_sums = torch.matmul(torch.abs(Wo_sums), torch.abs(alpha_comb_sums))
    WM_prods = torch.matmul(torch.abs(Wo_prods), torch.abs(alpha_comb_prods))
    
    return torch.mean(torch.abs(WM_sums) + torch.abs(WM_prods))







