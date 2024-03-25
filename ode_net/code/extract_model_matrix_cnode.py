# Imports
import sys
import os
import argparse
import inspect
from datetime import datetime
import numpy as np
from tqdm import tqdm
from math import ceil
from time import perf_counter, process_time

import torch
import torch.optim as optim
import torch.nn as nn

try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint

#from datagenerator import DataGenerator
from datahandler import DataHandler
from ode_net.code.PHX_base_model import ODENet
from read_config import read_arguments_from_file
from visualization_inte import *
import matplotlib.pyplot as plt

from pathreg_helper_PHX import L0_MLP, initial_position, ODEBlock, PathReg, L1


def average_zeros_across_arrays(*arrays):
    num_zeros = 0
    total_elements = 0
    
    for array in arrays:
        num_zeros += np.sum(np.abs(array) ==0)
        total_elements += array.size
    
    if total_elements == 0:
        return 0  # Avoid division by zero if the total number of elements is zero
    
    average_zeros = num_zeros / total_elements
    return average_zeros

#torch.set_num_threads(4) #CHANGE THIS!
pathreg_model = torch.load('/home/ubuntu/lottery_tickets_phoenix/ode_net/code/output/_pretrained_best_model/best_val_model.pt')

with torch.no_grad():
    

    # Check if Wo_prods and alpha_comb_prods exist
    if hasattr(pathreg_model[1].odefunc, 'output_prods'):
        print("prods model present!")
        Wo_sums = pathreg_model[1].odefunc.output_sums[1].weights.detach().numpy()
        alpha_comb_sums = pathreg_model[1].odefunc.output_sums[2].weights.detach().numpy()
        Bo_sums = np.transpose(pathreg_model[1].odefunc.output_sums[1].bias.detach().numpy())
        Wo_prods = pathreg_model[1].odefunc.output_prods[1].weights.detach().numpy()
        alpha_comb_prods = pathreg_model[1].odefunc.output_prods[3].weights.detach().numpy()
        Bo_prods = np.transpose(pathreg_model[1].odefunc.output_prods[1].bias.detach().numpy())
        effects_mat = np.matmul(Wo_sums, alpha_comb_sums) + np.matmul(Wo_prods, alpha_comb_prods)

    else:
        Wo_sums = pathreg_model[1].odefunc.output[1].weights.detach().numpy()
        alpha_comb_sums = pathreg_model[1].odefunc.output[2].weights.detach().numpy()
        #Bo_sums = np.transpose(sums_model.linear_out.bias.detach().numpy())
        effects_mat = np.matmul(Wo_sums,alpha_comb_sums) 

    gene_mult = np.transpose(torch.relu(pathreg_model[1].odefunc.gene_multipliers.detach()).numpy())

#effects_mat[np.abs(effects_mat)<1e-5] = 0.
effects_mat =   effects_mat * np.transpose(gene_mult)

if hasattr(pathreg_model[1].odefunc, 'output_prods'):
    my_sparsity = average_zeros_across_arrays(Wo_prods, Wo_sums, alpha_comb_prods* np.transpose(gene_mult), alpha_comb_sums* np.transpose(gene_mult))
else:
    my_sparsity = average_zeros_across_arrays(Wo_sums, alpha_comb_sums)        
print("Sparsity (incl. multipliers) : {:.2%}".format(my_sparsity))

#np.savetxt("/home/ubuntu/lottery_tickets_phoenix/ode_net/code/model_inspect/effects_mat_pathreg.csv", effects_mat, delimiter=",")

np.savetxt("/home/ubuntu/lottery_tickets_phoenix/ode_net/code/model_inspect/wo_sums.csv", Wo_sums, delimiter=",")
np.savetxt("/home/ubuntu/lottery_tickets_phoenix/ode_net/code/model_inspect/bo_sums.csv", Bo_sums, delimiter=",")
if hasattr(pathreg_model[1].odefunc, 'output_prods'):
    np.savetxt("/home/ubuntu/lottery_tickets_phoenix/ode_net/code/model_inspect/wo_prods.csv", Wo_prods, delimiter=",")
    np.savetxt("/home/ubuntu/lottery_tickets_phoenix/ode_net/code/model_inspect/bo_prods.csv", Bo_prods, delimiter=",")
    np.savetxt("/home/ubuntu/lottery_tickets_phoenix/ode_net/code/model_inspect/alpha_comb.csv", np.vstack((alpha_comb_sums, alpha_comb_prods)), delimiter=",")
else:
    np.savetxt("/home/ubuntu/lottery_tickets_phoenix/ode_net/code/model_inspect/wo_prods.csv", Wo_sums, delimiter=",")
    np.savetxt("/home/ubuntu/lottery_tickets_phoenix/ode_net/code/model_inspect/alpha_comb.csv", np.vstack((alpha_comb_sums, alpha_comb_sums)), delimiter=",")

np.savetxt("/home/ubuntu/lottery_tickets_phoenix/ode_net/code/model_inspect/gene_mult.csv", gene_mult, delimiter=",")



'''
with torch.no_grad():
    for i, layer in enumerate(pathreg_model[1].odefunc.layers):
        samp_weights = layer.weights
        if i ==0:
            all_weights = samp_weights.flatten()
            WM = samp_weights
        else:
            all_weights = torch.cat((all_weights,samp_weights.flatten()))
            WM = torch.matmul(WM,samp_weights)
            
WM = WM.detach().numpy()
WM[np.abs(WM)<1e-5] = 0.
model_spar = (torch.abs(all_weights)<1e-5).sum() / all_weights.shape[0]
'''



