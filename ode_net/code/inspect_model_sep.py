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

#torch.set_num_threads(4) #CHANGE THIS!
    
sums_model = torch.load('/home/ubuntu/lottery_tickets_phoenix/ode_net/code/output/_pretrained_best_model/best_val_model_sums.pt')
prods_model = torch.load('/home/ubuntu/lottery_tickets_phoenix/ode_net/code/output/_pretrained_best_model/best_val_model_prods.pt')
alpha_comb_sums = torch.load('/home/ubuntu/lottery_tickets_phoenix/ode_net/code/output/_pretrained_best_model/best_val_model_alpha_comb_sums.pt')
alpha_comb_prods = torch.load('/home/ubuntu/lottery_tickets_phoenix/ode_net/code/output/_pretrained_best_model/best_val_model_alpha_comb_prods.pt')
gene_mult = torch.load('/home/ubuntu/lottery_tickets_phoenix/ode_net/code/output/_pretrained_best_model/best_val_model_gene_multipliers.pt')

Wo_sums = np.transpose(sums_model.linear_out.weight.detach().numpy())
Bo_sums = np.transpose(sums_model.linear_out.bias.detach().numpy())
Wo_prods = np.transpose(prods_model.linear_out.weight.detach().numpy())
Bo_prods = np.transpose(prods_model.linear_out.bias.detach().numpy())
alpha_comb_sums = np.transpose(alpha_comb_sums.linear_out.weight.detach().numpy())
alpha_comb_prods = np.transpose(alpha_comb_prods.linear_out.weight.detach().numpy())
gene_mult = np.transpose(torch.relu(gene_mult.detach()).numpy())


np.savetxt("/home/ubuntu/lottery_tickets_phoenix/ode_net/code/model_inspect/wo_sums.csv", Wo_sums, delimiter=",")
np.savetxt("/home/ubuntu/lottery_tickets_phoenix/ode_net/code/model_inspect/bo_sums.csv", Bo_sums, delimiter=",")
np.savetxt("/home/ubuntu/lottery_tickets_phoenix/ode_net/code/model_inspect/wo_prods.csv", Wo_prods, delimiter=",")
np.savetxt("/home/ubuntu/lottery_tickets_phoenix/ode_net/code/model_inspect/bo_prods.csv", Bo_prods, delimiter=",")
np.savetxt("/home/ubuntu/lottery_tickets_phoenix/ode_net/code/model_inspect/alpha_comb.csv", np.vstack((alpha_comb_sums, alpha_comb_prods)), delimiter=",")
np.savetxt("/home/ubuntu/lottery_tickets_phoenix/ode_net/code/model_inspect/gene_mult.csv", gene_mult, delimiter=",")
