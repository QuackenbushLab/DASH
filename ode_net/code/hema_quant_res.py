import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint
import sys
import os
import argparse


from datahandler import DataHandler
from read_config import read_arguments_from_file
from ode_net.code.PHX_base_model import ODENet

from pathreg_helper_PHX import L0_MLP, initial_position, ODEBlock, PathReg, L1
from matplotlib.ticker import FuncFormatter

def validation_inhouse(odenet, data_handler, method, explicit_time):
    print(data_handler.val_set_indx)
    data, t, target_full, n_val = data_handler.get_validation_set()
    #not_nan_idx = [i for i in range(target_full.shape[0]) if not torch.isnan(target_full[i][0][0]).item()]
    #data = data[not_nan_idx]
    #t = t[not_nan_idx]
    #target_full = target_full[not_nan_idx]
    #n_val = len(not_nan_idx)
    #odenet.eval()
    with torch.no_grad():
        predictions = []
        targets = []
        # For now we have to loop through manually, their implementation of odenet can only take fixed time lists.
        for index, (time, batch_point, target_point) in enumerate(zip(t, data, target_full)):
            #IH: 9/10/2021 - added these to handle unequal time availability 
            #comment these out when not requiring nan-value checking
            
            not_nan_idx = [i for i in range(len(time)) if not torch.isnan(time[i])]
            time = time[not_nan_idx]
            not_nan_idx.pop()
            batch_point = batch_point[not_nan_idx]
            target_point = target_point[not_nan_idx]
            
            # Do prediction
            predictions.append(odeint(odenet, batch_point, time, method=method)[1])
            targets.append(target_point) #IH comment
            #predictions[index, :, :] = odeint(odenet, batch_point[0], time, method=method)[1:]

        # Calculate validation loss
        predictions = torch.cat(predictions, dim = 0).to(data_handler.device) #IH addition
        targets = torch.cat(targets, dim = 0).to(data_handler.device) 
        #loss = torch.mean((predictions - targets) ** 2) #regulated_loss(predictions, target, t, val = True)

        all_losses = (predictions - targets)**2
        loss_sum = torch.sum(all_losses)
        num_points = all_losses.numel()
        #print("gene_mult_mean =", torch.mean(torch.relu(odenet.gene_multipliers) + 0.1))
        
    return [loss_sum, num_points]


def validation_pathreg(pathreg_model, data_handler, method, explicit_time, num_reps = 1):
    data, t, target_full, n_val = data_handler.get_validation_set()
    if method == "trajectory":
        False

    init_bias_y = data_handler.init_bias_y
    #odenet.eval()
    with torch.no_grad():
        predictions = []
        targets = []
        # For now we have to loop through manually, their implementation of odenet can only take fixed time lists.
        for index, (time, batch_point, target_point) in enumerate(zip(t, data, target_full)):
            pathreg_model[1].set_times(time)
            temp_preds = torch.empty((num_reps,1, data_handler.dim), device=data_handler.device)

            # Repeat the prediction operation 100 times
            for i in range(num_reps):
                pred_z = pathreg_model(batch_point.unsqueeze(1))
                temp_preds[i] = pred_z[1]

            # Calculate the average along the first dimension (axis=0)
            average_pred_z = torch.mean(temp_preds, dim=0)

            predictions.append(average_pred_z)
            targets.append(target_point) #IH comment
            #predictions[index, :, :] = odeint(odenet, batch_point[0], time, method=method)[1:]

        # Calculate validation loss
        predictions = torch.cat(predictions, dim = 0).to(data_handler.device) #IH addition
        targets = torch.cat(targets, dim = 0).to(data_handler.device) 
        #loss = torch.mean((predictions - targets) ** 2) #regulated_loss(predictions, target, t, val = True)
        all_losses = (predictions - targets)**2
        loss_sum = torch.sum(all_losses)
        num_points = all_losses.numel()
        #print("gene_mult_mean =", torch.mean(torch.relu(odenet.gene_multipliers) + 0.1))
        
    return [loss_sum, num_points]

#torch.set_num_threads(4) #CHANGE THIS!
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



for lineage in ["eryth", "mono", "B"]:

    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--settings', type=str, default='config_hema.cfg')
    clean_name =  "hema_{}_529genes_3testsamples".format(lineage)
    parser.add_argument('--data', type=str, default='/home/ubuntu/lottery_tickets_phoenix/hema_data/clean_data/{}.csv'.format(clean_name))
    args = parser.parse_args()
    settings = read_arguments_from_file(args.settings)

    

    for my_model in ["pathreg", "blind_pruning","lambda_pruning"]:
        
        tot_loss = 0
        tot_points = 0
 
        if my_model != "pathreg":
            
            sums_model = torch.load('/home/ubuntu/lottery_tickets_phoenix/all_manuscript_models/hema_data_{}/{}/best_val_model_sums.pt'.format(lineage, my_model))
            prods_model = torch.load('/home/ubuntu/lottery_tickets_phoenix/all_manuscript_models/hema_data_{}/{}/best_val_model_prods.pt'.format(lineage, my_model))
            alpha_comb_sums = torch.load('/home/ubuntu/lottery_tickets_phoenix/all_manuscript_models/hema_data_{}/{}/best_val_model_alpha_comb_sums.pt'.format(lineage, my_model))
            alpha_comb_prods = torch.load('/home/ubuntu/lottery_tickets_phoenix/all_manuscript_models/hema_data_{}/{}/best_val_model_alpha_comb_prods.pt'.format(lineage, my_model))
            gene_mult = torch.load('/home/ubuntu/lottery_tickets_phoenix/all_manuscript_models/hema_data_{}/{}/best_val_model_gene_multipliers.pt'.format(lineage, my_model))

            Wo_sums = np.transpose(sums_model.linear_out.weight.detach().numpy())
            Bo_sums = np.transpose(sums_model.linear_out.bias.detach().numpy())
            Wo_prods = np.transpose(prods_model.linear_out.weight.detach().numpy())
            Bo_prods = np.transpose(prods_model.linear_out.bias.detach().numpy())
            alpha_comb_sums = np.transpose(alpha_comb_sums.linear_out.weight.detach().numpy())
            alpha_comb_prods = np.transpose(alpha_comb_prods.linear_out.weight.detach().numpy())
            gene_mult = np.transpose(torch.relu(gene_mult.detach()).numpy())

            
            for  hema_test_batch in range(3):
                data_handler = DataHandler.fromcsv(args.data, "cpu", val_split = 1, normalize=settings['normalize_data'], 
                        batch_type="trajectory", batch_time=settings['batch_time'], 
                        batch_time_frac=settings['batch_time_frac'],
                        noise = settings['noise'],
                        img_save_dir = None,
                        scale_expression = settings['scale_expression'],
                        log_scale = settings['log_scale'],
                        init_bias_y = settings['init_bias_y'],
                        my_fixed_val_for_hema = hema_test_batch)


                odenet = ODENet("cpu", data_handler.dim, explicit_time=settings['explicit_time'], neurons = settings['neurons_per_layer'], 
                log_scale = settings['log_scale'], init_bias_y = settings['init_bias_y'])
                odenet.float()    
                pretrained_model_file = '/home/ubuntu/lottery_tickets_phoenix/all_manuscript_models/hema_data_{}/{}/best_val_model.pt'.format(lineage, my_model)
                odenet.inherit_params(pretrained_model_file)

                val_loss_list = validation_inhouse(odenet, data_handler, settings['method'], settings['explicit_time'])
                tot_loss += val_loss_list[0]
                tot_points += val_loss_list[1]

        else:
            pathreg_model = torch.load('/home/ubuntu/lottery_tickets_phoenix/all_manuscript_models/hema_data_{}/{}/best_val_model.pt'.format(lineage, my_model))
            Wo_sums = pathreg_model[1].odefunc.output_sums[1].sample_weights().detach().numpy()
            alpha_comb_sums = pathreg_model[1].odefunc.output_sums[2].sample_weights().detach().numpy()
            Bo_sums = np.transpose(pathreg_model[1].odefunc.output_sums[1].bias.detach().numpy())
            Wo_prods = pathreg_model[1].odefunc.output_prods[1].sample_weights().detach().numpy()
            alpha_comb_prods = pathreg_model[1].odefunc.output_prods[3].sample_weights().detach().numpy()
            Bo_prods = np.transpose(pathreg_model[1].odefunc.output_prods[1].bias.detach().numpy())
            gene_mult = np.transpose(torch.relu(pathreg_model[1].odefunc.gene_multipliers.detach()).numpy())
            
            for  hema_test_batch in range(3):
                data_handler = DataHandler.fromcsv(args.data, "cpu", val_split = 1, normalize=settings['normalize_data'], 
                        batch_type="trajectory", batch_time=settings['batch_time'], 
                        batch_time_frac=settings['batch_time_frac'],
                        noise = settings['noise'],
                        img_save_dir = None,
                        scale_expression = settings['scale_expression'],
                        log_scale = settings['log_scale'],
                        init_bias_y = settings['init_bias_y'],
                        my_fixed_val_for_hema = hema_test_batch)

                val_loss_list =validation_pathreg(pathreg_model, data_handler, settings['method'], settings['explicit_time'], num_reps= 1)
                tot_loss += val_loss_list[0]
                tot_points += val_loss_list[1]


        effects_mat = np.matmul(Wo_sums,alpha_comb_sums) + np.matmul(Wo_prods,alpha_comb_prods)
        sparsity = average_zeros_across_arrays(Wo_prods, Wo_sums, alpha_comb_prods* np.transpose(gene_mult), alpha_comb_sums* np.transpose(gene_mult))
        #effects_mat[effects_mat != 0] = 1 # Set all non-zero elements to 1
        effects_mat =   gene_mult.T* effects_mat 

        # Count non-zero elements along each row
        out_degrees = np.count_nonzero(effects_mat, axis=1)

        # non_zero_counts will contain the number of non-zero elements for each row
        avg_out_deg = np.mean(out_degrees)
        print("tot param = {}".format(tot_points))
        print("lineage = {}, model = {}, sparsity = {:.2%}, avg out deg = {:.2f}, test_mse = {:.2E}".format(lineage,my_model,sparsity, avg_out_deg, tot_loss/tot_points))



