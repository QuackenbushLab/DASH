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
import torch.nn.utils.prune as prune

try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint

from datahandler import DataHandler
from ode_net.code.PHX_base_model import ODENet, get_mask_smallest_p_proportion
from read_config import read_arguments_from_file
from visualization_inte import *

#torch.set_num_threads(16) #CHANGE THIS!

def plot_MSE_new(epoch_so_far, training_loss, validation_loss, true_mean_losses, true_mean_losses_init_val_based, prior_losses, img_save_dir):
    
    # Create two subplots, one for the main MSE loss plot and one for the prior loss plot.
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    fig.set_size_inches(12, 6)

    ax1.plot(range(1, epoch_so_far + 1), training_loss, color="blue", label="Training loss")
    if len(validation_loss) > 0:
        ax1.plot(range(1, epoch_so_far + 1), validation_loss, color="red", label="Validation loss")

    ax2.plot(range(1, epoch_so_far + 1), prior_losses, color="magenta", label="Prior loss")

    ax1.set_yscale('log')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Error (MSE)")
    ax1.legend(loc='upper right')

    ax2.set_yscale('log')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Error (MSE)")
    ax2.set_title("Prior Loss")

    #plt.subplots_adjust(wspace=0.3)
    fig.tight_layout()
    plt.savefig("{}/MSE_loss.png".format(img_save_dir))
    np.savetxt('{}full_loss_info.csv'.format(output_root_dir), np.c_[training_loss, validation_loss, true_mean_losses, true_mean_losses_init_val_based], delimiter=',')

def plot_MSE(epoch_so_far, training_loss, validation_loss, true_mean_losses, true_mean_losses_init_val_based, prior_losses ,img_save_dir):
    plt.figure()
    plt.plot(range(1, epoch_so_far + 1), training_loss, color = "blue", label = "Training loss")
    if len(validation_loss) > 0:
        plt.plot(range(1, epoch_so_far + 1), validation_loss, color = "red", label = "Validation loss")
    #plt.plot(range(1, epoch_so_far + 1), true_mean_losses, color = "green", label = r'True $\mu$ loss')
    plt.plot(range(1, epoch_so_far + 1), prior_losses, color = "magenta", label = "Prior loss")
    
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.legend(loc='upper right')
    plt.ylabel("Error (MSE)")
    plt.savefig("{}/MSE_loss.png".format(img_save_dir))
    np.savetxt('{}full_loss_info.csv'.format(output_root_dir), np.c_[training_loss, validation_loss, true_mean_losses, true_mean_losses_init_val_based], delimiter=',')

def my_r_squared(output, target):
    x = output
    y = target
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    my_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return(my_corr**2)

def get_true_val_set_r2(odenet, data_handler, method, batch_type):
    data_pw, t_pw, target_pw = data_handler.get_true_mu_set_pairwise(val_only = True, batch_type =  batch_type)
    with torch.no_grad():
        predictions_pw = torch.zeros(data_pw.shape).to(data_handler.device)
        for index, (time, batch_point) in enumerate(zip(t_pw, data_pw)):
            predictions_pw[index, :, :] = odeint(odenet, batch_point, time, method=method)[1] 
        var_explained_pw = my_r_squared(predictions_pw, target_pw)
        true_val_mse = torch.mean((predictions_pw - target_pw)**2)
        
    #data, t, target = data_handler.get_true_mu_set_init_val_based(val_only = True) 
        #predictions = torch.zeros(target.shape).to(data_handler.device)
        #for index, (time, batch_point) in enumerate(zip(t, data)):
        #    predictions[index, :, :] = odeint(odenet, batch_point, time, method=method)[1:] 
        #var_explained_init_val_based = my_r_squared(predictions, target)
    
    return [var_explained_pw, true_val_mse]

def random_multiply(mat_torch):
  rand_torch = torch.rand(mat_torch.size())
  out_torch = torch.zeros(mat_torch.size())

  for i in range(mat_torch.size(0)):
    for j in range(mat_torch.size(1)):
      if rand_torch[i, j] > 0.50:
        out_torch[i, j] = mat_torch[i, j] * -1 #flip
      else:
        out_torch[i, j] = mat_torch[i, j] * 1 #keep

  return out_torch


def read_prior_matrix(prior_mat_file_loc, sparse = False, num_genes = 11165, absolute = False):
    if sparse == False: 
        mat = np.genfromtxt(prior_mat_file_loc,delimiter=',')
        mat_torch = torch.from_numpy(mat)
        mat_torch = mat_torch.float()
        if absolute:
            print("I AM SWITCHING ALL EDGE SIGNS to POSITIVE!")
            mat_torch = torch.abs(mat_torch)
    else: #when scaling up >10000
        mat = np.genfromtxt(prior_mat_file_loc,delimiter=',')
        sparse_mat = torch.sparse_coo_tensor([mat[:,0].astype(int)-1, mat[:,1].astype(int)-1], mat[:,2], ( num_genes,  num_genes))
        mat_torch = sparse_mat.to_dense().float()
        if absolute:
            print("I AM SWITCHING ALL EDGE SIGNS to POSITIVE!")
            mat_torch = torch.abs(mat_torch)
    return(mat_torch)

def normalize_values(my_tensor):
  """Normalizes the values of a PyTorch tensor.

  Args:
    tensor: The input tensor.

  Returns:
    A normalized tensor.
  """

  
  # Find the minimum and maximum absolute values in the tensor.
  min_val = torch.min(my_tensor)
  max_val = torch.max(my_tensor)

  # Normalize the absolute values to the range [0, 1].
  normalized_tensor = (my_tensor - min_val) / (max_val - min_val)

  return normalized_tensor




def validation(odenet, data_handler, method, explicit_time):
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
        loss = torch.mean((predictions - targets)**2)
        #print("gene_mult_mean =", torch.mean(torch.relu(odenet.gene_multipliers) + 0.1))
        
    return [loss, n_val]

def true_loss(odenet, data_handler, method):
    return [0,0]
    data, t, target = data_handler.get_true_mu_set() #tru_mu_prop = 1 (incorporate later)
    init_bias_y = data_handler.init_bias_y
    #odenet.eval()
    with torch.no_grad():
        predictions = torch.zeros(data.shape).to(data_handler.device)
        for index, (time, batch_point) in enumerate(zip(t, data)):
            predictions[index, :, :] = odeint(odenet, batch_point, time, method=method)[1] + init_bias_y #IH comment
        
        # Calculate true mean loss
        loss =  [torch.mean(torch.abs((predictions - target)/target)),torch.mean((predictions - target) ** 2)] #regulated_loss(predictions, target, t)
    return loss


def decrease_lr(opt, verbose, tot_epochs, epoch, lower_lr,  dec_lr_factor ):
    dir_string = "Decreasing"
    for param_group in opt.param_groups:
        param_group['lr'] = param_group['lr'] * dec_lr_factor
    if verbose:
        print(dir_string,"learning rate to: %f" % opt.param_groups[0]['lr'])


def reset_lr(opt, verbose, old_lr):
    dir_string = "Increasing"
    group_count = 0
    for param_group in opt.param_groups:
        group_count += 1 #gene_mult!
        param_group['lr'] = old_lr 
        if group_count == 6:
            param_group['lr'] = 5*old_lr 
        if verbose:
            print(dir_string,"learning rate to: %f" % param_group['lr'])

def setOptimizerLRScheduler(patience):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', 
                                                     factor=0.9, patience=patience, threshold=1e-07, 
                                                     threshold_mode='abs', cooldown=0, min_lr=4e-05, eps=1e-09, verbose=True)
    return scheduler



def training_step(odenet, data_handler, opt, method, batch_size, batch_for_prior, prior_grad, lambda_loss):
    batch, t, target = data_handler.get_batch(batch_size)
    
    '''
    not_nan_idx = [i for i in range(len(t)) if not torch.any(torch.isnan(t[i]))]
    t = t[not_nan_idx]
    batch = batch[not_nan_idx]
    target = target[not_nan_idx]
    '''

    init_bias_y = data_handler.init_bias_y
    opt.zero_grad()
    predictions = torch.zeros(batch.shape).to(data_handler.device)
    for index, (time, batch_point) in enumerate(zip(t, batch)):
        predictions[index, :, :] = odeint(odenet, batch_point, time, method= method  )[1] + init_bias_y #IH comment
    
    loss_data = torch.mean((predictions - target)**2) 
    
    pred_grad = odenet.prior_only_forward(t,batch_for_prior)
    loss_prior = torch.mean((pred_grad - prior_grad)**2)
    
    composed_loss = lambda_loss*loss_data + (1- lambda_loss)*loss_prior
    composed_loss.backward() #MOST EXPENSIVE STEP!
    opt.step()
    return [loss_data, loss_prior]




def _build_save_file_name(save_path, epochs):
    return '{}-{}-{}({};{})_{}_{}epochs'.format(str(datetime.now().year), str(datetime.now().month),
        str(datetime.now().day), str(datetime.now().hour), str(datetime.now().minute), save_path, epochs)

def save_model(odenet, folder, filename):
    odenet.save('{}{}.pt'.format(folder, filename))

parser = argparse.ArgumentParser('Testing')
parser.add_argument('--settings', type=str, default='config_hema.cfg')
clean_name =  "hema_B_529genes_2samples" 
parser.add_argument('--data', type=str, default='/home/ubuntu/lottery_tickets_phoenix/hema_data/clean_data/{}.csv'.format(clean_name))

args = parser.parse_args()

# Main function
if __name__ == "__main__":
    print('Setting recursion limit to 3000')
    sys.setrecursionlimit(3000)
    print('Loading settings from file {}'.format(args.settings))
    settings = read_arguments_from_file(args.settings)
    cleaned_file_name = clean_name
    save_file_name = _build_save_file_name(cleaned_file_name, settings['epochs'])

    if settings['debug']:
        print("********************IN DEBUG MODE!********************")
        save_file_name= '(DEBUG)' + save_file_name
    output_root_dir = '{}/{}/'.format(settings['output_dir'], save_file_name)

    img_save_dir = '{}img/'.format(output_root_dir)
    interm_models_save_dir = '{}interm_models/'.format(output_root_dir)
    #intermediate_models_dir = '{}intermediate_models/'.format(output_root_dir)

    # Create image and model save directory
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir, exist_ok=True)
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    if not os.path.exists(interm_models_save_dir):
        os.mkdir(interm_models_save_dir)

    # Save the settings for future reference
    with open('{}/settings.csv'.format(output_root_dir), 'w') as f:
        f.write("Setting,Value\n")
        for key in settings.keys():
            f.write("{},{}\n".format(key,settings[key]))

    # Use GPU if available
    if not settings['cpu']:
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        print("Trying to run on GPU -- cuda available: " + str(torch.cuda.is_available()))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Running on", device)
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        print("Running on CPU")
        device = 'cpu'
    
    data_handler = DataHandler.fromcsv(args.data, device, settings['val_split'], normalize=settings['normalize_data'], 
                                        batch_type=settings['batch_type'], batch_time=settings['batch_time'], 
                                        batch_time_frac=settings['batch_time_frac'],
                                        noise = settings['noise'],
                                        img_save_dir = img_save_dir,
                                        scale_expression = settings['scale_expression'],
                                        log_scale = settings['log_scale'],
                                        init_bias_y = settings['init_bias_y'])
    
    #Read in the prior matrix
    prior_mat_loc = '/home/ubuntu/lottery_tickets_phoenix/hema_data/clean_data/edge_prior_matrix_hema_529.csv'
    absolute_flag = True
    prior_mat = read_prior_matrix(prior_mat_loc, sparse = False, num_genes = data_handler.dim, absolute = absolute_flag)
    
    PPI_mat_loc = '/home/ubuntu/lottery_tickets_phoenix/hema_data/clean_data/PPI_matrix_hema_529.csv' #change soon
    PPI = read_prior_matrix(PPI_mat_loc, sparse = False, num_genes = data_handler.dim)
    PPI =  PPI / torch.sum(PPI)
    batch_for_prior = (torch.rand(10000,1,prior_mat.shape[0], device = data_handler.device) - 0.5)
    prior_grad = torch.matmul(batch_for_prior,prior_mat) #can be any model here that predicts the derivative
        
    noisy_PPI = PPI
    noisy_prior_mat = prior_mat
    
    loss_lambda_at_start = 1
    loss_lambda_at_end =  1#0.999
    

    masking_start_epoch = 3
    initial_hit_perc = 0.70
    num_epochs_till_mask = 10
    prune_perc = 0.10
    pruning_score_lambda_PPI = 0
    pruning_score_lambda_motif = 0 
    lr_schedule_patience = 3
    prop_force_to_zero_for_loaded_model = 0
    consider_multipliers = True

    odenet = ODENet(device, data_handler.dim, explicit_time=settings['explicit_time'], neurons = settings['neurons_per_layer'], 
                    log_scale = settings['log_scale'], init_bias_y = settings['init_bias_y'])
    odenet.float()


    my_current_custom_pruning_scores = {}
    my_current_custom_pruning_scores['net_prods.linear_out'] = torch.rand(settings['neurons_per_layer'], data_handler.dim)
    my_current_custom_pruning_scores['net_sums.linear_out'] = torch.rand(settings['neurons_per_layer'], data_handler.dim)
    my_current_custom_pruning_scores['net_alpha_combine_sums.linear_out'] = torch.rand(data_handler.dim, settings['neurons_per_layer'])
    my_current_custom_pruning_scores['net_alpha_combine_prods.linear_out'] = torch.rand(data_handler.dim, settings['neurons_per_layer'])

    
    if settings['pretrained_model']:
        pretrained_model_file = '/home/ubuntu/lottery_tickets_phoenix/ode_net/code/output/_pretrained_best_model/best_val_model.pt'
        odenet.inherit_params(pretrained_model_file)

    print('Using optimizer: {}'.format(settings['optimizer']))
    if settings['optimizer'] == 'rmsprop':
        opt = optim.RMSprop(odenet.parameters(), lr=settings['init_lr'], weight_decay=settings['weight_decay'])
    elif settings['optimizer'] == 'sgd':
        opt = optim.SGD(odenet.parameters(), lr=settings['init_lr'], weight_decay=settings['weight_decay'])
    elif settings['optimizer'] == 'adagrad':
        opt = optim.Adagrad(odenet.parameters(), lr=settings['init_lr'], weight_decay=settings['weight_decay'])
    else:
#       opt = optim.Adam(odenet.parameters(), lr=settings['init_lr'], weight_decay=settings['weight_decay'])
        num_gene = data_handler.dim
        opt = optim.Adam([
                {'params': odenet.net_sums.linear_out.weight}, 
                {'params': odenet.net_sums.linear_out.bias},
                {'params': odenet.net_prods.linear_out.weight},
                {'params': odenet.net_prods.linear_out.bias},
                {'params': odenet.net_alpha_combine_sums.linear_out.weight},
                {'params': odenet.net_alpha_combine_prods.linear_out.weight},
                {'params': odenet.gene_multipliers,'lr': 5*settings['init_lr']}
                
            ],  lr=settings['init_lr'], weight_decay=settings['weight_decay'])
    

        
    with open('{}/network.txt'.format(output_root_dir), 'w') as net_file:
        net_file.write(odenet.__str__())
        net_file.write('\n\n\n')
        net_file.write(inspect.getsource(ODENet.forward))
        if absolute_flag:
            net_file.write('\n')
            net_file.write("I AM SWITCHING ALL EDGE SIGNS to POSITIVE!")
        net_file.write('\n')
        net_file.write('lambda at start (first 5 epochs) = {}'.format(loss_lambda_at_start))
        net_file.write('\n')
        net_file.write('and then lambda = {}'.format(loss_lambda_at_end))
        net_file.write('\n')
        if prune_perc > 0 or initial_hit_perc > 0:
            net_file.write('causal lottery!')
            net_file.write('\n')
            net_file.write('doing PPI mask + T mask')
            if consider_multipliers: 
                net_file.write('\n')
                net_file.write('.....')
                net_file.write('Considering multipliers for final layer pruning')   
                net_file.write('.....')
            net_file.write('\n')
            net_file.write('pruning score lambda (PPI, Motif) = ({}, {})'.format(pruning_score_lambda_PPI, pruning_score_lambda_motif))
            net_file.write('\n')
            net_file.write('Initial hit = {} at epoch {}, then prune {} every {} epochs'.format(initial_hit_perc, masking_start_epoch, prune_perc, num_epochs_till_mask))
        else:
            net_file.write('No pruning!')
            net_file.write('\n')
        if settings['pretrained_model']:
            net_file.write('\n')
            net_file.write('LOADED in a pre-trained model but forced lowest {} perc of params to zero'.format(prop_force_to_zero_for_loaded_model*100))
            if consider_multipliers: 
                net_file.write('\n')
                net_file.write('.....')
                net_file.write('Considering multipliers for forced zeroing of pre-trained model')   
                net_file.write('.....')    
    # Init plot
    if settings['viz']:
        visualizer = Visualizator1D(data_handler, odenet, settings)

    # Training loop
    #batch_times = [] 
    epoch_times = []
    total_time = 0
    validation_loss = []
    training_loss = []
    prior_losses = []
    true_mean_losses = []
    true_mean_losses_init_val_based = []
    A_list = []

    min_loss = 0
    if settings['batch_type'] == 'single':
        iterations_in_epoch = ceil(data_handler.train_data_length / settings['batch_size'])
    elif settings['batch_type'] == 'trajectory':
        iterations_in_epoch = data_handler.train_data_length
    else:
        iterations_in_epoch = ceil(data_handler.train_data_length / settings['batch_size'])

    #quit()
    
    tot_epochs = settings['epochs']
    #viz_epochs = [round(tot_epochs*1/5), round(tot_epochs*2/5), round(tot_epochs*3/5), round(tot_epochs*4/5),tot_epochs]
    rep_epochs = [ 1, 5, 15, 30, 40, 50, 75, 100, 120, 150, 180, 200,220, 240, 260, 280, 300, 350, tot_epochs]
    viz_epochs = rep_epochs
    zeroth_drop_done = False
    first_drop_done = False 
    second_drop_done = False
    rep_epochs_train_losses = []
    rep_epochs_val_losses = []
    rep_epochs_mu_losses = []
    rep_epochs_time_so_far = []
    rep_epochs_so_far = []
    consec_epochs_failed = 0
    epochs_to_fail_to_terminate = 9999
    all_lrs_used = []

    

    scheduler = setOptimizerLRScheduler(patience=lr_schedule_patience)

    if settings['pretrained_model'] and prop_force_to_zero_for_loaded_model>0:
            total_pruned = 0
            total_params = 0
            for name, module in odenet.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if name in ['net_alpha_combine_sums.linear_out' , 'net_alpha_combine_prods.linear_out']:
                        if consider_multipliers:
                            print("Considering multipliers for zeroing of final layers")
                            current_NN_weights_abs = abs(module.weight.detach())
                            current_gene_mult_ReLU = torch.relu(odenet.gene_multipliers.detach().t())
                            this_loaded_module_prune_score = current_NN_weights_abs * current_gene_mult_ReLU
                            prune.l1_unstructured(module, name='weight', amount = prop_force_to_zero_for_loaded_model, importance_scores = this_loaded_module_prune_score) #CHANGE THIS!
                        else:
                            prune.l1_unstructured(module, name='weight', amount = prop_force_to_zero_for_loaded_model)        
                    else:
                        prune.l1_unstructured(module, name='weight', amount = prop_force_to_zero_for_loaded_model) 
                    
                    print("SETTING LOWEST {} % OF ELEMENTS in {} TO ZERO".format(prop_force_to_zero_for_loaded_model*100, name))
                    total_params += module.weight.nelement()
                    total_pruned += torch.sum(module.weight == 0)
            print("Updated mask based on prior! Current perc pruned: {:.2%}, num pruned: {}".format(total_pruned/total_params, total_pruned))
        
            
    if settings['viz']:
        with torch.no_grad():
            visualizer.visualize()
            visualizer.plot()
            visualizer.save(img_save_dir, 0)
    
    start_time = perf_counter()
    
    

    for epoch in range(1, tot_epochs + 1):
        print()
        print("[Running epoch {}/{}]".format(epoch, settings['epochs']))


        #Iterative magnitude pruning (IMP for lottery tickets)
        if (epoch == masking_start_epoch) or (epoch == masking_start_epoch +0 ) or (epoch >= masking_start_epoch and epoch < tot_epochs and epoch % num_epochs_till_mask in [0,0]):
            if epoch == masking_start_epoch or epoch == masking_start_epoch + 0:
                prune_this_epoch = initial_hit_perc
            else:
                prune_this_epoch = prune_perc

            total_pruned = 0
            total_params = 0

            for name, module in odenet.named_modules():
                if isinstance(module, torch.nn.Linear): # and (prune_perc > 0 or initial_hit_perc > 0)
                    #current_NN_weights = module.weight.detach()
                    if name == 'net_sums.linear_out': 
                        current_NN_weights_abs = abs(module.weight.detach()) #GRAD!.detach()
                        current_NN_weights_abs = current_NN_weights_abs/torch.sum(current_NN_weights_abs) #trying normalization this for PPI layers+
                        

                    elif name == 'net_prods.linear_out':
                        current_NN_weights_abs = torch.exp(module.weight.detach()) #GRAD! 
                        current_NN_weights_abs = current_NN_weights_abs/torch.sum(current_NN_weights_abs) #trying normalization this for PPI layers

                    elif name in ['net_alpha_combine_sums.linear_out','net_alpha_combine_prods.linear_out' ]: 
                        current_NN_weights_abs = abs(module.weight.detach()) #GRAD!.detach()
                        if consider_multipliers:
                            current_gene_mult_ReLU = torch.relu(odenet.gene_multipliers.detach().t())
                            current_NN_weights_abs = current_NN_weights_abs * current_gene_mult_ReLU
                        current_NN_weights_abs = current_NN_weights_abs/torch.sum(current_NN_weights_abs) #trying normalization this for PPI layers+
                    
                    if name in ['net_sums.linear_out','net_prods.linear_out'] and ((epoch == masking_start_epoch) or epoch % num_epochs_till_mask == 0): #name == 'net_prods.linear_out' or 
                        
                        mask_curr = my_current_custom_pruning_scores[name]
                        S_S_transpose_inv = torch.inverse(torch.matmul(mask_curr, torch.transpose(mask_curr,0,1)))
                        S_PPI = torch.matmul(mask_curr, noisy_PPI)
                        S_mask_best_guess = torch.matmul(S_S_transpose_inv, S_PPI)
                        
                        updated_score = pruning_score_lambda_PPI * torch.abs(S_mask_best_guess) + (1 - pruning_score_lambda_PPI) * current_NN_weights_abs
                        
                        prune.l1_unstructured(module, name='weight', amount=prune_this_epoch, importance_scores = updated_score)
                        my_current_custom_pruning_scores[name] = updated_score
                        
                    elif name in['net_alpha_combine_sums.linear_out', 'net_alpha_combine_prods.linear_out'] and ((epoch == masking_start_epoch +0 ) or (epoch % num_epochs_till_mask == 0)):

                        if name == 'net_alpha_combine_sums.linear_out':
                            incoming_mask_curr = my_current_custom_pruning_scores['net_sums.linear_out']
                        else:
                            incoming_mask_curr = my_current_custom_pruning_scores['net_prods.linear_out']
                        
                        T_tranpose_S_transpose = torch.transpose(torch.matmul(incoming_mask_curr,abs(noisy_prior_mat)),0,1)
                        S_S_transpose_inv = torch.inverse(torch.matmul(incoming_mask_curr, torch.transpose(incoming_mask_curr,0,1)))
                        C_mask_best_guess = torch.matmul(T_tranpose_S_transpose, S_S_transpose_inv)

                        
                        updated_score = pruning_score_lambda_motif * torch.abs(C_mask_best_guess)  + (1 - pruning_score_lambda_motif) * current_NN_weights_abs
                        
                        #updated_score = mask_T.contiguous()
                        
                        prune.l1_unstructured(module, name='weight', amount=prune_this_epoch, importance_scores = updated_score) # 
                        my_current_custom_pruning_scores[name] = updated_score
                        
                    
                    total_params += module.weight.nelement()
                    total_pruned += torch.sum(module.weight == 0)
                    
                    
                    
            print("Updated mask based on prior! Current perc pruned: {:.2%}, num pruned: {}".format(total_pruned/total_params, total_pruned))
            
            if prune_perc > 0 or initial_hit_perc > 0:
                reset_lr(opt, True, settings['init_lr']) #, 
                scheduler = setOptimizerLRScheduler(patience = lr_schedule_patience)

            
        start_epoch_time = perf_counter()
        iteration_counter = 1
        data_handler.reset_epoch()
        #visualizer.save(img_save_dir, epoch) #IH added to test
        this_epoch_total_train_loss = 0
        this_epoch_total_prior_loss = 0
        
        if epoch <= 3:
            loss_lambda = loss_lambda_at_start
        else:
            loss_lambda = loss_lambda_at_end    
        print("current loss_lambda =", loss_lambda)
        
        
        if settings['verbose']:
            pbar = tqdm(total=iterations_in_epoch, desc="Training loss:")
        while not data_handler.epoch_done:
            start_batch_time = perf_counter()
            
            loss_list = training_step(odenet, data_handler, opt, settings['method'], settings['batch_size'], batch_for_prior, prior_grad, loss_lambda)
            loss = loss_list[0]
            prior_loss = loss_list[1]
            #batch_times.append(perf_counter() - start_batch_time)

            this_epoch_total_train_loss += loss.item()
            this_epoch_total_prior_loss += prior_loss.item()
            # Print and update plots
            iteration_counter += 1

            if settings['verbose']:
                pbar.update(1)
                pbar.set_description("Training loss, Prior loss: {:.2E}, {:.2E}".format(loss.item(), prior_loss.item()))
        
        epoch_times.append(perf_counter() - start_epoch_time)

        #Epoch done, now handle training loss
        train_loss = this_epoch_total_train_loss/iterations_in_epoch
        training_loss.append(train_loss)
        prior_losses.append(this_epoch_total_prior_loss/iterations_in_epoch)
        #print("Overall training loss {:.5E}".format(train_loss))

        mu_loss = get_true_val_set_r2(odenet, data_handler, settings['method'], settings['batch_type'])
        #mu_loss = true_loss(odenet, data_handler, settings['method'])
        true_mean_losses.append(mu_loss[1])
        true_mean_losses_init_val_based.append(mu_loss[0])
        all_lrs_used.append(opt.param_groups[0]['lr'])
        
        if epoch == 1:
                min_train_loss = train_loss
        else:
            if train_loss < min_train_loss:
                min_train_loss = train_loss
                true_loss_of_min_train_model =  mu_loss[1]
                #save_model(odenet, output_root_dir, 'best_train_model')
        


        if settings['verbose']:
            pbar.close()

        if settings['solve_A']:
            A = solve_eq(odenet, settings['solve_eq_gridsize'], (-5, 5, 0, 10, -3, 3, -10, 10))
            A_list.append(A)
            print('A =\n{}'.format(A))

        #handle true-mu loss
       
        if data_handler.n_val > 0:
            val_loss_list = validation(odenet, data_handler, settings['method'], settings['explicit_time'])
            val_loss = val_loss_list[0]
            validation_loss.append(val_loss)
            if epoch == 1:
                min_val_loss = val_loss
                true_loss_of_min_val_model = mu_loss[1]
                print('Model improved, saving current model')
                best_vaL_model_so_far = odenet
                save_model(odenet, output_root_dir, 'best_val_model')
            else:
                if val_loss < min_val_loss:
                    consec_epochs_failed = 0
                    min_val_loss = val_loss
                    true_loss_of_min_val_model =  mu_loss[1]
                    #saving true-mean loss of best val model
                    print('Model improved, saving current model')
                    save_model(odenet, output_root_dir, 'best_val_model')
                else:
                    consec_epochs_failed = consec_epochs_failed + 1

                    
            print("Validation loss {:.5E}, using {} points".format(val_loss, val_loss_list[1]))
            scheduler.step(val_loss)

        print("Overall training loss {:.5E}".format(train_loss))

        print("True MSE of val traj (pairwise): {:.5E}".format(mu_loss[1]))
        print("True R^2 of val traj (pairwise): {:.2%}".format(mu_loss[0]))

            
        if (settings['viz'] and epoch in viz_epochs) or (settings['viz'] and epoch in rep_epochs) or (consec_epochs_failed == epochs_to_fail_to_terminate):
            print("Saving plot")
            with torch.no_grad():
                #print("nope..")
                visualizer.visualize()
                visualizer.plot()
                visualizer.save(img_save_dir, epoch)
        
        #print("Saving intermediate model")
        #save_model(odenet, intermediate_models_dir, 'model_at_epoch{}'.format(epoch))
    
        # Decrease learning rate if specified
        if settings['dec_lr'] : #and epoch % settings['dec_lr'] == 0
            decrease_lr(opt, True,tot_epochs= tot_epochs,
             epoch = epoch, lower_lr = settings['init_lr'], dec_lr_factor = settings['dec_lr_factor'])
        

        #val_loss < (0.01 * settings['scale_expression'])**1
        if (epoch in rep_epochs) or (consec_epochs_failed == epochs_to_fail_to_terminate):
            print()
            rep_epochs_so_far.append(epoch)
            print("Epoch=", epoch)
            rep_time_so_far = (perf_counter() - start_time)/3600
            print("Time so far= ", rep_time_so_far, "hrs")
            rep_epochs_time_so_far.append(rep_time_so_far)
            print("Best training (MSE) so far= ", min_train_loss)
            rep_epochs_train_losses.append(min_train_loss)
            if data_handler.n_val > 0:
                print("Best validation (MSE) so far = ", min_val_loss.item())
                #print("True loss of best validation model (MSE) = ", true_loss_of_min_val_model.item())
                rep_epochs_val_losses.append(min_val_loss.item())
                #rep_epochs_mu_losses.append(0)
                rep_epochs_mu_losses.append(true_loss_of_min_val_model.item())
            else:
                #print("True loss of best training model (MSE) = ", true_loss_of_min_train_model.item())
                print("True loss of best training model (MSE) = ", 0)
            print("Saving MSE plot...")
            plot_MSE_new(epoch, training_loss, validation_loss, true_mean_losses, true_mean_losses_init_val_based, prior_losses, img_save_dir)    
            

            print("Saving losses..")
            if data_handler.n_val > 0:
                L = [rep_epochs_so_far, rep_epochs_time_so_far, rep_epochs_train_losses, rep_epochs_val_losses, rep_epochs_mu_losses]
                np.savetxt('{}rep_epoch_losses.csv'.format(output_root_dir), np.transpose(L), delimiter=',')    
            
            if epoch >= 75:
                print("Saving best intermediate val model..")
                interm_model_file_name = 'trained_model_epoch_' + str(epoch)
                save_model(odenet, interm_models_save_dir , interm_model_file_name)
                    
            
        if consec_epochs_failed==epochs_to_fail_to_terminate:
            print("Went {} epochs without improvement; terminating.".format(epochs_to_fail_to_terminate))
            break


    total_time = perf_counter() - start_time

    
    save_model(odenet, output_root_dir, 'final_model')

    print("Saving times")
    np.savetxt('{}epoch_times.csv'.format(output_root_dir), epoch_times, delimiter=',')

        
    print("DONE!")

  
 