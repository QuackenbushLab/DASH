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
import pandas as pd
#import copy

import torch
import torch.optim as optim
import torch.nn as nn

#try:
#    from torchdiffeq.__init__ import odeint_adjoint as odeint
#except ImportError:
#    from torchdiffeq import odeint_adjoint as odeint

from datahandler import DataHandler
from read_config import read_arguments_from_file
from visualization_inte import *

from pathreg_helper_PHX import L0_MLP, initial_position, ODEBlock, PathReg, L1
from matplotlib.ticker import FuncFormatter
#torch.set_num_threads(16) #CHANGE THIS!

def plot_MSE(epoch_so_far, training_loss, validation_loss, true_mean_losses, true_mean_losses_init_val_based, prior_losses, img_save_dir):
    
    # Create two subplots, one for the main MSE loss plot and one for the prior loss plot.
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    fig.set_size_inches(12, 6)

    ax1.plot(range(1, epoch_so_far + 1), training_loss, color="blue", label="Training loss")
    ax1.plot(range(1, epoch_so_far + 1), true_mean_losses, color="green", label="Noiseless test loss")
    
    if len(validation_loss) > 0:
        ax1.plot(range(1, epoch_so_far + 1), validation_loss, color="red", label="Validation loss")

    ax2.plot(range(1, epoch_so_far + 1), prior_losses, color="magenta", label="Model sparsity")

    ax1.set_yscale('log')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Error (MSE)")
    ax1.legend(loc='upper right')

    def percentage_formatter(x, pos):
        return "{:.2%}".format(x)

    # Apply the custom formatter to the y-axis
    ax2.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Sparsity")
    ax2.set_title("Model sparsity")

    #plt.subplots_adjust(wspace=0.3)
    fig.tight_layout()
    plt.savefig("{}/MSE_loss.png".format(img_save_dir))
    np.savetxt('{}full_loss_info.csv'.format(output_root_dir), np.c_[training_loss, validation_loss, true_mean_losses, true_mean_losses_init_val_based], delimiter=',')

def my_r_squared(output, target):
    x = output
    y = target
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    my_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return(my_corr**2)

def get_true_val_set_r2(pathreg_model, data_handler, method, batch_type, num_reps = 1):
    data_pw, t_pw, target_pw = data_handler.get_true_mu_set_pairwise(val_only = True, batch_type =  batch_type)
    
    with torch.no_grad():
        predictions_pw = torch.zeros(data_pw.shape).to(data_handler.device)
        for index, (time, batch_point) in enumerate(zip(t_pw, data_pw)):
            pathreg_model[1].set_times(time)
            temp_preds = torch.empty((num_reps,1, data_handler.dim), device=data_handler.device)
            # Repeat the prediction operation 100 times
            for i in range(num_reps):
                pred_z = pathreg_model(batch_point.unsqueeze(1))
                temp_preds[i] = pred_z[1]
            # Calculate the average along the first dimension (axis=0)
            average_pred_z = torch.mean(temp_preds, dim=0)
            predictions_pw[index, :, :] = average_pred_z
        var_explained_pw = my_r_squared(predictions_pw, target_pw)
        true_val_mse = torch.mean((predictions_pw - target_pw)**2)
    
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
        return mat_torch
    else: #when scaling up >10000
        mat = np.genfromtxt(prior_mat_file_loc,delimiter=',')
        sparse_mat = torch.sparse_coo_tensor([mat[:,0].astype(int)-1, mat[:,1].astype(int)-1], mat[:,2], ( num_genes,  num_genes))
        mat_torch = sparse_mat.to_dense().float()
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




def validation(pathreg_model, data_handler, method, explicit_time, num_reps = 1):
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


def get_sparsity_OLD(model):
    
    # MODEL SPARSITY
    for i, layer in enumerate(model[1].odefunc.layers):
        if i ==0:
            all_weights = torch.abs(layer.sample_weights()).flatten()
        else:
            all_weights = torch.cat((all_weights,torch.abs(layer.sample_weights()).flatten()))
    model_spar = (torch.abs(all_weights)<1e-5).sum() / all_weights.shape[0]
    
    return model_spar

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

def get_sparsity_new(pathreg_model):
    Wo_sums = pathreg_model[1].odefunc.output_sums[1].sample_weights().detach().numpy()
    alpha_comb_sums = pathreg_model[1].odefunc.output_sums[2].sample_weights().detach().numpy()
    Wo_prods = pathreg_model[1].odefunc.output_prods[1].sample_weights().detach().numpy()
    alpha_comb_prods = pathreg_model[1].odefunc.output_prods[3].sample_weights().detach().numpy()
    
    my_sparsity = average_zeros_across_arrays(Wo_prods, Wo_sums, alpha_comb_prods, alpha_comb_sums)
    return(my_sparsity)


def training_step(pathreg_model, data_handler, opt, method, batch_size, lambda_l1, lambda_pathreg, num_reps = 1):
    batch, t, target = data_handler.get_batch(batch_size)
    opt.zero_grad()
    predictions = torch.zeros(batch.shape).to(data_handler.device)
    
    '''
    my_time = torch.unique(t.view(-1))
    pathreg_model[1].set_times(my_time)
    pred_z = pathreg_model(batch[0:1, ])
    predictions = pred_z[1:, :, :]
    '''
    for index, (time, batch_point) in enumerate(zip(t, batch)):
        pathreg_model[1].set_times(time)
        temp_preds = torch.empty((num_reps,1, data_handler.dim), device=data_handler.device)

        # Repeat the prediction operation 100 times
        for i in range(num_reps):
            pred_z = pathreg_model(batch_point.unsqueeze(1))
            temp_preds[i] = pred_z[1]

        # Calculate the average along the first dimension (axis=0)
        average_pred_z = torch.mean(temp_preds, dim=0)
        predictions[index, :, :] = average_pred_z 
    
    loss_data = torch.mean((predictions - target)**2) 
    path_reg = PathReg(pathreg_model)
    l1_reg = L1(pathreg_model)
    composed_loss = loss_data + lambda_l1 * l1_reg + lambda_pathreg * path_reg

    composed_loss.backward() #MOST EXPENSIVE STEP!
    opt.step()

    model_spar = get_sparsity_new(pathreg_model)

    return [loss_data, model_spar]




def _build_save_file_name(save_path, epochs):
    return '{}-{}-{}({};{})_pathreg_{}_{}epochs'.format(str(datetime.now().year), str(datetime.now().month),
        str(datetime.now().day), str(datetime.now().hour), str(datetime.now().minute), save_path, epochs)

def save_model_old(pathreg_model, opt, epoch, folder, filename):
    torch.save({
            'epoch': epoch,
            'model_state_dict': pathreg_model.state_dict(),
            'optimizer_state_dict': opt.state_dict()}, 
            '{}/{}.pt'.format(folder, filename))
  
def save_model(pathreg_model, opt, epoch, folder, filename):
    torch.save(pathreg_model, 
            '{}/{}.pt'.format(folder, filename))  


parser = argparse.ArgumentParser('Testing')
parser.add_argument('--settings', type=str, default='config_breast.cfg')
clean_name =  "desmedt_11165genes_1sample_186T" 
parser.add_argument('--data', type=str, default='/home/ubuntu/lottery_tickets_phoenix/breast_cancer_data/clean_data/{}.csv'.format(clean_name))

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
    device = 'cpu'
    
    data_handler = DataHandler.fromcsv(args.data, device, settings['val_split'], normalize=settings['normalize_data'], 
                                        batch_type=settings['batch_type'], batch_time=settings['batch_time'], 
                                        batch_time_frac=settings['batch_time_frac'],
                                        noise = settings['noise'],
                                        img_save_dir = img_save_dir,
                                        scale_expression = settings['scale_expression'],
                                        log_scale = settings['log_scale'],
                                        init_bias_y = settings['init_bias_y'])
    
    lr_schedule_patience = 3
    lambda_l1 =  0 #0.001
    lambda_pathreg = 0#0.001
    num_reps = 1


    nhidden = 300
    data_dim = data_handler.dim
    feature_layers = [initial_position(data_dim, nhidden), 
                    ODEBlock(
                        odefunc = L0_MLP(input_dim=data_dim, layer_dims=(nhidden,nhidden,data_dim), temperature = 2/3, my_lambda = 1), #, N=150
                        dim = data_dim, 
                        tol = 1e-3,
                        method = settings['method']
                    )]
    pathreg_model = nn.Sequential(*feature_layers).to(device)
    loss_func = nn.MSELoss()
    

    opt = optim.Adam(pathreg_model.parameters(), lr=settings['init_lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', 
                                                     factor=0.9, patience=lr_schedule_patience, threshold=1e-06, 
                                                     threshold_mode='abs', cooldown=0, min_lr=4e-05, eps=1e-09, verbose=True)
    

    
    with open('{}/network.txt'.format(output_root_dir), 'w') as net_file:
        print(pathreg_model, file = net_file)
        print(".......", file = net_file)
        print("lambda_l1 = {}".format(lambda_l1), file = net_file)
        print("lambda_pathreg = {}".format(lambda_pathreg), file = net_file)
        print("num Monte Carlo reps = {}".format(num_reps), file = net_file)
        
    # Init plot
    if settings['viz']:
        visualizer = Visualizator1D(data_handler, pathreg_model, settings)

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
    rep_epochs = [2, 5, 10, 15, 30, 40, 50, 75, 100, 120, 150, 180, 200,220, 240, 260, 280, 300, 350, tot_epochs]
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

    

    
            
    if settings['viz']:
        with torch.no_grad():
            visualizer.visualize_pathreg()
            visualizer.plot()
            visualizer.save(img_save_dir, 0)
    
    start_time = perf_counter()
    
    

    for epoch in range(1, tot_epochs + 1):
        print()
        print("[Running epoch {}/{}]".format(epoch, settings['epochs']))

        
        print("current lambda_l1 =", lambda_l1)
        print("current lambda_pathreg =", lambda_pathreg)
            
        start_epoch_time = perf_counter()
        iteration_counter = 1
        data_handler.reset_epoch()
        #visualizer.save(img_save_dir, epoch) #IH added to test
        this_epoch_total_train_loss = 0
        this_epoch_total_prior_loss = 0
        
        
        if settings['verbose']:
            pbar = tqdm(total=iterations_in_epoch, desc="Training loss:")
        while not data_handler.epoch_done:
            start_batch_time = perf_counter()
            
            loss_list = training_step(pathreg_model, data_handler, opt, settings['method'], settings['batch_size'], lambda_l1, lambda_pathreg, num_reps)
            loss = loss_list[0]
            prior_loss = loss_list[1]
            #batch_times.append(perf_counter() - start_batch_time)

            this_epoch_total_train_loss += loss.item()
            this_epoch_total_prior_loss += prior_loss.item()
            # Print and update plots
            iteration_counter += 1

            if settings['verbose']:
                pbar.update(1)
                pbar.set_description("Training loss, Sparsity: {:.2E}, {:.2%}".format(loss.item(), prior_loss.item()))
        
        epoch_times.append(perf_counter() - start_epoch_time)

        #Epoch done, now handle training loss
        train_loss = this_epoch_total_train_loss/iterations_in_epoch
        training_loss.append(train_loss)
        prior_losses.append(this_epoch_total_prior_loss/iterations_in_epoch)
        #print("Overall training loss {:.5E}".format(train_loss))

        mu_loss = get_true_val_set_r2(pathreg_model, data_handler, settings['method'], settings['batch_type'], num_reps)
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

        #handle true-mu loss
       
        if data_handler.n_val > 0:
            val_loss_list = validation(pathreg_model, data_handler, settings['method'], settings['explicit_time'], num_reps)
            val_loss = val_loss_list[0]
            validation_loss.append(val_loss)
            if epoch == 1:
                min_val_loss = val_loss
                true_loss_of_min_val_model = mu_loss[1]
                print('Model improved, saving current model')
                best_vaL_model_so_far = pathreg_model
                save_model(pathreg_model, opt, epoch, output_root_dir, 'best_val_model')
                
            else:
                if val_loss < min_val_loss:
                    consec_epochs_failed = 0
                    min_val_loss = val_loss
                    true_loss_of_min_val_model =  mu_loss[1]
                    #saving true-mean loss of best val model
                    print('Model improved, saving current model')
                    save_model(pathreg_model, opt, epoch, output_root_dir, 'best_val_model')
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
                visualizer.visualize_pathreg()
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
            plot_MSE(epoch, training_loss, validation_loss, true_mean_losses, true_mean_losses_init_val_based, prior_losses, img_save_dir)    
            

            print("Saving losses..")
            if data_handler.n_val > 0:
                L = [rep_epochs_so_far, rep_epochs_time_so_far, rep_epochs_train_losses, rep_epochs_val_losses, rep_epochs_mu_losses]
                np.savetxt('{}rep_epoch_losses.csv'.format(output_root_dir), np.transpose(L), delimiter=',')    
            
            #print("Saving best intermediate val model..")
            #interm_model_file_name = 'trained_model_epoch_' + str(epoch)
            #save_model(odenet, interm_models_save_dir , interm_model_file_name)
                
            
           
        

        if consec_epochs_failed==epochs_to_fail_to_terminate:
            print("Went {} epochs without improvement; terminating.".format(epochs_to_fail_to_terminate))
            break


    total_time = perf_counter() - start_time

    save_model(pathreg_model, opt, epoch, output_root_dir, 'final_model')    
    print("DONE!")

  
 