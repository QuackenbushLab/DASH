import torch
import torch.nn as nn
import sys
import numpy as np

from torch.nn.init import calculate_gain
#torch.set_num_threads(36)



def get_mask_smallest_p_proportion(my_tensor, p):
    # Compute the P-th percentile of the absolute tensor.
    k = int(p * my_tensor.numel())
    abs_flattened_tensor = torch.abs(my_tensor.view(-1))
    values, indices = torch.topk(abs_flattened_tensor, k, largest=False)
    p_th_percentile = values[-1]

    # Create a mask of all elements in the tensor that are greater than or equal to the
    # P-th percentile.
    mask = torch.gt(torch.abs(my_tensor), p_th_percentile)

    # Multiply the tensor by the mask. This will set all elements in the tensor that
    # are less than or equal to the P-th percentile to 0.
    return mask.long()

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


class ODENet(nn.Module):
    ''' ODE-Net class implementation '''

    
    def __init__(self, device, ndim, explicit_time=False, neurons=100, log_scale = "linear", init_bias_y = 0):
        ''' Initialize a new ODE-Net '''
        super(ODENet, self).__init__()

        self.ndim = ndim
        self.explicit_time = explicit_time
        self.log_scale = log_scale
        self.init_bias_y = init_bias_y
        #only use first 68 (i.e. TFs) as NN inputs
        #in general should be num_tf = ndim
        self.num_tf = 73 
        
        # Create a new sequential model with ndim inputs and outputs
        if explicit_time:
            self.net = nn.Sequential(
                nn.Linear(ndim + 1, neurons),
                nn.LeakyReLU(),
                nn.Linear(neurons, neurons),
                nn.LeakyReLU(),
                nn.Linear(neurons, neurons),
                nn.LeakyReLU(),
                nn.Linear(neurons, ndim)
            )
        else: #6 layers
           
            self.net_prods = nn.Sequential()
            self.net_prods.add_module('activation_0',  LogShiftedSoftSignMod()) #
            self.net_prods.add_module('linear_out', nn.Linear(ndim, neurons, bias = True)) #bias = True
            
            self.net_sums = nn.Sequential()
            self.net_sums.add_module('activation_0', SoftsignMod())
            self.net_sums.add_module('linear_out', nn.Linear(ndim, neurons, bias = True)) #bias = True

            self.net_alpha_combine_sums = nn.Sequential()
            self.net_alpha_combine_sums.add_module('linear_out',nn.Linear(neurons, ndim, bias = False))

            self.net_alpha_combine_prods = nn.Sequential()
            self.net_alpha_combine_prods.add_module('linear_out',nn.Linear(neurons, ndim, bias = False))
          
          
            self.gene_multipliers = nn.Parameter(torch.rand(1,ndim), requires_grad= True)
            #self.gene_multipliers = nn.Parameter(torch.ones(ndim), requires_grad= True)
                
        # Initialize the layers of the model
        for n in self.net_sums.modules():
            if isinstance(n, nn.Linear):
                nn.init.sparse_(n.weight,  sparsity=0.95, std = 0.05) 
                #nn.init.orthogonal_(n.weight, gain = calculate_gain("sigmoid"))
        
        for n in self.net_prods.modules():
            if isinstance(n, nn.Linear):
                nn.init.sparse_(n.weight,  sparsity=0.95, std = 0.05)
                #nn.init.sparse_(n.weight,  sparsity=0.95, std = 0.05) #0.05
                #nn.init.orthogonal_(n.weight, gain = calculate_gain("sigmoid"))
                
        for n in self.net_alpha_combine_sums.modules():
            if isinstance(n, nn.Linear):
                nn.init.orthogonal_(n.weight, gain = calculate_gain("sigmoid"))
                #nn.init.sparse_(n.weight,  sparsity=0.95, std = 0.05)

        for n in self.net_alpha_combine_prods.modules():
            if isinstance(n, nn.Linear):
                nn.init.orthogonal_(n.weight, gain = calculate_gain("sigmoid"))
         
        
        #self.net_prods.to(device)
        self.gene_multipliers.to(device)
        self.net_sums.to(device)
        self.net_prods.to(device)
        self.net_alpha_combine_sums.to(device)
        self.net_alpha_combine_prods.to(device)
        
        
        
    def forward(self, t, y):
        sums = self.net_sums(y)
        prods = torch.exp(self.net_prods(y))
        joint = self.net_alpha_combine_sums(sums) + self.net_alpha_combine_prods(prods)
        #final = joint-torch.relu(self.gene_multipliers)*y
        final = torch.relu(self.gene_multipliers)*(joint-y)
        return(final) 

    def prior_only_forward(self, t, y):
        sums = self.net_sums(y)
        prods = torch.exp(self.net_prods(y))
        joint = self.net_alpha_combine_sums(sums) + self.net_alpha_combine_prods(prods)
        return(joint)

    def save(self, fp):
        ''' Save the model to file '''
        idx = fp.index('.')
        alpha_comb_sums_path = fp[:idx] + '_alpha_comb_sums' + fp[idx:]
        alpha_comb_prods_path = fp[:idx] + '_alpha_comb_prods' + fp[idx:]
        gene_mult_path = fp[:idx] + '_gene_multipliers' + fp[idx:]
        
        prod_path =  fp[:idx] + '_prods' + fp[idx:]
        sum_path = fp[:idx] + '_sums' + fp[idx:]
        torch.save(self.net_prods, prod_path)
        torch.save(self.net_sums, sum_path)
        torch.save(self.net_alpha_combine_sums, alpha_comb_sums_path)
        torch.save(self.net_alpha_combine_prods, alpha_comb_prods_path)
        torch.save(self.gene_multipliers, gene_mult_path)
        

    def load_dict(self, fp):
        ''' Load a model from a dict file '''
        self.net.load_state_dict(torch.load(fp))
    
    def inherit_params(self, fp):
        idx = fp.index('.pt')
        gene_mult_path = fp[:idx] + '_gene_multipliers' + fp[idx:]
        prod_path =  fp[:idx] + '_prods' + fp[idx:]
        sum_path = fp[:idx] + '_sums' + fp[idx:]
        alpha_comb_sums_path = fp[:idx] + '_alpha_comb_sums' + fp[idx:]
        alpha_comb_prods_path = fp[:idx] + '_alpha_comb_prods' + fp[idx:]

        with torch.no_grad():
            X = torch.load(prod_path)
            self.net_prods.linear_out.weight = nn.Parameter(X.linear_out.weight) 
            self.net_prods.linear_out.bias = nn.Parameter(X.linear_out.bias)

            X = torch.load(sum_path)
            self.net_sums.linear_out.weight = nn.Parameter(X.linear_out.weight) 
            self.net_sums.linear_out.bias = nn.Parameter(X.linear_out.bias)

            X = torch.load(alpha_comb_sums_path)
            self.net_alpha_combine_sums.linear_out.weight = nn.Parameter(X.linear_out.weight) 

            X = torch.load(alpha_comb_prods_path)
            self.net_alpha_combine_prods.linear_out.weight = nn.Parameter(X.linear_out.weight) 
            
            X = torch.load(gene_mult_path)
            self.gene_multipliers = X
            
        self.net_prods.to('cpu')
        self.net_sums.to('cpu')
        self.gene_multipliers.to('cpu')
        self.net_alpha_combine_sums.to('cpu')
        self.net_alpha_combine_prods.to('cpu')

        print("Inherited params from pre-trained model!")
        
            
    def load_model(self, fp):
        ''' Load a model from a file '''
        idx = fp.index('.pt')
        gene_mult_path = fp[:idx] + '_gene_multipliers' + fp[idx:]
        prod_path =  fp[:idx] + '_prods' + fp[idx:]
        sum_path = fp[:idx] + '_sums' + fp[idx:]
        alpha_comb_sums_path = fp[:idx] + '_alpha_comb_sums' + fp[idx:]
        alpha_comb_prods_path = fp[:idx] + '_alpha_comb_prods' + fp[idx:]
        
        self.net_prods = torch.load(prod_path)
        self.net_sums = torch.load(sum_path)
        self.gene_multipliers = torch.load(gene_mult_path)
        self.net_alpha_combine_sums = torch.load(alpha_comb_sums_path)
        self.net_alpha_combine_prods = torch.load(alpha_comb_prods_path)
        

        self.net_prods.to('cpu')
        self.net_sums.to('cpu')
        self.gene_multipliers.to('cpu')
        self.net_alpha_combine_sums.to('cpu')
        self.net_alpha_combine_prods.to('cpu')

    def load(self, fp):
        ''' General loading from a file '''
        try:
            print('Trying to load model from file= {}'.format(fp))
            self.load_model(fp)
            print('Done')
        except:
            print('Failed! Trying to load parameters from file...')
            try:
                self.load_dict(fp)
                print('Done')
            except:
                print('Failed! Network structure is not correct, cannot load parameters from file, exiting!')
                sys.exit(0)

    def to(self, device):
        self.net.to(device)