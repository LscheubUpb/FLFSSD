"""
This file is used for the Selective Synaptic Dampening method
Strategy files use the methods from here
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, dataset
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from typing import Dict, List
from iresnet_arc import CustomPReLU
from iresnet import CustomPReLU as CustomPReLU_Mag

###############################################
# Clean implementation
###############################################
    
class ParameterPerturber:
    def __init__(
        self,
        model,
        opt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        parameters=None,
    ):
        self.model = model
        self.opt = opt
        self.device = device
        self.alpha = None
        self.xmin = None

        print(parameters)
        self.lower_bound = parameters["lower_bound"]# type: ignore
        self.exponent = parameters["exponent"]# type: ignore
        self.magnitude_diff = parameters["magnitude_diff"]  # unused# type: ignore
        self.min_layer = parameters["min_layer"]# type: ignore
        self.max_layer = parameters["max_layer"]# type: ignore
        self.forget_threshold = parameters["forget_threshold"]# type: ignore
        self.dampening_constant = parameters["dampening_constant"]# type: ignore
        self.selection_weighting = parameters["selection_weighting"] # type: ignore

    def get_layer_num(self, layer_name: str) -> int:
        layer_id = layer_name.split(".")[1]
        if layer_id.isnumeric():
            return int(layer_id)
        else:
            return -1

    def zerolike_params_dict(self, model: torch.nn) -> Dict[str, torch.Tensor]: # type: ignore
        """
        Taken from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Returns a dict like named_parameters(), with zeroed-out parameter valuse
        Parameters:
        model (torch.nn): model to get param dict from
        Returns:
        dict(str,torch.Tensor): dict of zero-like params
        """
        return dict(
            [
                (k, torch.zeros_like(p, device=p.device))
                for k, p in model.named_parameters() # type: ignore
            ]
        )

    def fulllike_params_dict(
        self, model: torch.nn, fill_value, as_tensor: bool = False # type: ignore
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict like named_parameters(), with parameter values replaced with fill_value

        Parameters:
        model (torch.nn): model to get param dict from
        fill_value: value to fill dict with
        Returns:
        dict(str,torch.Tensor): dict of named_parameters() with filled in values
        """

        def full_like_tensor(fillval, shape: list) -> list:
            """
            recursively builds nd list of shape shape, filled with fillval
            Parameters:
            fillval: value to fill matrix with
            shape: shape of target tensor
            Returns:
            list of shape shape, filled with fillval at each index
            """
            if len(shape) > 1:
                fillval = full_like_tensor(fillval, shape[1:])
            tmp = [fillval for _ in range(shape[0])]
            return tmp

        dictionary = {}

        for n, p in model.named_parameters(): # type: ignore
            _p = (
                torch.tensor(full_like_tensor(fill_value, p.shape), device=self.device)
                if as_tensor
                else full_like_tensor(fill_value, p.shape)
            )
            dictionary[n] = _p
        return dictionary

    def subsample_dataset(self, dataset: dataset, sample_perc: float) -> Subset: # type: ignore
        """
        Take a subset of the dataset

        Parameters:
        dataset (dataset): dataset to be subsampled
        sample_perc (float): percentage of dataset to sample. range(0,1)
        Returns:
        Subset (float): requested subset of the dataset
        """
        sample_idxs = np.arange(0, len(dataset), step=int((1 / sample_perc))) # type: ignore
        return Subset(dataset, sample_idxs) # type: ignore

    def split_dataset_by_class(self, dataset: dataset) -> List[Subset]: # type: ignore
        """
        Split dataset into list of subsets
            each idx corresponds to samples from that class

        Parameters:
        dataset (dataset): dataset to be split
        Returns:
        subsets (List[Subset]): list of subsets of the dataset,
            each containing only the samples belonging to that class
        """
        n_classes = len(set([target for _, target in dataset])) # type: ignore
        subset_idxs = [[] for _ in range(n_classes)]
        for idx, (x, y) in enumerate(dataset): # type: ignore
            subset_idxs[y].append(idx)

        return [Subset(dataset, subset_idxs[idx]) for idx in range(n_classes)] # type: ignore

    def calc_importance(self, dataloader: DataLoader):
        batch_size = int(os.getenv('BATCH_SIZE',12))
        currentMode = os.getenv('IMP_DATASET',"None")
        arch = os.getenv('ARCH',"UNDEFINED")
        sampling = os.getenv('SAMPLING',"none")
        part = int(os.getenv('PART',"0"))
        
        running_means = {}
        calculateBool = True

        if (part>=2): # For individual classes unlearning, recalculating after forgetting two classes
            # Uncomment first line and comment second line to recaulculate importances after part 2
            # savePath = f"./importances/{arch}/Full_imp_sampled_part_over_2"
            savePath = f"./importances/{arch}/MS1M_V2_{currentMode}_imp"
        else:
            savePath = f"./importances/{arch}/MS1M_V2_{currentMode}_imp"
        if os.path.exists(savePath) and not currentMode == "None": # and sampling == "none":
            if(os.getenv('Unlearned_Full_imp',"None") != "None" and currentMode == "Full"):
                print(f"Loading importances for {currentMode} dataset using the unlearned 1 importances")
                calculateBool = False
                running_means = torch.load(savePath + os.getenv('Unlearned_Full_imp',"None"))
                print(f"Loaded importances from {savePath + os.getenv('Unlearned_Full_imp','None')}")
            else:
                print(f"Loading importances for {currentMode} dataset")
                calculateBool = False
                running_means = torch.load(savePath)
                print(f"Loaded importances from {savePath}")

        if(calculateBool):
            # Define a hook to capture activations
            running_means = {}
            total_samples = 0

            def hook_fn(name):
                def hook(module, input, output):
                    if name not in running_means:
                        running_means[name] = torch.zeros_like(output.detach().mean(dim=[0, 2, 3]) if output.dim() == 4 else output.detach().mean(dim=0))
                    
                    # Compute mean across batch
                    cur_mean = output.detach().mean(dim=[0, 2, 3]) if output.dim() == 4 else output.detach().mean(dim=0)

                    # Update running mean
                    running_means[name] = (running_means[name] * total_samples + cur_mean * batch_size) / (total_samples + batch_size)

                return hook

            # Register hooks for all layers
            for name, module in self.model.named_modules():
                module.register_forward_hook(hook_fn(name))

            try:
                for batch in tqdm(dataloader, desc="Calculating Activations"):
                    x, _, _ = batch
                    x = x.to(self.device)

                    self.opt.zero_grad()
                    _ = self.model(x)  # Forward pass to trigger hooks

                    total_samples += batch_size

                # Remove hooks after extraction
                for name, module in self.model.named_modules():
                    module._forward_hooks.clear()

                if(currentMode != "None"):# and sampling == "none"):
                    print("Saving importances")
                    os.makedirs(f"./importances/{arch}", exist_ok=True)
                    torch.save(running_means, savePath)
                    print("Saved importances")

            except:
                # print("encountered error, saving importances up to this point")
                # print("Saving importances")
                # os.makedirs(f"./importances/{arch}", exist_ok=True)
                # torch.save(running_means, savePath)
                # print("Saved importances")
                a = 0/0
            
        return running_means

    def modify_weight(
        self,
        original_importance,
        forget_importance
    ) -> None:
    
        # The activations were calculated "incorrectly", this fixes them
        for name, value in original_importance.items():
            if isinstance(value, torch.Tensor):
                while value.dim() > 1:
                    value = value.mean(dim=-1)  
                original_importance[name] = value
        for name, value in forget_importance.items():
            if isinstance(value, torch.Tensor):
                while value.dim() > 1:
                    value = value.mean(dim=-1)  
                forget_importance[name] = value

        with torch.no_grad():

            bn2_layers = [layer for layer in forget_importance.keys() if "bn2" in layer]
            # if "features" in bn2_layers[0]:
            #     bn2_layers = bn2_layers[-3:]
            # all_diffs = []

            for layer in bn2_layers:
                if layer == "bn2" or layer == "features.bn2":
                    continue

                prev_layer = layer # used for calculating the indicies and adjtstments to the bias

                layer = layer.replace("bn2","prelu") # used for finding the weights that are to be adjusted

                layer_forget_importance = forget_importance[prev_layer].to(self.device)
                layer_original_importance = original_importance[prev_layer].to(self.device)

                forget_sample_count = int(os.getenv("Forget_Samples",'100'))
                
                layer_retain_importances = (5822653 * layer_original_importance - forget_sample_count * layer_forget_importance) / (5822653 - forget_sample_count)
                diffs = layer_forget_importance - layer_retain_importances

                # all_diffs.append(diffs.cpu())

                ###### Top k of each layer ############################
                # top_indices = torch.topk(diffs, int(self.selection_weighting)).indices
                ######################################################

                ###### Top lamda % of each layer ######################
                num_neurons = len(diffs)
                num_top = int((self.selection_weighting / 100) * num_neurons)         
                top_indices = torch.topk(diffs, num_top).indices
                #######################################################

                ###### Treshold of alpha ##############################
                # max_diff = max(diffs)
                # top_indices = (diffs > self.selection_weighting).nonzero(as_tuple=True)[0]
                #######################################################

                # print(f"Len Top indices: {len(top_indices)}, indicies: {top_indices}, values {diffs[top_indices]}")

                # Get the last hidden layer
                last_hidden_layer = self.model
                attributes = layer.split('.')
                for attr in attributes:
                    last_hidden_layer = getattr(last_hidden_layer, attr)

                # Make bias
                bias = torch.zeros_like(layer_forget_importance)
                bias[top_indices] = self.dampening_constant * diffs[top_indices]
                # bias[top_indices] = self.dampening_constant * layer_forget_importance[top_indices]

                # Apply bias
                if isinstance(last_hidden_layer, nn.PReLU):
                    num_parameters = last_hidden_layer.num_parameters
                    new_module = CustomPReLU(num_parameters, bias)
                    new_module.prelu.weight.data = last_hidden_layer.weight.data.clone()
                    last_hidden_layer = new_module
                elif isinstance(last_hidden_layer, CustomPReLU) or isinstance(last_hidden_layer, CustomPReLU_Mag):
                    last_hidden_layer.bias += nn.Parameter(bias)
                else:
                    print("Not a prelu layer?????")
                    print(last_hidden_layer)
                    exit()
                    # a = 0/0

            # plt.clf()
            # i = 0
            # for diffs in all_diffs:
            #     i += 1
            #     plt.hist(diffs, bins=30, alpha=0.5, label=f"Layer {i} of size {len(diffs)}", density=True)

            # plt.title('Activation Difference')
            # plt.xlabel('Amount')
            # plt.ylabel('Density')
            # # plt.yscale('log')
            # plt.legend()
            # plt.savefig(f".\\distributions\\activation_diffs_all_layers.png", dpi=300, bbox_inches='tight')
            # a = 0/0

###############################################