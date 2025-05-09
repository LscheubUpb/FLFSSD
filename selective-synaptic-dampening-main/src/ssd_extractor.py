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
import time
import copy
import os
import pdb
import math
import shutil
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from typing import Dict, List

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

    def calc_importance(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Adapated from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Calculate per-parameter, importance
            returns a dictionary [param_name: list(importance per parameter)]
        Parameters:
        DataLoader (DataLoader): DataLoader to be iterated over
        Returns:
        importances (dict(str, torch.Tensor([]))): named_parameters-like dictionary containing list of importances for each parameter
        """
        ################################################
        # Uncomment for SSD
        # criterion = nn.CrossEntropyLoss()
        ################################################
        batch_size = int(os.getenv('BATCH_SIZE',12))
        currentMode = os.getenv('IMP_DATASET',"None")
        arch = os.getenv('ARCH',"UNDEFINED")
        sampling = os.getenv('SAMPLING',"none")
        part = int(os.getenv('PART',"0"))

        importances = self.zerolike_params_dict(self.model)# type: ignore
        calculateBool = True

        if (part>=2): # For individual classes unlearning, recalculating after forgetting two classes
            savePath = f"./importances/{arch}/Full_imp_sampled_part_over_2"
        else:
            savePath = f"./importances/{arch}/MS1M_V2_{currentMode}_imp"
        print(savePath)
        if os.path.exists(savePath) and not currentMode == "None":
            if(os.getenv('Unlearned_Full_imp',"None") != "None" and currentMode == "Full"):
                print(f"Loading importances for {currentMode} dataset using the unlearned 1 importances")
                calculateBool = False
                importances = torch.load(savePath + os.getenv('Unlearned_Full_imp',"None"))
                print(f"Loaded importances from {savePath + os.getenv('Unlearned_Full_imp','None')}")
            else:
                print(f"Loading importances for {currentMode} dataset")
                calculateBool = False
                importances = torch.load(savePath)
                print(f"Loaded importances from {savePath}")

        if(calculateBool):
            for batch in tqdm(dataloader, "Calculating Importances"):
                x, _, _ = batch
                x = x.to(self.device)
                self.opt.zero_grad()
                out = self.model(x)
                ################################################
                # Uncomment first line for SSD, second for LFSSD
                # loss = criterion(out, y)
                loss = torch.norm(out, p="fro", dim=1).pow(2).mean()
                ################################################
                loss.backward()

                for (k1, p), (k2, imp) in zip(
                    self.model.named_parameters(), importances.items()
                ):
                    if p.grad is not None:
                        ################################################
                        # Uncomment first line for SSD, second for LFSSD
                        # imp.data += p.grad.data.clone().pow(2)
                        imp.data += p.grad.data.clone().abs()
                        ################################################
                # torch.cuda.empty_cache()

            # average over mini batch length
            for _, imp in importances.items():
                imp.data /= float(len(dataloader))

            if(currentMode != "None"): #and sampling == "none"):
                print("Saving importances")
                os.makedirs(f"./importances/{arch}", exist_ok=True)
                torch.save(importances, savePath)
                print("Saved importances")
        return importances

    def modify_weight(
        self,
        original_importance: List[Dict[str, torch.Tensor]],
        forget_importance: List[Dict[str, torch.Tensor]],
    ) -> None:
        """
        Perturb weights based on the SSD equations given in the paper
        Parameters:
        original_importance (List[Dict[str, torch.Tensor]]): list of importances for original dataset
        forget_importance (List[Dict[str, torch.Tensor]]): list of importances for forget sample
        threshold (float): value to multiply original imp by to determine memorization.

        Returns:
        None
        """

        with torch.no_grad():
            for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                self.model.named_parameters(),
                original_importance.items(), # type: ignore
                forget_importance.items(), # type: ignore
            ):
                # Synapse Selection with parameter alpha
                oimp_norm = oimp.mul(self.selection_weighting)
                locations = torch.where(fimp > oimp_norm)

                # Synapse Dampening with parameter lambda
                weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(
                    self.exponent
                )
                update = weight[locations]
                # Bound by 1 to prevent parameter values to increase.
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound
                p[locations] = p[locations].mul(update)
###############################################
