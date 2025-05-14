'Entries: 50/50 (Modified A Lot)'

"""
Refer to forget_full_class_... for comments
This file is near identical with minimal modifications to facilitate random forgetting.
Seperate file to allow for easy reuse.
"""

import torch

from unlearn import *
from utils_ssd import *
import lfssd as lfssd_module
import lfssdv2 as lfssdv2_module
import ssd_extractor as ssd_module
from types import SimpleNamespace
sys.path.append('C:/Users/leosc/Documents/_wichtigeDokumente/Bachelorarbeit/MagFace')
import metrics
from eval.eval_recognition.eval_1v1 import main as eval_1v1_main # type: ignore
from inference.gen_feat import main as gen_feat_main # type: ignore
import utils.utils # type: ignore
import MIA
from colorama import Fore, Style

import os

def get_metric_scores(
    datasetArg,
    savePath,
    model_name,
    identifier,
    retain_train_dl, 
    forget_train_dl, 
    valid_dl, 
    model,
    forgetClasses,
    verificationMethod
):
    print(Fore.GREEN + "Generating Features" + Style.RESET_ALL)
    # Generate Features using the untrained model
    os.environ['PATH_PREFIX'] = "C:/Users/leosc/Documents/_wichtigeDokumente/Bachelorarbeit/MagFace/eval/eval_recognition/"
    featList = f"C:/Users/leosc/Documents/_wichtigeDokumente/Bachelorarbeit/MagFace/eval/eval_recognition/features/magface_{model_name}/{datasetArg}_{identifier}_unlearned.list"
    infList = f"C:/Users/leosc/Documents/_wichtigeDokumente/Bachelorarbeit/MagFace/eval/eval_recognition/data/{datasetArg}/img.list"
    gen_feat(model_name, featList, infList, savePath)
    
    print(Fore.GREEN + "Extracting Features for Cosine Similarity and MIA" + Style.RESET_ALL)
    # Cosine Similarity
    cosSim_Unlearned = 0
    cosSim_Reference = 0

    if (identifier == "baseline"):
        identifier = f"baseline_{forgetClasses}"

    compparison_Size = os.getenv('COMPARISONSIZE','100')
    # Unlearned
    print("Generating Features of Data not in the Unlearning Set using the Unlearned Model")
    os.environ['PATH_PREFIX'] = ""
    featList_Unlearned_Reference = f"C:/Users/leosc/Documents/_wichtigeDokumente/Bachelorarbeit/selective-synaptic-dampening-main/src/SimilarityLists/{model_name}/NotUnlearnedEmbeddings_{identifier}.list"
    infList = f"C:/Users/leosc/Documents/_wichtigeDokumente/Bachelorarbeit/selective-synaptic-dampening-main/src/SimilarityLists/{compparison_Size}_retain.list"
    print(f"Using Similarities of path {featList_Unlearned_Reference.split('/')[-1]}")
    gen_feat(model_name, featList_Unlearned_Reference, infList, savePath)

    print("Generating Features of Data in the Unlearning Set using the Unlearned Model")
    featList_Unlearned_Unlearned = f"C:/Users/leosc/Documents/_wichtigeDokumente/Bachelorarbeit/selective-synaptic-dampening-main/src/SimilarityLists/{model_name}/UnlearnedEmbeddings_{identifier}.list"
    infList = f"C:/Users/leosc/Documents/_wichtigeDokumente/Bachelorarbeit/selective-synaptic-dampening-main/src/SimilarityLists/unlearned_{forgetClasses}_img.list"
    print(f"Using Similarities of path {featList_Unlearned_Unlearned.split('/')[-1]}")
    gen_feat(model_name, featList_Unlearned_Unlearned, infList, savePath)

    # Baseline
    print("Generating Features of Data not in the Unlearning Set using the Baseline Model")
    savePath = os.getenv('BASELINE_PATH')
    featList_Baseline_Reference = f"C:/Users/leosc/Documents/_wichtigeDokumente/Bachelorarbeit/selective-synaptic-dampening-main/src/SimilarityLists/{model_name}/NotUnlearnedEmbeddings_Baseline_{forgetClasses}.list"
    infList = f"C:/Users/leosc/Documents/_wichtigeDokumente/Bachelorarbeit/selective-synaptic-dampening-main/src/SimilarityLists/{compparison_Size}_retain.list"
    print(f"Using Similarities of path {featList_Baseline_Reference.split('/')[-1]}")
    gen_feat(model_name, featList_Baseline_Reference, infList, savePath)

    print("Generating Features of Data in the Unlearning Set using the Baseline Model")
    featList_Baseline_Unlearned = f"C:/Users/leosc/Documents/_wichtigeDokumente/Bachelorarbeit/selective-synaptic-dampening-main/src/SimilarityLists/{model_name}/UnlearnedEmbeddings_Baseline_{forgetClasses}.list"
    infList = f"C:/Users/leosc/Documents/_wichtigeDokumente/Bachelorarbeit/selective-synaptic-dampening-main/src/SimilarityLists/unlearned_{forgetClasses}_img.list"
    print(f"Using Similarities of path {featList_Baseline_Unlearned.split('/')[-1]}")
    gen_feat(model_name, featList_Baseline_Unlearned, infList, savePath)

    if (identifier == f"baseline_{forgetClasses}"):
        identifier = "baseline"

    # metrics.makeTSNE(featList_Unlearned_Reference, featList_Unlearned_Unlearned ,"Unlearned")
    # metrics.makeTSNE(featList_Baseline_Reference, featList_Baseline_Unlearned, "Baseline")

    print(Fore.GREEN + "Calculating Cosine Similarities" + Style.RESET_ALL)
    print("Calculating Cosine Similarities for the unlearned set")
    forget_similarity_set, cosSim_Unlearned = metrics.cosineSimilarity(featList_Unlearned_Unlearned, featList_Baseline_Unlearned)
    print(f"Cosine Similiarity Unlearned: {cosSim_Unlearned}")

    print("Calculating Cosine Similarities for the referene set")
    retain_similarity_set, cosSim_Reference = metrics.cosineSimilarity(featList_Unlearned_Reference, featList_Baseline_Reference)
    print(f"Cosine Similiarity Reference: {cosSim_Reference}")

    # print("Calculating Cosine Similarities for the valid set")
    # valid_similarity_set, cosSim_Valid = metrics.cosineSimilarity(featList_Unlearned_Valid, featList_Baseline_Valid)
    # print(f"Cosine Similiarity Valid: {cosSim_Valid}")

    print(Fore.GREEN + "Evaluating Mia" + Style.RESET_ALL)
    
    # MIA_methods = ["MIA","Kolmogorov","PIC"]
    MIA_method = verificationMethod
    mia, D_f_r, D_f_t = 0, 0, 0
    if(MIA_method == "MIA"):
        mia = metrics.get_membership_attack_prob(retain_train_dl, forget_train_dl, valid_dl, model)
    elif(MIA_method == "Kolmogorov"):
        D_f_r = MIA.calculate_MIA(forget_similarity_set, retain_similarity_set)
    elif(MIA_method == "PIC"):
        # mia = MIA.calculate_PIC(forget_similarity_set, retain_similarity_set)
        MIA_method = "BUS"
    elif(MIA_method == "both"):
        # mia = MIA.calculate_PIC(forget_similarity_set, retain_similarity_set)
        D_f_r = MIA.calculate_MIA(forget_similarity_set, retain_similarity_set)
        MIA_method = "BUS"

    print(Fore.GREEN + "Evaluating Pairwise Feature Accuracy" + Style.RESET_ALL)
    # Evaluate the features generated by the untrained model
    featList = f"C:/Users/leosc/Documents/_wichtigeDokumente/Bachelorarbeit/MagFace/eval/eval_recognition/features/magface_{model_name}/{datasetArg}_{identifier}_unlearned.list"
    pairList = f"C:/Users/leosc/Documents/_wichtigeDokumente/Bachelorarbeit/MagFace/eval/eval_recognition/data/{datasetArg}/pair.list"
    args = {
        "arch": model_name,
        "pair_list": pairList,
        "feat_list": featList, 
        "eval_type": "1v1",
        "distance_metric": 1,
        "test_folds": 10,
        "offset": 0
    }

    mean_acc, std_acc, tpr, fpr = eval_1v1_main(SimpleNamespace(**args))

    return mean_acc, std_acc, tpr, fpr, cosSim_Unlearned, cosSim_Reference, mia, D_f_r, MIA_method

def gen_feat(model_name, featList, infList, checkpointPath):
    args = {
        "arch": model_name,
        "inf_list": infList,
        "feat_list": featList, 
        "workers": 4,
        "batch_size": 16,
        "embedding_size": 512,
        "resume": checkpointPath,
        "print_freq": 100,
        "cpu_mode": False,
        "dist": 1
    }
    # if not os.path.exists(featList):
    gen_feat_main(SimpleNamespace(**args))

def ssd_tuning(
    model,
    forget_train_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device,
    model_name,
    dataset_name,
    method,
    version,
    forget_perc,
    ssd_version,
    retain_train_dl,
    valid_dl,
    sample_dl,
    **kwargs,
):
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1,
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": dampening_constant,
        "selection_weighting": selection_weighting,
    }

    identifier = f"{version}_{forget_perc}"
    savePath = f"C:\\Users\\leosc\\Documents\\_wichtigeDokumente\\Bachelorarbeit\\selective-synaptic-dampening-main\\src\\checkpoint\\unlearned\\extractor\\{model_name}"
    os.makedirs(savePath, exist_ok=True)
    savePath = f"{savePath}\\{identifier}_{method}_{dampening_constant}_{selection_weighting}_unlearned.pth"
    modelWithoutHead = model

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    if(ssd_version == "lfssd"):
        print("Using lfssd")
        ssd = lfssd_module
    elif(ssd_version == "lfssdv2"):
        print("Using lfssdv2")
        ssd = lfssdv2_module
    else:
        print("Using ssd")
        ssd = ssd_module
        headPath = "C:/Users/leosc/Documents/_wichtigeDokumente/Bachelorarbeit/selective-synaptic-dampening-main/src/checkpoint/arcface_iresnet50_v1.0_pretrained/rank-0_softmax_weight.pkl"
        model = metrics.addClassificationHead(model, headPath)

    ssd = ssd.ParameterPerturber(model, optimizer, device, parameters)
    model = model.eval()

    # Added environment variable for saving/loading the importances
    os.environ['ARCH'] = f"{model_name}_{ssd_version}"
    # os.environ['IMP_DATASET'] = f"Partial_{identifier}"
    os.environ['IMP_DATASET'] = f"None"
    sample_importances = ssd.calc_importance(forget_train_dl)

    os.environ['PROB_DATASET'] = f"Partial_{identifier}"

    os.environ['IMP_DATASET'] = "Full"

    if(sample_dl is not None):
        original_importances = ssd.calc_importance(sample_dl)
    else:
        original_importances = ssd.calc_importance(full_train_dl)

    ssd.modify_weight(original_importances, sample_importances) # type: ignore

    # Saving the model beacuse the embedding extraction and evaluation from magface load the checkpoint from a file (from savePath)
    checkpoint = {
        'epoch': 1,
        'arch': model_name,
        'state_dict': ssd.model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, savePath)
    os.environ['last_checkpoint'] = savePath

    verificationMethod = kwargs.get('verificationMethod')

    return get_metric_scores(dataset_name, savePath, model_name, f"{identifier}_{dampening_constant}_{selection_weighting}", retain_train_dl, forget_train_dl, valid_dl, modelWithoutHead, forget_perc, verificationMethod)

def baseline(
    model,
    retain_train_dl, 
    forget_train_dl, 
    valid_dl,
    savePath,
    **kwargs,
):
    model_name = kwargs.get('model_name')
    dataset_name = kwargs.get('dataset_name')
    forget_perc = kwargs.get('forget_perc')
    verificationMethod = kwargs.get('verificationMethod')

    return get_metric_scores(dataset_name, savePath, model_name, "baseline", retain_train_dl, forget_train_dl, valid_dl, model, forget_perc, verificationMethod)