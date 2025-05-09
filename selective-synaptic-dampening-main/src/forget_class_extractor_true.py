import random
import os
import wandb
import time
import numpy as np
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset, random_split
import torchvision.transforms as transforms
from torch.utils.data import Subset
import datasets
from unlearn import *
from utils_ssd import *
import forget_class_extractor_strategies_true
from training_utils import *
from network_inf import builder_inf
import matplotlib.pyplot as plt
import copy
from colorama import Fore, Style

# Beacuse of how the dataset is loaded, sorted alphabetically not numerically
def num_items_in_folder(folder_path):
    return len(os.listdir(folder_path))

def create_unlearning_image_list_noBorder(ds):
    numForgetClasses = int(args.forgetClasses.split("-")[1])-int(args.forgetClasses.split("-")[0])+1
    fileName = f"./SimilarityLists/unlearned_{numForgetClasses}_img.list"
    with open(fileName, "w") as f:
        for _, path, _ in ds:
            f.write(f"C:/Users/leosc/Documents/_wichtigeDokumente/Bachelorarbeit/selective-synaptic-dampening-main/src/{path[1:]}\n")
            print(f"{path[1:]}")

def split_dataset_by_class(ds, root, num_classes_first_group, num_images):
    """num_classes_first_group is the number of classes i want to unlearn, this gets the amount of images
    to be unlearned using the class number. Then the first x images get alotted to the first dataset, the rest
    to the second one."""

    num_classes_first_group_start = int(num_classes_first_group.split("-")[0])-1
    num_classes_first_group_end = int(num_classes_first_group.split("-")[1])
    start_index = 0

    if not num_images:
        files = os.listdir(root)
        foldersInOrder = sorted(files, key=str)[:2*num_classes_first_group_end]
        for i in range(0,num_classes_first_group_start):
            target_folder_path = os.path.join(root, str(foldersInOrder[i]))
            start_index += num_items_in_folder(target_folder_path)
        last_index = start_index + 1
        for i in range(num_classes_first_group_start,num_classes_first_group_end):
            target_folder_path = os.path.join(root, str(foldersInOrder[i]))
            last_index += num_items_in_folder(target_folder_path)
        last_index -= 1
    else:
        last_index = num_images
    if(start_index >= last_index):
        print("FORGET CLASSES NOT CORRECT-----INDEX ERROR")
        _=0/0
    indices_first_group = range(start_index, last_index)
    print(f"forget set, indicies: {range(start_index, last_index + 1)}")
    # indices_second_group = range(0, start_index)
    indices_third_group = range(last_index + 1, 5822653)

    # Create subsets
    subset1 = Subset(ds, indices_first_group)
    # subset2 = Subset(ds, indices_second_group)    #Second group is usually very small if it even exists, so it is negletable 
    subset3 = Subset(ds, indices_third_group)

    # create_unlearning_image_list(ds,last_index)
    create_unlearning_image_list_noBorder(subset1)

    return subset1, subset3

def build_validset(ds, portion):
    indices_first_group = range(int(5822653*(1-portion)), 5822653)
    return Subset(ds, indices_first_group)


def plot_roc_curve(fpr, tpr):
    # Create the ROC curve plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label='ROC curve')
    plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line for random chance
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()

    # Log the ROC plot to wandb as an image
    roc_image = wandb.Image(plt, caption="ROC Curve")

    return roc_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-arch", type=str, default="iresnet100", help="net type")
    parser.add_argument("-weight_path",type=str, default="MagFace", help="The model weights") #choices=["MagFace","ArcFace", <WeightPath>]
    parser.add_argument("-gpu", action="store_true", default=True, help="use gpu or not")
    parser.add_argument("-warm", type=int, default=1, help="warm up training phase")
    parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
    parser.add_argument("-method",type=str, default="ssd_tuning" ,nargs="?",choices=["ssd_tuning","baseline"],help="select unlearning method from choice set",)
    parser.add_argument("-dataset",type=str,required=True,nargs="?",choices=["agedb", "lfw", "cfp", "Adience","Pins"],help="dataset to test on",)
    parser.add_argument("-epochs", type=int, default=1, help="number of epochs of unlearning method to use")
    parser.add_argument("-seed", type=int, default=0, help="seed for runs")
    parser.add_argument('-embedding_size', default=512, type=int,help='The embedding feature size')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',help="Batch size for Dataloaders")
    parser.add_argument('-resume', default=None, type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('-p', '--print-freq', default=100, type=int,metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-cpu-mode', action='store_true', help='Use the CPU.')
    parser.add_argument('-dist', default=1, help='use this if model is trained with dist')
    parser.add_argument('-forgetClasses', type=str ,default="1-1", help='Range of classes to forget')
    parser.add_argument('-alpha', type=str, default=-1, help='alpha used in ssd unlearning') # Supports values like "1,2,3,4,5" to run 5 experiments
    parser.add_argument('-lamda', type=str, default=-1, help='lambda used in ssd unlearning') # Supports values like "1,2,3,4,5" to run 5 experiments
    parser.add_argument('-ssd', type=str, default="lfssd", choices=["lfssd","ssd","lfssdv2"],help='lambda used in ssd unlearning')
    parser.add_argument('-runBaseline', action='store_true', help='If the baseline should be run or not')
    parser.add_argument('-noSSD', action='store_true', help='If the ssd should be run or not')
    parser.add_argument("-verificationMethod",type=str, default="both" ,choices=["MIA","PIC","Kolmogorov","both"],help="select unlearning verification method from choice set",)
    parser.add_argument('-forgetImages', type=int ,default=0, help='Number of images to forget, overrides forgetClasses')
    parser.add_argument('-distanceMethod', type=str ,default="cosine",  choices=["cosine","euclidian"], help='Number of images to forget, overrides forgetClasses')
    parser.add_argument('-sampling', type=str, default="none", help="If the importances/activations shold be only calculated from a sample off the dataset")
    args = parser.parse_args()

    sampling_count = 100
    if(args.sampling != "none"):
        sampling_count = int(args.sampling)

    magFacePath = "C:\\Users\\leosc\\Documents\\_wichtigeDokumente\\Bachelorarbeit\\MagFace\\eval\\eval_recognition\\magface_iresnet100_quality.pth"
    arcFacePath = "C:\\Users\\leosc\\Documents\\_wichtigeDokumente\\Bachelorarbeit\\selective-synaptic-dampening-main\\src\\checkpoint\\arcface_iresnet50_v1.0_pretrained\\arcface_iresnet50_v1.0_pretrained.pdparams"
    unlearnedPath = "C:\\Users\\leosc\\Documents\\_wichtigeDokumente\\Bachelorarbeit\\selective-synaptic-dampening-main\\src\\checkpoint\\class_1_ssd_tuning_0.5_4.0_unlearned.pth"
    aggregateResultsFile = "./AggregatedResults.txt"

    if (args.weight_path == "MagFace"):
        args.weight_path = magFacePath
    elif (args.weight_path == "ArcFace"):
        args.weight_path = arcFacePath
        args.arch = "iresnet50"
    elif(args.weight_path == "Unlearned1"):
        os.environ['Unlearned_Full_imp'] = "_unlearned1"
        args.weight_path = unlearnedPath
        args.arch = "iresnet50"
    else:
        if("iresnet50" in args.weight_path):
            args.arch = "iresnet50"
        print(f"Using chechpoint path {args.weight_path}")

    batch_size = args.batch_size
    args.resume = args.weight_path

    os.environ['SAMPLING'] = args.sampling
    os.environ['BASELINE_PATH'] = args.weight_path
    os.environ['distanceMethod'] = args.distanceMethod
    os.environ['BATCH_SIZE'] = str(batch_size)
    os.environ['ARCH'] = str(args.arch)

    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0., 0., 0.],
                std=[1., 1., 1.]),
    ])

    img_size = 112
    root= "./data/faces_emore"
    validRoot = "./data/casia"

    # PinsFaceRecognition is just an extension for ImageFolder in this case, it is used to load the Faces emore dataset
    trainset = datasets.PinsFaceRecognition(root=root, download=True, train=True, unlearning=True, img_size=img_size)
    validset = datasets.PinsFaceRecognition(root=validRoot, download=True, train=True, unlearning=False, img_size=img_size)
    # validset = build_validset(validset, 0.2)
    valid_dl = None
    valid_dl = DataLoader(validset, batch_size)

    if(args.forgetImages):
        args.forgetClasses = "0-0"

    forget_train, retain_train = split_dataset_by_class(trainset, root, args.forgetClasses, args.forgetImages)
    forget_range = args.forgetClasses
    args.forgetClasses = int(args.forgetClasses.split("-")[1])-int(args.forgetClasses.split("-")[0])+1

    os.environ['Forget_Samples'] = str(len(forget_train))

    forget_train_dl = DataLoader(forget_train, batch_size)
    retain_train_dl = DataLoader(retain_train, batch_size, shuffle=True)
    full_train_dl = DataLoader(
        ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)),
        batch_size=batch_size,
    )

    if(args.sampling != "none"):
        sampling_indicies = range(5822653 - sampling_count, 5822653)
        sample_dl = DataLoader(Subset(trainset, sampling_indicies), batch_size)
    else:
        sample_dl = None

    net = builder_inf(args)

    if args.gpu:
        net = net.cuda()

    model_size_scaler = 1
    dampening_constant = args.lamda                          # Dampening     | Smaller -> Bigger Change  | Values in (0,1 ; 10)   | Lambda in the Paper
    selection_weighting = args.alpha * model_size_scaler     # Threshold     | Smaller -> More Changes   | Values in (0,1 ; 100) | Alpha in the Paper

    if (args.lamda == -1):
        dampening_constants = [-0.9,-0.5,-0.3,-0.1]
    else:
        dampening_constants = [float(x) for x in args.lamda.split(",")]
    if (args.alpha == -1):
        selection_weightings = [0.25,0.5,1,2]
    else:
        selection_weightings = [float(x) for x in args.alpha.split(",")]

    if not args.noSSD:
        for dampening_constant in dampening_constants:
            for selection_weighting in selection_weightings:
                kwargs = {
                    "model": copy.deepcopy(net),
                    "retain_train_dl": retain_train_dl,
                    "forget_train_dl": forget_train_dl,
                    "full_train_dl": full_train_dl,
                    "sample_dl": sample_dl,
                    "valid_dl": valid_dl,
                    "dampening_constant": dampening_constant,
                    "selection_weighting": selection_weighting,
                    "num_classes": args.embedding_size,
                    "dataset_name": args.dataset,
                    "device": "cuda" if args.gpu else "cpu",
                    "model_name": args.arch,
                    "method": args.method,
                    "forget_perc": args.forgetClasses,
                    "version": "class",
                    "ssd_version": args.ssd,
                    "verificationMethod": args.verificationMethod
                }

                
                print(Fore.GREEN + f"Running Run Alpha {selection_weighting}, Lambda {dampening_constant}" + Style.RESET_ALL)
                start = time.time()
                mean_acc, std_acc, tpr, fpr, cosSim_Unlearned, cosSim_Reference, mia, D_f_r, mia_method = getattr(forget_class_extractor_strategies_true, args.method)(**kwargs)
                end = time.time()

                time_elapsed = end - start

                print(Fore.BLUE + "Results")
                print(
                    {
                        "Mean Verification Accuracy": mean_acc,
                        "Verification Accuracy Std": std_acc,
                        "Cosine Similarity Unlearned": cosSim_Unlearned,
                        "Cosine Similarity Reference": cosSim_Reference,
                        f"{mia_method}": mia,
                        "D_f_r": D_f_r,
                        "MethodTime": time_elapsed,
                    }
                )
                print(Style.RESET_ALL)

                with open(aggregateResultsFile, "a") as f:
                    f.write(f"{args.arch}_{args.dataset}_{args.ssd}_forget_classes_{forget_range}_lambda:{dampening_constant}_alpha:{selection_weighting}_{mia_method}: {D_f_r}, {cosSim_Unlearned}, {cosSim_Reference}, {int(time_elapsed)}, {std_acc}, {mean_acc}\n")


    #################### Baseline ########################
    if (args.runBaseline):
        kwargs = {
                    "model": copy.deepcopy(net),
                    "retain_train_dl": retain_train_dl,
                    "forget_train_dl": forget_train_dl,
                    "full_train_dl": full_train_dl,
                    "valid_dl": valid_dl,
                    "dampening_constant": dampening_constant,
                    "selection_weighting": selection_weighting,
                    "num_classes": args.embedding_size,
                    "dataset_name": args.dataset,
                    "device": "cuda" if args.gpu else "cpu",
                    "model_name": args.arch,
                    "method": args.method,
                    "forget_perc": args.forgetClasses,
                    "savePath": args.weight_path,
                    "version": "class",
                    "verificationMethod": args.verificationMethod
                }
        os.environ['ARCH'] = f"{args.arch}_{args.ssd}"
        os.environ['PROB_DATASET'] = f"Partial_class_{args.forgetClasses}"

        print(Fore.GREEN + f"Running Run Baseline" + Style.RESET_ALL)
        start = time.time()
        mean_acc, std_acc, tpr, fpr, cosSim_Unlearned, cosSim_Reference, mia, D_f_r, mia_method = getattr(forget_class_extractor_strategies_true, "baseline")(**kwargs)
        end = time.time()

        time_elapsed = end - start

        print(Fore.BLUE)
        print(
            {
                "Mean Verification Accuracy": mean_acc,
                "Verification Accuracy Std": std_acc,
                "Cosine Similarity Unlearned": cosSim_Unlearned,
                "Cosine Similarity Reference": cosSim_Reference,
                f"{mia_method}": mia,
                "D_f_r": D_f_r,
                "MethodTime": time_elapsed,
            }
        )
        print(Style.RESET_ALL)

        with open(aggregateResultsFile, "a") as f:
            f.write(f"{args.arch}_{args.dataset}_{args.ssd}_forget_classes_{forget_range}_Baseline_{mia_method}: {mean_acc}, {mia}, {D_f_r}, {cosSim_Unlearned}, {cosSim_Reference}, {time_elapsed}, {std_acc}\n")
