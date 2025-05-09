'Entries: 1/50 (Not Modified Often)'

"""
Original Leo Scheubel "Selective Synaptic Damplening in Face Recognition"
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import datasets
import models  # SSD models
import os
import argparse
import conf

# Configuration (adjust as needed)
CHECKPOINT_PATH = 'C:\\Users\\leosc\\Documents\\_wichtigeDokumente\\Bachelorarbeit\\selective-synaptic-dampening-main\\src\\checkpoint\\ResNet18\\Cifar100\\ssd_unlearned.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64  # Batch size for evaluation

# CIFAR-100 class names mapping
CLASSES = conf.class_dict
parser = argparse.ArgumentParser(description="Evaluate model on a specific class.")
parser.add_argument("-c", type=str, required=True, help="Class name to evaluate (e.g. 'rocket').")
parser.add_argument("-checkpoint", type=str, required=False, help="(Relative) Path to the checkpoint")
parser.add_argument("-net", type=str, required=True, help="net type")
parser.add_argument("-dataset", type=str, required=True, help="dataset to train on")
parser.add_argument("-classes", type=int, required=True, help="number of classes")
args = parser.parse_args()

net = getattr(models, args.net)(num_classes=args.classes)
net = net.to(DEVICE)

CHECKPOINT_PATH = args.checkpoint
CLASS_NAME = args.c

root = "./data/105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"
img_size = 224 if args.net == "ViT" else 32

# Load the checkpoint
if os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint)
    print(f"Checkpoint loaded successfully from '{CHECKPOINT_PATH}'")
else:
    raise FileNotFoundError(f"Checkpoint file not found at '{CHECKPOINT_PATH}'")

# Set the model to evaluation mode
net.eval()

# Load Testset
testset = getattr(datasets, args.dataset)(
    root=root, download=True, train=False, unlearning=False, img_size=img_size
)

try:
    CLASS_IDX = int(args.c)
except:
    if(args.c in CLASSES):
        CLASS_IDX = CLASSES[args.c]
    else:
        try:
            CLASS_IDX = testset.class_to_idx[args.c]
        except:
            raise ValueError(f"Invalid class.")

# Filter the dataset for the class
test_testset = [item for item in testset if item[2] == CLASS_IDX] if args.dataset == "PinsFaceRecognition" else [item for item in testset if item[1] == CLASS_IDX]

# Create a DataLoader for the rocket class subset
testloader = DataLoader(test_testset, batch_size=BATCH_SIZE, shuffle=False)

# Function to evaluate the model on the rocket class subset
def evaluate_model_on_class():
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for testing
        for batch in testloader:
            # If dataset is "PinsFaceRecognition", expect three values; otherwise, two.
            if args.dataset == "PinsFaceRecognition":
                images, _, labels = batch
            else:
                images, labels = batch
            
            # Move data to the device
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            outputs = net(images)
            
            # Get the predictions
            _, predicted = outputs.max(1)
            
            # Update total and correct counts
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on '{args.c}' class: {accuracy:.2f}%")

# Run the evaluation
if __name__ == "__main__":
    print(f"Evaluating model on the '{args.c}' class...")
    evaluate_model_on_class()
