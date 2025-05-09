"""
From https://github.com/vikram2000b/bad-teaching-unlearning / https://arxiv.org/abs/2205.08096
"""

from torch.nn import functional as F
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import os
import re
import pickle
import matplotlib.pyplot as plt
import paddle.nn as nnp
import torch.nn as nn
from sklearn.manifold import TSNE

def JSDiv(p, q):
    m = (p + q) / 2
    return 0.5 * F.kl_div(torch.log(p), m) + 0.5 * F.kl_div(torch.log(q), m)


# ZRF/UnLearningScore https://arxiv.org/abs/2205.08096
def UnLearningScore(tmodel, gold_model, forget_dl, batch_size, device):
    model_preds = []
    gold_model_preds = []
    with torch.no_grad():
        for batch in forget_dl:
            x, y, cy = batch
            x = x.to(device)
            model_output = tmodel(x)
            gold_model_output = gold_model(x)
            model_preds.append(F.softmax(model_output, dim=1).detach().cpu())
            gold_model_preds.append(F.softmax(gold_model_output, dim=1).detach().cpu())

    model_preds = torch.cat(model_preds, axis=0) # type: ignore
    gold_model_preds = torch.cat(gold_model_preds, axis=0) # type: ignore
    return 1 - JSDiv(model_preds, gold_model_preds)


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

def batch_entropy(p, batch_size=10000, dim=-1, keepdim=False):
    results = []
    total_batches = (p.size(0) + batch_size - 1) // batch_size
    for i in tqdm(range(0, p.size(0), batch_size), desc="Entropy Calculation", total=total_batches):
        batch = p[i : i + batch_size]
        entropy_batch = -torch.where(batch > 0, batch * batch.log(), batch.new([0.0])).sum(dim=dim, keepdim=keepdim)
        results.append(entropy_batch.cpu())
    return torch.cat(results)

def collect_prob(data_loader, model, current):
    print(f"Determining parobabilities for {current} Dataset")
    batch_size = 32
    data_loader = torch.utils.data.DataLoader(
        data_loader.dataset, batch_size=batch_size, shuffle=False  # Batch size 1?
    )
    prob = []
    part = 0
    i = 0

    # Get already calculated probabilities and make a subset of the remaining data to be computed
    currentMode = os.getenv('PROB_DATASET',"None")  # "Full" or "Partial_class_{amount_of_forget_classes}"
    arch = os.getenv('ARCH',"UNDEFINED")            # Architecture ("iResnet100")
    os.makedirs(f"./MIA/{arch}", exist_ok=True)
    savePath = f"./MIA/{arch}/MS1M_V2_{currentMode}_{current}_part"
    rangeOfAlreadyCalculated = [-1]
    all_files = os.listdir(f"./MIA/{arch}")
    file_list = [f for f in all_files if f.startswith(f"MS1M_V2_{currentMode}_{current}_part_")]
    for file_name in file_list:
        match = re.search(r'part_(/d+)$', file_name)
        if match:
            rangeOfAlreadyCalculated.append(int(match.group(1)))
        else:
            print(f"No number found in {file_name}")
    part = max(rangeOfAlreadyCalculated)
    print(f"Already found parts calculated up to {part}")
    startIndex = ((part + 1) * 15001)
    part += 1
    i = startIndex
    endIndex = (len(data_loader) - 1) * batch_size
    if(startIndex):
        subset_indices = range(startIndex * batch_size, endIndex - 1)
        subset_dataset = torch.utils.data.Subset(data_loader.dataset, subset_indices)
        data_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)

    if startIndex * batch_size < endIndex:
        with torch.no_grad():
            for batch in tqdm(data_loader):
                i += 1
                batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
                data, _, _ = batch
                output = model(data)
                prob.append(F.softmax(output, dim=-1).data)
                if i % 5000 == 0:
                    torch.cuda.empty_cache()
                elif i % 15001 == 0:
                    torch.save(torch.cat(prob), f"{savePath}_{part}")
                    prob.clear()
                    prob = []
                    part += 1
        if prob:
            torch.save(torch.cat(prob), f"{savePath}_{part}")
            part += 1
    else:
        print("Skipping calculation sice all parts are already calculated")

    prob = []
    savePath = f"./MIA/{arch}/MS1M_V2_{currentMode}_{current}_part"
    for i in range(part):
        print(f"Collecting partial results {i}")
        tmp_prob = torch.load(f"{savePath}_{i}", map_location="cpu")
        prob.append(tmp_prob)

    return torch.cat(prob)


# https://arxiv.org/abs/2205.08096
def get_membership_attack_data(retain_loader, forget_loader, test_loader, model):
    headPath = "C:/Users/leosc/Documents/_wichtigeDokumente/Bachelorarbeit/selective-synaptic-dampening-main/src/checkpoint/arcface_iresnet50_v1.0_pretrained/rank-0_softmax_weight.pkl"
    model = addClassificationHead(model, headPath)

    retain_prob = collect_prob(retain_loader, model, "retain")
    forget_prob = collect_prob(forget_loader, model, "forget")
    test_prob = collect_prob(test_loader, model, "valid")

    print("Collected all the Probabilities")

    forget_prob_len = len(forget_prob)
    retain_prob_len = len(retain_prob)
    test_prob_len = len(test_prob)

    forget_entropy = batch_entropy(forget_prob)
    del forget_prob
    print("Calculated the forget entropy")
    retain_entropy = batch_entropy(retain_prob)
    del retain_prob
    print("Calculated the retain entropy")
    test_entropy = batch_entropy(test_prob)
    del test_prob
    print("Calculated the test entropy")

    X_r = (
        torch.cat([retain_entropy, test_entropy])
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    Y_r = np.concatenate([np.ones(retain_prob_len), np.zeros(test_prob_len)])

    X_f = forget_entropy.cpu().numpy().reshape(-1, 1)
    
    # I commented Y_f out as it is not used in the original code
    # Y_f = np.concatenate([np.ones(forget_prob_len)])

    del forget_entropy
    del retain_entropy
    del test_entropy

    return X_f, X_r, Y_r#, Y_f


# https://arxiv.org/abs/2205.08096
def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model):
    X_f, X_r, Y_r = get_membership_attack_data( #, Y_f
        retain_loader, forget_loader, test_loader, model
    )
    print("Fitting Logistic Regression")
    # clf = SVC(C=3,gamma='auto',kernel='rbf')
    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs", multi_class="multinomial"
    )
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    returnValue = results.mean()
    print(f"MIA: {returnValue}")
    return returnValue


@torch.no_grad()
def actv_dist(model1, model2, dataloader, device="cuda"):
    sftmx = nn.Softmax(dim=1) # type: ignore
    distances = []
    for batch in dataloader:
        x, _, _ = batch
        x = x.to(device)
        model1_out = model1(x)
        model2_out = model2(x)
        diff = torch.sqrt(
            torch.sum(
                torch.square(
                    F.softmax(model1_out, dim=1) - F.softmax(model2_out, dim=1)
                ),
                axis=1,
            ) # type: ignore
        )
        diff = diff.detach().cpu()
        distances.append(diff)
    distances = torch.cat(distances, axis=0) # type: ignore
    return distances.mean()

def cosineSimilarity(featPath1, featPath2):
    feat1 = loadFeatures(featPath1)
    feat2 = loadFeatures(featPath2)

    if(os.getenv('distanceMethod',"cosine") == "euclidian"):
        print("Using euclidian")
        cos_similarities = np.linalg.norm(feat1-feat2, axis=1)
    else:
        cos_similarities = compute_cosine_similarity(feat1, feat2)

    mean_similarity = np.mean(cos_similarities)

    return cos_similarities, mean_similarity

def compute_cosine_similarity(features_before, features_after):
    if len(features_before) != len(features_after):
        print(len(features_before))
        print(len(features_after))
        raise ValueError("Feature arrays must have the same length")
    
    similarities = []
    for feat1, feat2 in tqdm(zip(features_before, features_after), total=len(features_before), desc="Computing Cosine Similarity"):
        dot_product = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        similarity = dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0
        similarities.append(similarity)
    
    return np.array(similarities)

def loadFeatures(featPath, amount = -1):
    features = []
    with open(featPath) as f:
        ls = f.readlines()
    endIndex = amount
    if(amount == -1):
        endIndex = len(ls)
    for idx in range(endIndex):
        tmp = ls[idx].split(".jpg")[1].strip().split(" ")
        features.append([float(x) for x in tmp])
    
    return np.array(features, dtype=np.float32)

# def makeTSNE(features_list, title):
#     feat = loadFeatures(features_list)
#     tsne = TSNE(n_components=2, perplexity=30, random_state=42)
#     X_tsne = tsne.fit_transform(feat)

#     plt.figure(figsize=(10, 8))
#     scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=50, alpha=0.7)
#     plt.title("t-SNE on Face Image Dataset")
#     plt.xlabel("Component 1")
#     plt.ylabel("Component 2")
#     plt.grid(True)
#     plt.colorbar(scatter)
#     plt.savefig(f"./tsne/{title}.png", dpi=300, bbox_inches='tight')

def makeTSNE(features_list1, features_list2, title):
    # Load both feature sets
    feat1 = loadFeatures(features_list1)
    feat2 = loadFeatures(features_list2)

    # Combine features
    all_features = np.vstack((feat1, feat2))

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(all_features)

    # Split transformed features
    X1_tsne = X_tsne[:len(feat1)]
    X2_tsne = X_tsne[len(feat1):]

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(X1_tsne[:, 0], X1_tsne[:, 1], s=50, alpha=0.7, label='Set 1', color='blue')
    plt.scatter(X2_tsne[:, 0], X2_tsne[:, 1], s=50, alpha=0.7, label='Set 2', color='orange')
    plt.title("t-SNE on Face Image Dataset")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./tsne/{title}.png", dpi=300, bbox_inches='tight')

# Add the classification Head
class ModelWithHead(nn.Module):
    def __init__(self, backbone, classifier):
        super(ModelWithHead, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        features = self.backbone(x)  # Backbone extracts features
        logits = self.classifier(features)  # Classification head
        return logits

def addClassificationHead(model, headPath):
    if os.path.isfile(headPath):
        with open(headPath, "rb") as f:
            softmax_weights = pickle.load(f)
        num_classes = softmax_weights.shape[0]
        hidden_size = softmax_weights.shape[1]
        classifier = nn.Linear(hidden_size, num_classes)
        classifier.weight = nn.Parameter(torch.tensor(softmax_weights, dtype=classifier.weight.dtype).to("cuda"))
        model = ModelWithHead(model, classifier)
        return model
    print("Not a valid HeadPath")
    a = 0/0
    return model