'Entries: 50/50 (Modified A Lot)'

#!/usr/bin/env python
import sys
sys.path.append("..")
import iresnet
from collections import OrderedDict
from tqdm import tqdm
from termcolor import cprint
import os
import torch.nn.functional as F
import torch.nn as nn
import torch
import paddle

def load_features(args):
    if args.arch == 'iresnet34':
        features = iresnet.iresnet34(
            pretrained=False,
            num_classes=args.embedding_size,
        )
    elif args.arch == 'iresnet18':
        features = iresnet.iresnet18(
            pretrained=False,
            num_classes=args.embedding_size,
        )
    elif args.arch == 'iresnet50':
        features = iresnet.iresnet50(
            pretrained=False,
            num_classes=args.embedding_size,
        )
    elif args.arch == 'iresnet100':
        features = iresnet.iresnet100(
            pretrained=False,
            num_classes=args.embedding_size,
        )
    else:
        raise ValueError()
    return features


class NetworkBuilder_inf(nn.Module):
    def __init__(self, args):
        super(NetworkBuilder_inf, self).__init__()
        self.features = load_features(args)

    def forward(self, input):
        # add Fp, a pose feature
        x = self.features(input)
        return x
    
def load_dict_inf(args, model):
    if os.path.isfile(args.resume):
        cprint('=> loading pth from {} ...'.format(args.resume))
        if args.cpu_mode:
            checkpoint = torch.load(args.resume, map_location=torch.device("cpu"))
        else:
            checkpoint = torch.load(args.resume)
        _state_dict = clean_dict_inf(model, checkpoint['state_dict'])
        model_dict = model.state_dict()
        model_dict.update(_state_dict)
        model.load_state_dict(model_dict)
        # delete to release more space
        del checkpoint
        del _state_dict
    else:
        sys.exit("=> No checkpoint found at '{}'".format(args.resume))
    return model

def clean_dict_inf(model, state_dict):
    _state_dict = OrderedDict()
    for k, v in state_dict.items():
        try:
            v_size = v.size()
        except:
            v_size = torch.Size(v.shape)
        # assert k[0:1] == 'features.module.'
        new_k = 'features.'+'.'.join(k.split('.')[2:])
        new_kk = '.'.join(k.split('.')[1:])
        if new_k in model.state_dict().keys() and \
           v_size == model.state_dict()[new_k].size():
            _state_dict[new_k] = v
        # assert k[0:1] == 'module.features.'
        elif new_kk in model.state_dict().keys() and \
           v_size == model.state_dict()[new_kk].size():
            _state_dict[new_kk] = v
        elif k in model.state_dict().keys() and \
           v_size == model.state_dict()[k].size():
            _state_dict[k] = v
    arch = os.getenv('ARCH',"UNDEFINED")
    if("100" in arch or "Mag" in arch):
        curr_key = ""
        curr_bias = ""
        curr_weight = ""
        diffs = model.state_dict().keys() - _state_dict.keys()
        try:
            for key in diffs:
                curr_key = key
                if "prelu.bias" in key:
                    curr_bias = key.replace("bias","weight").replace("features.","features.module.")
                    _state_dict[key] = torch.zeros_like(state_dict[curr_bias])
                elif "prelu.prelu" in key:
                    curr_weight = key.replace("prelu.prelu","prelu").replace("features.","features.module.")
                    _state_dict[key] = state_dict[curr_weight]
        except:
            print("ERROR?????????????????????????????????????")
            # print(model.state_dict().keys())
            # print("========")
            # print(_state_dict.keys())
            # print("========")
            print(diffs)
            print("========") # The key has to be in this
            prelus = [key for key in state_dict.keys() if "prelu" in key]
            print(prelus)
            print("========") # The base key
            print(curr_key)
            print("========") # The key that is looked for in case of a bias
            print(curr_bias)
            print("========") # The key that is looked for in case of a weight
            print(curr_weight)
            a = 0/0
    num_model = len(model.state_dict().keys())
    num_ckpt = len(_state_dict.keys())
    if num_model != num_ckpt:
        # print(model.state_dict().keys() - _state_dict.keys())
        sys.exit("=> Not all weights loaded, model params: {}, loaded params: {}".format(
            num_model, num_ckpt))
    return _state_dict


def builder_inf(args):
    try:
        model = NetworkBuilder_inf(args)
        # Used to run inference
        model = load_dict_inf(args, model)
    except:
        import iresnet_arc
        print("Loading iresnet50 from argface")
        try:
            checkpoint = paddle.load(args.resume)
            torch_checkpoint = {}
            prelu_weights_to_initialize = []
            for key, value in checkpoint.items(): # type: ignore
                if isinstance(value, paddle.Tensor):
                    value = torch.tensor(value.numpy())
                
                # Rename keys for BatchNorm and PReLU
                if "._mean" in key:
                    key = key.replace("._mean", ".running_mean")
                if "._variance" in key:
                    key = key.replace("._variance", ".running_var")
                if "._weight" in key:
                    key = key.replace("._weight", ".weight")
                if key == "fc.weight":
                    value = value.t()
                if "prelu" in key and not "prelu.prelu" in key:
                    bias = key.replace("weight","bias")
                    if(bias not in checkpoint): # type: ignore
                        prelu_weights_to_initialize.append(bias)
                    key = key.replace("prelu", "prelu.prelu")
        
                torch_checkpoint[key] = value
            
            for prelu_weight in prelu_weights_to_initialize:
                weight = prelu_weight.replace("bias", "prelu.weight")
                torch_checkpoint[prelu_weight] = torch.zeros_like(torch_checkpoint[weight])

            model = iresnet_arc.iresnet50()
            model.load_state_dict(torch_checkpoint)
        except:
            checkpoint = torch.load(args.resume)
            model = iresnet_arc.iresnet50()
            model.load_state_dict(checkpoint['state_dict'])
    return model
