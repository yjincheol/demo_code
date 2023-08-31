import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.transforms as transforms
from torchvision import datasets

import model 
import utils
import data

parser = argparse.ArgumentParser(description='Quantization basic')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--histogram', action='store_true',
                    help='histogram of parmater distributions')

###Quantization Options ###
parser.add_argument('bit_W',default=32, type=int,
                    help='weight bit option')
parser.add_argument('bit_A',default=32, type=int,
                    help='activation bit option')
parser.add_argument('mode_W',default='sym', type=str,
                    help='weight value using sym or asym method')
parser.add_argument('mode_A',default='asym', type=str,
                    help='activation value using sym or asym method')
parser.add_argument('per_ch', action='store_true',
                    help='granularity : hardware recomended /per tensor(activation) or perchannel(weight)')
parser.add_argument('-bn_fold', acttion='stor_true',
                    help='convolution, BatchNorm folding')

def hist_distribution(layername):
    max_val = layername.flatten().max
    plt.hist(layername.flatten(), bins=1000)
    plt.xlabel('value')
    plt.ylabel('count')
    plt.xlim([-max_val,max_val]) 
    plt.title('fp value')
    plt.grid()
    plt.savefig(layername+'.png')

def weight_asym_quantization_method(bit_W, value):
    ### basic method ###
    qmin = 0.
    qmax = 2. ** bit_W - 1.
    scale = (max_val - min_val) / (qmax - qmin)

    zero_point = qmax - max_val / scale

    if zero_point < qmin:
        zero_point = torch.tensor([qmin], dtype=torch.float32).to(min_val.device)
    elif zero_point > qmax:
        # zero_point = qmax
        zero_point = torch.tensor([qmax], dtype=torch.float32).to(max_val.device)
    
    zero_point.round_()

    return scale, zero_point
    max_val = np.max(value)
    min_val = np.min(value)
    scale = (max_val-min_val)/(2**(bit)-1)
    z_p = np.round(np.abs(min_val)/scale)
    q_value = np.clamp(np.round(value)/scale,min_val,max_val)
def main():
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    for name, param in model.named_parameters():
        layername = '.'.join(name.split('.'))[:1]
        if layername == 'conv2':
            conv2 = param.cpu().detach().numpy()
            if args.histogram:
               hist_distribution(layername=conv2)
if __name__ == '__main__':
    main()