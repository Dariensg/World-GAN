import random
import pickle
import numpy as np
import torch
import torchvision
from torch.nn.functional import interpolate, grid_sample
import matplotlib.pyplot as plt


def set_seed(seed=0):
    """ Set the seed for all possible sources of randomness to allow for reproduceability. """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def interpolate3D(data, shape, mode='bilinear', align_corners=False):
    d_1 = torch.linspace(-1, 1, shape[0])
    d_2 = torch.linspace(-1, 1, shape[1])
    d_3 = torch.linspace(-1, 1, shape[2])
    meshz, meshy, meshx = torch.meshgrid((d_1, d_2, d_3))
    grid = torch.stack((meshx, meshy, meshz), 3)
    grid = grid.unsqueeze(0).to(data.device)

    scaled = grid_sample(data, grid, mode=mode, align_corners=align_corners)
    return scaled


def save_pkl(obj, name, prepath='output/'):
    with open(prepath + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(name, prepath='output/'):
    with open(prepath + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def get_discriminator1_scaling_tensor(opt, outputD1, ignore_nan=False):
    if (opt.alpha_layer_type == "half-and-half"):

        first_half = torch.tensor([np.nan if opt.use_nan and not ignore_nan else 0.]).expand(outputD1.size()).split(math.floor(outputD1.size()[4] / 2), dim=4)
        second_half = torch.tensor([1.]).expand(outputD1.size()).split(math.ceil(outputD1.size()[4] / 2), dim=4)
        
        return torch.cat((first_half[0], second_half[0]), dim=4).to(opt.device)
    elif (opt.alpha_layer_type == "all-ones"):
        return torch.ones_like(outputD1).to(opt.device)
    elif (opt.alpha_layer_type == "all-zeros"):
        
        if (opt.use_nan and not ignore_nan):
            return torch.tensor([np.nan]).expand(outputD1.size()).to(opt.device)

        return torch.zeros_like(outputD1).to(opt.device)

def get_discriminator2_scaling_tensor(opt, outputD2, ignore_nan=False):
    if (opt.alpha_layer_type == "half-and-half"):
        first_half = torch.tensor([1.]).expand(outputD2.size()).split(math.floor(outputD2.size()[4] / 2), dim=4)
        second_half = torch.tensor([np.nan if opt.use_nan and not ignore_nan else 0.]).expand(outputD2.size()).split(math.ceil(outputD2.size()[4] / 2), dim=4)
        
        return torch.cat((first_half[0], second_half[0]), dim=4).to(opt.device)
    elif (opt.alpha_layer_type == "all-ones"):

        if (opt.use_nan and not ignore_nan):
            return torch.tensor([np.nan]).expand(outputD2.size()).to(opt.device)

        return torch.zeros_like(outputD2).to(opt.device)
    elif (opt.alpha_layer_type == "all-zeros"):
        return torch.ones_like(outputD2).to(opt.device)
    
def get_lerping_tensor(opt, output):
    if (opt.alpha_layer_type == "half-and-half"):
        first_half = torch.tensor([0.]).expand(output.size()).split(math.floor(output.size()[4] / 2), dim=4)
        second_half = torch.tensor([1.]).expand(output.size()).split(math.ceil(output.size()[4] / 2), dim=4)

        return torch.cat((first_half[0], second_half[0]), dim=4).to(opt.device)
    elif (opt.alpha_layer_type == "all-ones"):
        return torch.ones_like(output).to(opt.device)
    elif (opt.alpha_layer_type == "all-zeros"):
        return torch.zeros_like(output).to(opt.device)