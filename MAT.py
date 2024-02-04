import cv2
import pyspng
import glob
import os
import re
import random
from typing import List, Optional

import click
import dnnlib 
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as v2

import legacy 
from networks.mat import Generator


def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def init_gan_model(resolution=512, network_pkl='pretrain/CelebA-HQ_512.pkl'):
    print('Initializing MAT model.....')
    print(f'Loading networks from: {network_pkl}')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False) # type: ignore
    net_res = 512 if resolution > 512 else resolution
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=3).to(device).eval().requires_grad_(False)
    copy_params_and_buffers(G_saved, G, require_all=True)
    print('Successfully Initialized')
    return G


def generate_images(
    image,
    mask,
    G,
    resolution = 512,
    truncation_psi = 1,
    noise_mode = 'const',
):
    """
    Generate images using pretrained network pickle.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ori_width = image.width
    ori_height = image.height

    # no Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    def read_image(image):
        image = np.array(image)
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
            image = np.repeat(image, 3, axis=2)
        image = image.transpose(2, 0, 1) # HWC => CHW
        image = image[:3]
        return image

    def to_image(image, lo, hi):
        image = np.asarray(image, dtype=np.float32)
        image = (image - lo) * (255 / (hi - lo))
        image = np.rint(image).clip(0, 255).astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        return image

    if resolution != 512:
        noise_mode = 'random'
    with torch.no_grad():
        image = read_image(image)
        image = (torch.from_numpy(image).float().to(device) / 127.5 - 1).unsqueeze(0)
        image = v2.Resize(size=(resolution, resolution), antialias=True)(image)

        mask = mask.convert('L')
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask).float().to(device).unsqueeze(0).unsqueeze(0)
        mask = v2.Resize(size=(resolution, resolution), antialias=True)(mask)

        z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
        output = G(image, mask, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        
        output = v2.Resize(size=(ori_height, ori_width), antialias=True)(output)
        output = torch.nn.functional.interpolate(output, size=(ori_height, ori_width), mode='bicubic', align_corners=False)
        
        output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
        output = output[0].cpu().numpy()
        
        pil_image = PIL.Image.fromarray(output, 'RGB')
        
    return pil_image