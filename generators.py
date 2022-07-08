import os
import sys

from typing import Tuple

import numpy as np
import torch

sys.path.append('./stylegan3')

import PIL.Image
import dnnlib
import legacy

device = torch.device('cuda')


def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


def generate_styleganv2(model_path, n_sample, batch_size):
    # with dnnlib.util.open_url(network_pkl) as f:
    with open(model_path, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore


    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    # Generate images.
    latents = torch.randn(n_sample, G.z_dim).to(device)
    img_batches = []
    n_batch = n_sample // batch_size + 1
    for batch_idx in range(n_batch):
        print('Stylegan generating image for batch %d (%d/%d) ...' % (batch_idx, batch_idx, n_batch))

        # img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = G(latents[batch_idx: (batch_idx + 1) * batch_size], label)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_batches.append(img)
    imgs_tensor = torch.cat(img_batches)
    imgs_pil = []
    for img in imgs_tensor:
        pil = PIL.Image.fromarray(img.cpu().numpy(), 'RGB')
        imgs_pil.append(pil)

    return imgs_tensor, imgs_pil


if __name__ == '__main__':
    imgs_tensor, imgs_pil = generate_styleganv2('./stylegan3/stylegan2-ffhq-256x256.pkl', 23, 5)
    print(images)