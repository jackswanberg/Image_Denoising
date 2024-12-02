import numpy as np
import torch

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def PSNR(img, noisy_img):
    img = img.numpy()
    noisy_img = noisy_img.numpy()
    return peak_signal_noise_ratio(img,noisy_img)

def SSIM(img,noisy_img):
    img = img.numpy()
    noisy_img = noisy_img.numpy()
    return structural_similarity(img,noisy_img)

def normalize_image(img):
    shift = img.min()
    img = img-shift
    scale = 255/img.max()
    norm_img = (img)*scale
    return norm_img.to(torch.uint8),shift,scale

def restore_image(img,shift,scale):
    img = img/scale+shift
    img[img<0]=0
    img[img>255]=255
    return img.to(torch.uint8)