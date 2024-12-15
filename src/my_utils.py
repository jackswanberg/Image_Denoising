import numpy as np
import torch

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def PSNR(img, noisy_img):
    if not isinstance(img,np.ndarray):
        img = img.cpu().detach().numpy()
    if not isinstance(noisy_img,np.ndarray):
        noisy_img = noisy_img.cpu().detach().numpy()
    return peak_signal_noise_ratio(img,noisy_img)

def SSIM(img,noisy_img):
    if not isinstance(img,np.ndarray):
        img = img.cpu().detach().numpy()
    if not isinstance(noisy_img,np.ndarray):
        noisy_img = noisy_img.cpu().detach().numpy()
    return structural_similarity(img,noisy_img,channel_axis=2)

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

def downsample(x):
    """
    :param x: (C, H, W)
    :param noise_sigma: (C, H/2, W/2)
    :return: (4, C, H/2, W/2)
    """
    # x = x[:, :, :x.shape[2] // 2 * 2, :x.shape[3] // 2 * 2]
    N, C, W, H = x.size()
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

    Cout = 4 * C
    Wout = W // 2
    Hout = H // 2

    if 'cuda' in x.type():
        down_features = torch.cuda.FloatTensor(N, Cout, Wout, Hout).fill_(0)
    else:
        down_features = torch.FloatTensor(N, Cout, Wout, Hout).fill_(0)
    
    for idx in range(4):
        down_features[:, idx:Cout:4, :, :] = x[:, :, idxL[idx][0]::2, idxL[idx][1]::2]

    return down_features

def upsample(x):
    """
    :param x: (n, C, W, H)
    :return: (n, C/4, W*2, H*2)
    """
    N, Cin, Win, Hin = x.size()
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]
    
    Cout = Cin // 4
    Wout = Win * 2
    Hout = Hin * 2

    up_feature = torch.zeros((N, Cout, Wout, Hout))
    for idx in range(4):
        up_feature[:, :, idxL[idx][0]::2, idxL[idx][1]::2] = x[:, idx:Cin:4, :, :]

    return up_feature