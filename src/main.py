import torch
import argparse
import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datasets import load_dataset

from Noise_Generator import Noise_Generator
from my_utils import PSNR, SSIM, normalize_image, restore_image

def get_dataset(dataset,splits):
    train, val, test = None, None, None
    for split in splits:
        ds = load_dataset("GATE-engine/mini_imagenet",split=split)
        ds.set_format('torch',columns=['image','label'])
        if 'train' == split:
            train = ds['image']
        if 'validation' == split:
            val  = ds['image']
        if 'test' == split:
            test = ds['image']
    return (train, val, test)


if __name__=="__main__":
    noise_type = 'gaussian'
    noise_level = 30
    noise_generator = Noise_Generator(noise_type,noise_level)
    # dataset = "GATE-engine/mini_imagenet"
    # splits = ['validation']
    # train, val, test = get_dataset(dataset, splits)
    # print(val['image'])
    
    test_img = cv2.imread("FFDNet_pytorch/test_data/color.png")
    print(test_img)
    test_img = torch.tensor(test_img)
    test_img = torch.permute(test_img,(2,0,1))
    noisy_img = noise_generator.add_noise(test_img)
    plt.figure()
    plt.subplot(121)
    plt.imshow(torch.permute(test_img,(1,2,0)))
    plt.title("Input image")
    plt.subplot(122)
    plt.imshow(torch.permute(noisy_img,(1,2,0)))
    plt.title("Noisy image")
    plt.show()



