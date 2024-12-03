import torch
import torchvision
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
from PIL import Image

from Noise_Generator import Noise_Generator
from my_utils import PSNR, SSIM, normalize_image, restore_image
from model import FFDNet

def get_dataloaders(dataset,splits):
    train_dataloader, val_dataloader, test_dataloader = None, None, None
    ds = load_dataset("GATE-engine/mini_imagenet")
    return ds
    # for split in splits:
    #     ds = load_dataset("GATE-engine/mini_imagenet",split=split)
    #     ds.set_format('torch',columns=['image','label'])
    #     if 'train' == split:
    #         train_dataloader = DataLoader(ds['image'],1)
    #     if 'validation' == split:
    #         val_dataloader  = DataLoader(ds['image'],1)
    #     if 'test' == split:
    #         test_dataloader = DataLoader(ds['image'])
    # return (train_dataloader, val_dataloader, test_dataloader)


if __name__=="__main__":
    noise_type = 'gaussian'
    noise_level = 30
    noise_generator = Noise_Generator(noise_type,noise_level)
    dataset = "GATE-engine/mini_imagenet"
    # dataset = "ioxil/imagenetsubset"
    splits = ['train']
    batchsize = 4
    lr = 10e-4
    num_epochs = 5

    ds = get_dataloaders(dataset,False)
    ds.set_format('torch',columns=['image','label'])
    train_dataloader = DataLoader(ds['train'])
    # train_dataloader, val_dataloader, test_dataloader = get_dataloaders(dataset, splits)
    # if train is not None:
    #     train_dataloader = DataLoader(train,4)
    # if val is not None:
    #     val_dataloader = DataLoader(val,4)
    # if test is not None:
    #     test_dataloader = DataLoader(test,4)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('mps') if torch.mps.is_available() else device

    print(device)

    model = FFDNet(False)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(),lr=lr)
    criterion = nn.MSELoss()

    count=0

    for epoch in range(num_epochs):
        for output in iter(train_dataloader):
            training_loss = 0
            count+=1
            print(f"iter: {count}")
            optim.zero_grad()
            image = output['image']
            # print(torchvision.io.read_image(image))
            
            noisy_image = noise_generator.add_noise(image)
            image = image.to(device)
            noise_image = noisy_image.to(device)

            noise_sigma = torch.FloatTensor(np.array([noise_level for idx in range(image.shape[0])]))
            noise_sigma = Variable(noise_sigma)
            noise_sigma = noise_sigma.to(device)

            denoised = model.forward(noisy_image,noise_sigma)
            denoised = denoised.to(device)

            #Make sure they are both the same type
            loss = criterion(denoised.uint8(),image.uint8())
            loss.backward()

            optim.step()
            break   

    # print(val['image'])
    
    # test_img = cv2.imread("FFDNet_pytorch/test_data/color.png")
    # print(test_img)
    # test_img = torch.tensor(test_img)
    # test_img = torch.permute(test_img,(2,0,1))
    # noisy_img = noise_generator.add_noise(test_img)
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(torch.permute(test_img,(1,2,0)))
    # plt.title("Input image")
    # plt.subplot(122)
    # plt.imshow(torch.permute(noisy_img,(1,2,0)))
    # plt.title("Noisy image")
    # plt.show()



