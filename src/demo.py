import torch
import torchvision
import argparse
import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt
import lpips
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.io import read_image
from datasets import load_dataset
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode


from Noise_Generator import Noise_Generator
from my_utils import PSNR, SSIM, normalize_image, restore_image
from model import FFDNet, ResidualFFDNet, AttentionFFDNet, Res2_FFDNet, Res3_FFDNet, ResidualLargeFFDNet


if __name__=="__main__":
    # noise_type = 'gaussian' #Options are gaussian, or poisson
    # noise_level = [0,50]
    # noise_distribution = "80-20" # Distribution from One Size Fits All: https://arxiv.org/pdf/2005.09627
    # noise_generator = Noise_Generator(noise_type,noise_level,noise_distribution=noise_distribution)
    

    dirname = os.path.dirname(__file__)
    os.chdir(dirname)
    os.chdir("..")
    cwd = os.getcwd()
    model_save_path = os.path.join(cwd,"model_saves")

    
    
    tests = ['original_png','noisy15','noisy25','noisy50','noisy75','poisson']
    ######################### Can change parameters here ######################################
    load_model = True
    model_type = 'attention'           #options are regular, residual, attention, reslarge (resnet blocks), res2 and res3
    test = 2                          #Options listed above, valid inputs are 0-5
    noise_level=25                      #Parameter to set noise_map value for feeding into model, typically matched with noise level
    visualization = False                #Set to false if just want to run through test data and get metrics
    ####################################################################

    # model_save_path = os.path.join("model_saves","residuals_12epochs")
    model_save_path = os.path.join("model_saves",model_type)
    test_dir = os.path.join(cwd,"test_data","CBSD68")
    image_dir = os.path.join(cwd,"test_data","CBSD68","original_png")
    noisy_images = glob.glob(os.path.join(test_dir,tests[test],"*"))
    clean_images = glob.glob(os.path.join(image_dir,"*"))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('mps') if torch.mps.is_available() else device
    print(device)

    loss_fn_alex = lpips.LPIPS('weights=AlexNet_Weights.DEFAULT')

    if model_type=='residual':
        model = ResidualFFDNet()
    elif model_type=='attention':
        model = AttentionFFDNet()
    elif model_type=='regular':
        model = FFDNet()
    elif model_type=='res2':
        model = Res2_FFDNet()
    elif model_type=='res3':
        model = Res3_FFDNet()
    elif model_type=='reslarge':
        model = ResidualLargeFFDNet()
    else:
        print("Unknown model type input, defaulting to regular FFDNet")
        model = FFDNet()

    # print(f"Model type selected was: {model}")
    model.to(device)
    print(os.listdir())
    if load_model:
        model.load_state_dict(torch.load(model_save_path,weights_only=True))
    
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    with torch.no_grad():
        for image, noisy_image in zip(clean_images,noisy_images):
            img = read_image(image)
            img = img[:,:img.shape[1]//2*2,:img.shape[2]//2*2]
            img = img[None,:,:,:]

            noisy_img = read_image(noisy_image)
            noisy_img = noisy_img[:,:noisy_img.shape[1]//2*2,:noisy_img.shape[2]//2*2]
            noisy_img = noisy_img[None,:,:,:]
            

            # noise_sigma = torch.FloatTensor(np.array([torch.tensor([50]) for idx in range(img.shape[0])]))
            noise_sigma = torch.FloatTensor([noise_level])
            noise_sigma = Variable(noise_sigma)
            noise_sigma = noise_sigma.to(device)
            denoised_img = model.forward(noisy_img,noise_sigma)
            denoised_img[denoised_img>255.] = 255.
            denoised_img[denoised_img<0.0] = 0.0

            if visualization:
                plt.figure()
                plt.subplot(131)
                plt.title("Original Image")
                plt.imshow(torch.permute(img[0]/255.0,(1,2,0)).cpu().detach().numpy())
                plt.axis("off")
                plt.subplot(132)
                plt.title("Noisy Image")
                plt.imshow(torch.permute(noisy_img[0]/255.0,(1,2,0)).cpu().detach().numpy())
                plt.axis("off")
                plt.subplot(133)
                plt.title("Denoised Image")
                plt.imshow(torch.permute(denoised_img[0]/255.0,(1,2,0)).cpu().detach().numpy())
                plt.axis("off")
                plt.show()

            total_lpips += loss_fn_alex(img, denoised_img)
            noisy_img = torch.permute(noisy_img[0],(1,2,0)).cpu().detach().numpy()
            denoised_img = torch.permute(denoised_img[0],(1,2,0)).cpu().detach().numpy()
            img = torch.permute(img[0],(1,2,0)).cpu().detach().numpy()
            img[img<0]=0
            img[img>255]=255
            img[denoised_img<0]=0
            img[denoised_img>255]=255

            img = img.astype(np.uint8)
            denoised_img = denoised_img.astype(np.uint8)

            total_psnr += PSNR(img,denoised_img)
            total_ssim += SSIM(img,denoised_img)
        
n = len(noisy_images)
print(f"Results for {model_type}, {tests[test]}")
print(f"Average PSNR: {total_psnr/n}")
print(f"Total SSIM: {total_ssim/n}")
print(f"Total LPIPS: {total_lpips/n}")