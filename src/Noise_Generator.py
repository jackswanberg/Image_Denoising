import numpy as np
import matplotlib.pyplot as plt
import torch
from my_utils import normalize_image, restore_image
import skimage
from skimage.util import random_noise
from torchvision.io import read_image
from torchvision.utils import save_image
import glob
import os

class Noise_Generator():
    def __init__(self,noise_type='gaussian',noise_level=[0,50],num_bins=5,noise_distribution='uniform') -> None:
        self.noise_type = noise_type
        self.noise_level = noise_level
        # self.noise_range = [0,50]
        self.noise_bins = torch.tensor([0,10,20,30,40])
        if noise_distribution=="uniform":
            self.noise_distribution = torch.arange(1/num_bins,1+1/num_bins,1/num_bins)
        elif noise_distribution=="80-20":
            self.noise_distribution = torch.tensor([0.7,0.85,0.93,0.98,1])              # Distribution from One Size Fits All: https://arxiv.org/pdf/2005.09627
        elif noise_distribution=="20-80":
            self.noise_distribution = torch.tensor([0.02,0.07,0.15,0.3,1])    
                    # Flipped Distribution from One Size Fits All: https://arxiv.org/pdf/2005.09627

    def set_noise_type(self,noise_type):
        self.noise_type = noise_type
        #Add some checks to what is a valid type, Gaussian, Poisson
        print(f"Noise type has been set to {self.noise_type}")
    
    def set_noise_level(self, noise_level):
        self.noise_level = noise_level
        print(f"Noise level has been set to {self.noise_level}")

    def add_noise(self,img):
        img = img.float()
        norm_img, shift, scale = normalize_image(img)
        noisy_img = torch.zeros_like(img)
        noise_level = 0
        if self.noise_type == 'gaussian':
            roll = torch.rand(1)
            for i, prob in enumerate(self.noise_distribution):
                if roll<prob:
                    noise_level = torch.rand(1)*10+self.noise_bins[i]
                    noisy_img = norm_img + torch.normal(0,float(noise_level),norm_img.shape)
                    break
        if self.noise_type == 'poisson':
            noisy_img = random_noise(img/255,self.noise_type)
            noisy_img = noisy_img*255
        noisy_img = restore_image(noisy_img,shift,scale)
        noisy_img[noisy_img<0]=0
        noisy_img[noisy_img>255]=255
        noisy_img = noisy_img.to(torch.uint8)
        return noisy_img,noise_level
    
if __name__=="__main__":
    noise_generator = Noise_Generator(noise_type='gaussian')
    files = glob.glob("test_data/CBSD68/original_png/*")
    for file in files:
        image_file = file[-8:]
        path = "test_data/CBSD68/noisy75/"
        image = read_image(file)
        noisy_image,_ = noise_generator.add_noise(image)
        plt.figure()
        plt.imshow(torch.permute(noisy_image/255.0,(1,2,0)).cpu().detach().numpy())
        plt.show()

        # save_image(noisy_image/255.,os.path.join("test_data","CBSD68","noisy75",image_file))


   
    # bins = np.ones(5)
    # for i in range(1000):
    #     _,j = noise_generator.add_noise(img)
    #     bins[j]+=1
    # print(bins)

