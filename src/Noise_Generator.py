import numpy as np
import torch
from my_utils import normalize_image


class Noise_Generator():
    def __init__(self,noise_type='gaussian',noise_level='30') -> None:
        self.noise_type = noise_type
        self.noise_level = noise_level

    def set_noise_type(self,noise_type):
        self.noise_type = noise_type
        #Add some checks to what is a valid type, Gaussian, Poisson
        print(f"Noise type has been set to {self.noise_type}")
    
    def set_noise_level(self, noise_level):
        self.noise_level = noise_level
        print(f"Noise level has been set to {self.noise_level}")

    def add_noise(self,img):
        norm_img, shift, scale = normalize_image(img)
        if self.noise_type == 'gaussian':
            noisy_img = (norm_img + torch.normal(0,self.noise_level,norm_img.shape))
        if self.noise_type == 'poisson':
            #TODO
            pass
        noisy_img[noisy_img<0]=0
        noisy_img[noisy_img>255]=255
        noisy_img = noisy_img.to(torch.uint8)
        return noisy_img