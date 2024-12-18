import torch
import numpy as np
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datasets import load_dataset

from Noise_Generator import Noise_Generator
from my_utils import PSNR, SSIM, normalize_image, restore_image
from model import FFDNet, ResidualFFDNet, AttentionFFDNet, Res2_FFDNet, Res3_FFDNet, ResidualLargeFFDNet

def get_dataloaders(dataset,splits):
    train_dataloader, val_dataloader, test_dataloader = None, None, None
    ds = load_dataset("GATE-engine/mini_imagenet")
    return ds

if __name__=="__main__":
    torch.manual_seed(16)       #Set seed for reproducable training

    dirname = os.path.dirname(__file__)
    os.chdir(dirname)
    os.chdir("..")
    cwd = os.getcwd()

    model_save_sie = os.path.join(cwd,"model_saves")
    noise_level = [0,50] #Not customizable yet 
    dataset = "GATE-engine/mini_imagenet"
    ####################################Controllable parameters #############################################################
    noise_type = 'gaussian' #Options are gaussian, or poisson
    noise_distribution = "80-20" # Options are "80-20", "20-80", uniform
    
    model_type = 'reslarge'           #options are regular, residual, attention
    load_model = True               #Set to false if training from scratch

    batchsize = 16
    lr = 1e-4
    num_epochs = 5
    ##########################################################################################################################
    model_save_path = os.path.join("model_saves","reslarge")
    splits = ['train','validation','test']

    noise_generator = Noise_Generator(noise_type,noise_level,noise_distribution=noise_distribution)
    ds = get_dataloaders(dataset,False)
    # ds = load_dataset('parquet',data_dir="mini_imagenet/data")
    ds.set_format('torch',columns=['image','label'])
    train_dataloader = DataLoader(ds['train'],batchsize,shuffle=True)
    val_dataloader = DataLoader(ds['validation'],batchsize,shuffle=True)
    test_dataloader = DataLoader(ds['test'],batchsize,shuffle=False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('mps') if torch.mps.is_available() else device

    print(device)

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

    print(f"Model type selected was: {model}")
    model.to(device)
    if load_model:
        model.load_state_dict(torch.load(model_save_path,weights_only=True))
    optim = torch.optim.AdamW(model.parameters(),lr=lr)
    criterion = nn.MSELoss()

    count=0
    train_loss = []
    val_loss = []
    print(len(train_dataloader))

    #Training and validation
    for epoch in range(num_epochs):
        training_loss = 0
        for output in iter(train_dataloader):
            
            count+=1
            if count%100==0:
                print(f"iter: {count}")
            optim.zero_grad()
            image = output['image']
            # print(torchvision.io.read_image(image))
            
            
            noisy_image,noise_level = noise_generator.add_noise(image)
            image = image.to(device)
            noise_image = noisy_image.to(device)

            noise_sigma = torch.FloatTensor(np.array([noise_level for idx in range(image.shape[0])]))
            noise_sigma = Variable(noise_sigma)
            noise_sigma = noise_sigma.to(device)

            denoised = model.forward(noisy_image,noise_sigma)
            denoised = denoised.to(device)

            #Make sure they are both the same type
            denoised = denoised.to(torch.float32)
            image = image.to(torch.float32)
            loss = criterion(denoised,image)
            loss.backward()

            optim.step()
   
            training_loss+=loss.item()
            train_loss.append(loss.item())

            # print(image.shape)
            # print(noisy_image.shape)
            # print(denoised.shape)

            if count%500==0:
                noisy_image = torch.permute(noisy_image[0]/255.0,(1,2,0)).cpu().detach().numpy()
                denoised_image = torch.permute(denoised[0]/255.0,(1,2,0)).cpu().detach().numpy()
                image = torch.permute(image[0]/255.0,(1,2,0)).cpu().detach().numpy()
                plt.figure()
                # fig.axes.get_yaxis().set_visible(False)
                plt.subplot(131)
                plt.imshow(noisy_image)
                plt.text(112,-25,"Noisy Image",horizontalalignment='center')
                plt.text(112,-5,f"PSNR: {PSNR(image,noisy_image):0.2f}",horizontalalignment='center')
                plt.axis('off')
                plt.subplot(132)
                plt.imshow(denoised_image)
                plt.text(112,-25,"Denoised Image",horizontalalignment='center')
                plt.text(112,-5,f"PSNR: {PSNR(image,denoised_image):0.2f}",horizontalalignment='center')
                plt.axis('off')
                plt.subplot(133)
                plt.imshow(image)
                plt.title("True Image")
                filename = f"results/{epoch}_{count}"
                plt.axis('off')
                plt.savefig(filename)
                plt.close()
                if count%2000:
                    model_path=f"model_saves/{epoch}_{count}"
                    torch.save(model.state_dict(),model_path)
        with torch.no_grad():
            validation_loss = 0
            for output in iter(val_dataloader):
                
                count+=1
                if count%100==0:
                    print(f"iter: {count}")

                image = output['image']
                # print(torchvision.io.read_image(image))
                
                
                noisy_image,noise_level = noise_generator.add_noise(image)
                image = image.to(device)
                noise_image = noisy_image.to(device)

                noise_sigma = torch.FloatTensor(np.array([noise_level for idx in range(image.shape[0])]))
                noise_sigma = Variable(noise_sigma)
                noise_sigma = noise_sigma.to(device)

                denoised = model.forward(noisy_image,noise_sigma)
                denoised = denoised.to(device)

                #Make sure they are both the same type
                denoised = denoised.to(torch.float32)
                image = image.to(torch.float32)
                loss = criterion(denoised,image)
    
                validation_loss+=loss.item()
                val_loss.append(loss.item())
        print(f"Training loss in epoch {epoch}: {training_loss/(len(train_dataloader))}")
        print(f"Validation loss in epoch {epoch}: {validation_loss/len(val_dataloader)}")
        model_path=f"model_saves/{epoch}_{count}"
        torch.save(model.state_dict(),model_path)



    plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.show()
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