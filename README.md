Project for implementing FFDNet for image denoising

Training is done by running "main.py", parameters are controlled inside the script (Starting at line 33). The script needs access to the internet in order to get data (https://huggingface.co/datasets/GATE-engine/mini_imagenet).

Parameters that can be selected:
    noise_type: can be gaussian or poisson
    noise_distribution: 80-20, 20-80 or uniform.                  80-20 is the recommended 80-20 split from One Size Fits All: https://arxiv.org/pdf/2005.09627

    model_type: options are 'regular', 'residual', 'reslarge', 'attention', 'res2', and 'res3'. Reslarge utilizes resnetblocks with two convolutions per block, whereas residual has 1 convolution per residual block
    load_model: True/False if you would like to load a pretrained model or not. It is expected the weights are in a directory at the same level as "src", with weight files with the same name as the model_type (e.g. a 'residual' model will try to load weights located at 'model_saves/residual')

    Hyperparameters
    batchsize: for all models except attention batchsize of 16 can fit in 6GB of RAM
    lr
    epochs


To run a demonstration of the models, demo.py can be run, again the parameters can be accessed inside the script, starting at line 45
To run this file, you need model weights inside the model_saves directory, as well as test data at Image_Denoising/test_data/CBSD68/{test_dir}, current valid test directories in the code are ['original_png','noisy15','noisy25','noisy50','noisy75','poisson']

    load_model = True, will load model weights that have the same name as the model_type
    model_type, same as above, options are regular, residual, attention, reslarge (resnet blocks), res2 and res3
    test: valid inputs are 0-5 selecting what test images are input from ['original_png','noisy15','noisy25','noisy50','noisy75','poisson']
    noise_level: Integer input                      #Parameter to set noise_map value for feeding into model, typically matched with noise level
    visualization: True/False if you would like to visualize every input or not 

