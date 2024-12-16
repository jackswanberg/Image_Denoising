import torch
import time
from prettytable import PrettyTable
from torch.utils.flop_counter import FlopCounterMode
from model import FFDNet, ResidualFFDNet, AttentionFFDNet,ResidualLargeFFDNet

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def get_flops(model, inp, with_backward=False):
    
    istrain = model.training
    model.eval()
    
    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)

    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
    with flop_counter:
        if with_backward:
            model(inp).sum().backward()
        else:
            model(inp,torch.randn(1))
    total_flops =  flop_counter.get_total_flops()
    if istrain:
        model.train()
    return total_flops

def get_runtime(model, inp, iterations=1000, with_backward=False):
    model.eval()
    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)
    noise_sigma = torch.randn(1)
    start_time = time.time()
    for i in range(iterations):
        model(inp,noise_sigma)
    end_time = time.time()
    return end_time-start_time

model = AttentionFFDNet()          # Just change to whatever model you want to get metrics from
model.cuda()
count_parameters(model)
print(f"gpu used: {torch.cuda.max_memory_allocated(device=None)/(1024**2)} MB")
print(f"Flops: {get_flops(model,(1,3,224,224))/(10**9)} GFLOPS")               #The second input "(1,3,224,224)" is the dimension of the model input
print(f"Average runtime: {get_runtime(model,(1,3,224,224))/1000} ms")           #The second input "(1,3,224,224)" is the dimension of the model input

