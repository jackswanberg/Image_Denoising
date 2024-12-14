import torch
from prettytable import PrettyTable
from model import FFDNet, ResidualFFDNet, AttentionFFDNet

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
    
model = AttentionFFDNet()
count_parameters(model)