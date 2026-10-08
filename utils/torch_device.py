import torch

def torch_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.xpu.is_available():
        device = torch.device('xpu')
    else:
        device = torch.device('cpu')

    print(f'Using device: {device}')
    return device
