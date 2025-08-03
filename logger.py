import torch

def log_weights(model):
    weights = {}
    # Calculate the mean of the square root of the squares of the weights
    for name, param in model.named_parameters():
        if 'weight' in name or 'K_hit' in name or 'K_miss' in name:
            weights["weights/" + name + "_mean"] = torch.sqrt(param.data ** 2).mean().item()

    return weights
