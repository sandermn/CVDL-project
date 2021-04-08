import torch

def save(model, PATH):
    torch.save(model.state_dict(), PATH)