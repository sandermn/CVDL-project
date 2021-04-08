import torch

def load(model, PATH):
    model.load_state_dict(torch.load(PATH))
    return model