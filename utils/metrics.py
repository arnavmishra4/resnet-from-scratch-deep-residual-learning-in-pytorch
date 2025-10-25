import torch

def accuracy(preds, targets):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == targets).sum().item()
    return correct / len(targets)
