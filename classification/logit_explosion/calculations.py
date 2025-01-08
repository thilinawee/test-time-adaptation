from typing import List
import torch
import math

def calc_logit_norm(logits: torch.Tensor):
    """
    Calculate the norm of the selected logits for all the samples in the batch and return the average
    """
    per_class_norm = torch.mean(torch.norm(logits, dim=1)/math.sqrt(logits.shape[1]))
    norm = torch.mean(torch.norm(logits, dim=1))

    return per_class_norm.item(), norm.item()