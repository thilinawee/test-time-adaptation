from typing import List

import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler

def generate_sample_indices(partial_classes: List[int], 
                            n_final_samples: int,
                            original_samples_per_class: int,
                            original_total_classes: int,
                            seed: int = 2024):
    

    for cls in partial_classes:
        assert cls < original_total_classes and cls >= 0, f'Class {cls} is not in the original dataset'


    partial_classes = sorted(partial_classes)
    n_partial_classes = len(partial_classes)

    generator = torch.Generator()
    generator.manual_seed(seed)

    weighted_random_sampler = WeightedRandomSampler([1.0] * n_partial_classes * original_samples_per_class,
                                                     n_final_samples,
                                                     replacement=True, 
                                                     generator=generator)
    
    # relative indices for the partial dataset created by WeightedRandomSampler
    sampler_indices = np.array(list(weighted_random_sampler))


    # map relative indices to original indices
    indice_map = {}

    for i, cls in enumerate(partial_classes):
        for j in range(original_samples_per_class):
            relative_index = i * original_samples_per_class + j
            original_index = cls * original_samples_per_class + j

            if relative_index not in indice_map:
                indice_map[relative_index] = original_index


    for i in range(len(sampler_indices)):
        sampler_indices[i] = indice_map[sampler_indices[i]]

    return sampler_indices