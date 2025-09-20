from typing import List, Optional, Union, Sequence

import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler

def generate_oversample_indices(partial_classes: List[int], 
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


def generate_sample_indices(partial_classes: List[int], 
                            original_samples_per_class: int,
                            original_total_classes: int,
                            seed: int = 2024):
    
    for cls in partial_classes:
        assert cls < original_total_classes and cls >= 0, f'Class {cls} is not in the original dataset'

    partial_classes = sorted(partial_classes)

    indices = []

    for i, cls in enumerate(partial_classes):
        for j in range(original_samples_per_class):
            original_index = cls * original_samples_per_class + j
            indices.append(original_index)

    return indices


def class_counts(data_loader):
    class_counts = {}
    for i, (data) in enumerate(data_loader):
        _, targets = data[0], data[1]
        for target in targets:
            if target.item() not in class_counts:
                class_counts[target.item()] = 0
            class_counts[target.item()] += 1
    sorted_class_counts = dict(sorted(class_counts.items(), key=lambda x: x[0]))
    return sorted_class_counts


def generate_oversample_indices_imbalanced_dataset(
    partial_classes: Optional[List[int]],
    n_final_samples: int,
    labels: Union[Sequence[int], np.ndarray, torch.Tensor],
    seed: int = 2024,
) -> np.ndarray:
    """
    Create an oversampled index list for datasets with variable images per class.
    - partial_classes: classes to include (e.g., a subset present in the target split).
        If None, all classes found in `labels` are used.
    - n_final_samples: total number of samples to draw (with replacement).
    - labels: 1D array-like of class ids for each dataset item (len == dataset size).
    - seed: RNG seed for reproducibility.

    Returns:
        np.ndarray of dataset indices (global indices), length == n_final_samples.
    """
    # Normalize labels -> numpy int array
    if isinstance(labels, torch.Tensor):
        y = labels.detach().cpu().numpy().astype(int)
    else:
        y = np.asarray(labels, dtype=int)

    # Determine which classes to include
    if partial_classes is None:
        keep_classes = np.unique(y).tolist()
    else:
        keep_classes = sorted(partial_classes)

    # Indices belonging to the kept classes
    mask = np.isin(y, keep_classes)
    selected_indices = np.nonzero(mask)[0]

    if selected_indices.size == 0:
        raise ValueError("No samples found for the requested partial_classes.")

    # Per-class counts among the selected subset
    # (build a small dict to avoid huge bincounts if labels are sparse)
    class_counts = {}
    for idx in selected_indices:
        c = int(y[idx])
        class_counts[c] = class_counts.get(c, 0) + 1

    # Inverse-frequency weight for each selected index
    weights = np.array([1.0 / class_counts[int(y[idx])] for idx in selected_indices], dtype=np.float64)

    # Torch sampler expects a 1D tensor of weights
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=int(n_final_samples),
        replacement=True,
        generator=generator
    )

    # sampler yields positions relative to `selected_indices`; map them to dataset indices
    rel = np.fromiter(iter(sampler), dtype=np.int64, count=n_final_samples)
    sampled_global_indices = selected_indices[rel]
    return sampled_global_indices