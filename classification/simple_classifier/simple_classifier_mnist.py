import os
from copy import deepcopy
import numpy as np
import random
import warnings

import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import calc_logit_norm, validate, extract_features, \
    visualize_features, get_reduced_features, add_gaussian_noise, \
    entropy_loss, collect_params, configure_model, adapt_model, \
    get_animated_features, get_partial_dataloader, animate_3d_features, logit_adjusted_adaptation,\
    class_counts

from models import CNN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Replace "0" with the desired GPU device index


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning)

    random_seed = 2025

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


    # Parameters
    LOG_FREQUENCY =  1000
    DO_ADJUST_LOGITS = False
    TAU = 1.0
    PARTIAL_CLASSES = [i for i in range(3)]
    
    test_class_distribution = []
    for i in range(10):
        if i in PARTIAL_CLASSES:
            test_class_distribution.append(1.0/len(PARTIAL_CLASSES))
        else:
            test_class_distribution.append(0.0)


    transform = transforms.Compose([
        transforms.ToTensor(),  # converts to tensor and scales image pixel values to [0, 1]
        transforms.Normalize((0.1307,), (0.3081,))  # normalize using MNIST's mean and std
    ])

    train_dataset = datasets.MNIST(root='/home/thilina/SSD2/thilina/datasets/mnist', train=True, download=False, transform=transform)
    test_dataset  = datasets.MNIST(root='/home/thilina/SSD2/thilina/datasets/mnist', train=False, download=False, transform=transform)

    model = torch.load('/home/thilina/SSD2/thilina/test-time-adaptation-further_experiments/classification/simple_classifier/mnist_model.pth')
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)


    # Clean data

    # features, feature_labels = extract_features(model, test_loader, 'fc1')
    # # reducer = visualize_features(features, feature_labels, "Clean Features")
    # clean_reduced_features, initial_labels, reducer = get_reduced_features(features, feature_labels, dimensions=3)


    # Noisy data
    noisy_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: add_gaussian_noise(x, severity=5)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    noisy_test_dataset = datasets.MNIST(root='/home/thilina/SSD2/thilina/datasets/mnist', train=False, download=False, transform=noisy_transform)
    
    # when doing TTA, dataloader should be shuffled
    noisy_train_loader = DataLoader(noisy_test_dataset, batch_size=64, shuffle=True)

    noisy_test_loader = DataLoader(noisy_test_dataset, batch_size=64, shuffle=False)
    # zero_shot_accuracy, _ = validate(model, noisy_test_loader)
    # print(f"Zero shot Test Accuracy: {zero_shot_accuracy: .2%}")


    # noisy_features, _ = extract_features(model, noisy_test_loader, 'fc1')
    # noisy_reduced_features, _, _ = get_reduced_features(noisy_features, initial_labels, reducer)

    ########################################################################
    # Adapt the model to balanced dataset
    # test_acc_list, reduced_feature_list, _ = adapt_model(model, noisy_train_loader, noisy_test_loader, reducer, log_frequency=500, feature_layer='fc1')
    # print(f"Class Counts: {class_counts(noisy_train_loader)}")
    # test_acc_list, reduced_feature_list, _ = logit_adjusted_adaptation(model, noisy_train_loader, noisy_test_loader, do_adjust_logits=False)

    # get_animated_features(reduced_feature_list, initial_labels, "balanced_adaptation_2d.mp4")
    # animate_3d_features(reduced_feature_list, initial_labels, "balanced_adaptation_umap_3d.mp4")

    ########################################################################
    # Adapt the model to Imbalanced dataset
    imbalanced_test_loader = get_partial_dataloader(noisy_test_dataset, PARTIAL_CLASSES, final_samples=10000)
    print(f"Class Counts: {class_counts(imbalanced_test_loader)}")

    model = torch.load('/home/thilina/SSD2/thilina/test-time-adaptation-further_experiments/classification/simple_classifier/mnist_model.pth')

    test_acc_list, reduced_feature_list, _ = logit_adjusted_adaptation(model, imbalanced_test_loader, noisy_test_loader, class_distribution=test_class_distribution,
                                                                       do_adjust_logits=DO_ADJUST_LOGITS, log_frequency=LOG_FREQUENCY, tau=TAU, epochs=40)
    # test_acc_list, reduced_feature_list, _ = adapt_model(model, imbalanced_test_loader, noisy_test_loader, reducer, log_frequency=500, feature_layer='fc1')
    # reduced_feature_list = [noisy_reduced_features] + reduced_feature_list
    # get_animated_features(reduced_feature_list, initial_labels, "partial_adaptation_2d.mp4")
    # animate_3d_features(reduced_feature_list, initial_labels, "partial_adaptation_umap_3d.mp4")





    