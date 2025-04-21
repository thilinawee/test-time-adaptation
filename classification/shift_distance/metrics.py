from typing import Dict

import torch
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torchvision.models.feature_extraction import create_feature_extractor


def get_silhouette_score(data, labels):

    unique_labels = np.unique(labels)

    print(f"Unique Labels: {unique_labels}")

    kmeans = KMeans(n_clusters=len(unique_labels), random_state=0)
    kmeans.fit(data)

    predicted_labels = kmeans.labels_

    score = silhouette_score(data, predicted_labels)

    return score


def get_mean_embeddings(features, labels):
    
    unique_labels = np.unique(labels)

    mean_embeddings = {}

    for label in unique_labels:
        class_data = features[labels == label]
        mean_embeddings[label] = np.mean(class_data, axis=0)

    return mean_embeddings


def calc_distance(a, b, distance_metric="euclidean", **kwargs):

    """
    a: (np.ndarray) Point A
    b: (np.ndarray) Point B
    """

    if distance_metric == "euclidean":
        distance = np.linalg.norm(a - b)

    elif distance_metric == "mahalanobis":
        raise NotImplementedError("Mahalanobis distance not implemented yet")

    return distance


def calc_mean_shift_distance(point_list_a, point_list_b, distance_metric="euclidean", **kwargs):

    cost_matrix = np.zeros((len(point_list_a), len(point_list_b)))

    for i, a in enumerate(point_list_a):
        for j, b in enumerate(point_list_b):
            distance = calc_distance(a, b, distance_metric=distance_metric, kwargs=kwargs)
            cost_matrix[i, j] = distance
    
    min_row_indices, min_col_indices = linear_sum_assignment(cost_matrix)
    mean_shift_distance = cost_matrix[min_row_indices, min_col_indices].sum().mean()

    return mean_shift_distance


def get_closest_cluster_centers(mean_embeddings, cluster_centers):

    cost_matrix = np.zeros((len(mean_embeddings), len(cluster_centers)))

    for i, a in enumerate(mean_embeddings.values()):
        for j, b in enumerate(cluster_centers):
            distance = calc_distance(a, b)
            cost_matrix[i, j] = distance
    
    min_row_indices, min_col_indices = linear_sum_assignment(cost_matrix)

    closest_cluster_centers = {}

    for i, j in zip(min_row_indices, min_col_indices):
        cls = list(mean_embeddings.keys())[i]
        closest_cluster_centers[cls] = cluster_centers[j]

    return closest_cluster_centers


def get_cluster_centers(X, n_clusters):
    
    cluster_centers = {}
    # Fit KMeans to find cluster centers
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)
    cluster_centers = kmeans.cluster_centers_

    return cluster_centers


def get_features(model, data_loader, nodes, debug=False, device="cuda"):

    feature_list = []
    label_list = []

    model.eval()
    feature_extractor = create_feature_extractor(model, nodes)

    for i, (data) in enumerate(data_loader):
        images, labels = data[0], data[1]
        features_dict = feature_extractor(images.to(device))
        features = list(features_dict.values())[0]
    
        if debug:
            print(f"Features: {features.shape}")
        
        reshaped_features = features.view(features.size(0), -1)
        feature_list.append(reshaped_features)
        label_list.extend(labels)

    feature_list = torch.cat(feature_list, dim=0).detach().cpu().numpy()

    label_list = np.array([x.cpu() for x in label_list])
    
    return feature_list, label_list



