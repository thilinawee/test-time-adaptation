from typing import List
from copy import deepcopy
import os

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import matplotlib.animation as animation

from torch.utils.data import Subset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap, BoundaryNorm


def calc_logit_norm(logits: torch.Tensor):
    """
    Calculate the norm of the selected logits for all the samples in the batch and return the average
    """
    per_class_norm = torch.mean(torch.norm(logits, dim=1)/math.sqrt(logits.shape[1]))
    norm = torch.mean(torch.norm(logits, dim=1))

    return per_class_norm.item(), norm.item()

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


def validate(model, dataloader, criterion=nn.CrossEntropyLoss()):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    accuracy = correct / total
    validation_loss = total_loss / len(dataloader)
    return accuracy, validation_loss

def logit_adjusted_validation(model, dataloader, class_distribution, criterion=nn.CrossEntropyLoss(), tau=1.0):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)

            # adjust logits
            adjusted_logits = outputs - tau * torch.log(class_distribution + 1e-12)

            # Get predictions from the maximum value
            _, predicted = torch.max(adjusted_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    accuracy = correct / total
    validation_loss = total_loss / len(dataloader)
    return accuracy, validation_loss



def visualize_features(features, labels, title="Umap Visualization of CNN Feature Outputs", reducer=None, plot=True):
    
    if reducer is None:
        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(features)
    else:
        embedding = reducer.transform(features)

    if plot:
        # Plot the UMAP results
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.title(title)
        plt.colorbar(scatter, ticks=range(10), label='Digit Label')
        plt.show()

    return reducer

def get_reduced_features(features, lables, reducer=None, dimensions=2, n_neighbors=15):
    if reducer is None:
        reducer = umap.UMAP(n_components=dimensions, n_neighbors=n_neighbors, random_state=42)
        embedding = reducer.fit_transform(features)
    else:
        embedding = reducer.transform(features)
    return embedding, lables, reducer

def extract_features(model, data_loader, layer_name='fc2'):
    # Create a list to store the features and labels
    features = []
    true_labels_list = []
    predicted_labels_list = []

    # Define a forward hook to capture the output of the specified layer
    def hook(module, input, output):
        features.append(output.detach().cpu())

    # Retrieve the layer by its name and register the hook
    target_layer = getattr(model, layer_name)
    hook_handle = target_layer.register_forward_hook(hook)

    # Set the model to evaluation mode
    model.eval()

    # Iterate over the data loader to extract features
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to('cuda')
            labels = labels.to('cuda')
            # Forward pass (the hook will capture the activations of the target layer)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            # Save the labels for each batch
            true_labels_list.extend(labels.cpu().numpy())
            predicted_labels_list.extend(predictions.cpu().numpy())


    # Remove the hook
    hook_handle.remove()

    # Concatenate all feature batches
    features_array = torch.cat(features, dim=0).numpy()
    return features_array, predicted_labels_list

def add_gaussian_noise(X, severity=5, random_seed=2025):
    scale = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    # Add Gaussian noise to the data
    generator = torch.Generator().manual_seed(random_seed)
    noise = torch.normal(size=X.shape, std=scale, mean=0.0, generator=generator)
    noisy_X = X + noise
    return torch.clamp(noisy_X, 0., 1.)


def entropy_loss(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def collect_params(model, freeze_layers=[]):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            if any([f'{layer}' in nm for layer in freeze_layers]):
                continue
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def configure_model(model):
    """Configure model for use with eata."""
    # train mode, because eata optimizes the model to minimize entropy
    # self.model.train()
    model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
    # disable grad, to (re-)enable only what eata updates
    model.requires_grad_(False)
    # configure norm for eata updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        elif isinstance(m, nn.BatchNorm1d):
            m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
            m.requires_grad_(True)
            
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)


def adapt_model(model, adaptation_loader, test_loader, reducer, log_frequency=100, feature_layer='fc1', epochs=10, do_adjust_logits=False,
                tau=1.0, class_distribution=[1.0 for i in range(10)]):

    test_acc_list = []
    logit_norm_list = []
    reduced_feature_list = []
    predicted_labels_list = []


    steps = 0
    total_steps = len(adaptation_loader) * epochs


    #####  Zero shot inference #####

    deepcopy_model = deepcopy(model)
    # test accuracy
    test_accuracy,_ = validate(deepcopy_model, test_loader)
    test_acc_list.append((test_accuracy, steps))
    print(f"Adaptation step {steps}/{total_steps}")
    print(f"Test Accuracy: {test_accuracy:.2f}")

    features, labels = extract_features(deepcopy_model, test_loader, feature_layer)

    # get reduced features
    if reducer is not None:
        features, _, _ = get_reduced_features(features, labels, reducer)
    reduced_feature_list.append(features)
    predicted_labels_list.append(labels)


    #### Model Configuration ####
    configure_model(model)
    bn_params, bn_names = collect_params(model, freeze_layers=["bn100"])
    print(bn_names)
    tent_optimizer = optim.Adam(bn_params, lr=1e-3)




    for epoch in range(epochs):

        for x_data, y_data in adaptation_loader:
            tent_optimizer.zero_grad()
            x_data, y_data = x_data.to('cuda'), y_data.to('cuda')
            logits = model(x_data)

            # adjust logits
            if do_adjust_logits:
                adjusted_logits = logits + tau * torch.log(class_distribution + 1e-12)
            else:
                adjusted_logits = logits

            loss = entropy_loss(adjusted_logits).mean(0)
            loss.backward()
            tent_optimizer.step()


            if (steps+1) % log_frequency == 0:
                deepcopy_model = deepcopy(model)
                # test accuracy
                test_accuracy,_ = validate(deepcopy_model, test_loader)
                test_acc_list.append((test_accuracy, steps))
                print(f"Adaptation step {steps}/{total_steps}, Entropy Loss: {loss.item():.4f}")
                print(f"Test Accuracy: {test_accuracy:.2f}")

                features, labels = extract_features(deepcopy_model, test_loader, feature_layer)

                # get reduced features
                if reducer is not None:
                    features, _, _ = get_reduced_features(features, labels, reducer)
                reduced_feature_list.append(features)
                predicted_labels_list.append(labels)

                
                logit_norm, _ = calc_logit_norm(torch.Tensor(logits[:,[0]]))
                logit_norm_list.append((logit_norm, steps))

            steps += 1

    return test_acc_list, reduced_feature_list, predicted_labels_list


def get_animated_features(features, labels, save_path="umap_feature_drift.mp4"):
    num_frames = len(features)
    
    # Compute global x and y limits across all frames
    # Concatenate all frames (each is an array of shape [num_points, 2])
    all_data = np.concatenate(features, axis=0)
    x_min, x_max = np.min(all_data[:, 0]), np.max(all_data[:, 0])
    y_min, y_max = np.min(all_data[:, 1]), np.max(all_data[:, 1])
    
    # Add margins (10% of the range)
    x_margin = 0.1 * (x_max - x_min)
    y_margin = 0.1 * (y_max - y_min)
    
    # Set up the figure and initial scatter plot with colors defined by labels.
    fig, ax = plt.subplots()
    scat = ax.scatter(
        features[0][:, 0],
        features[0][:, 1],
        c=labels,          # Points colored by their class labels.
        cmap='tab10',      # You can choose any colormap.
        s=20
    )
    
    # Set axis limits based on the computed values
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_title("UMAP Feature Drift with Class Colors")
    
    unique_labels = len(np.unique(labels))
    plt.colorbar(scat, ticks=range(unique_labels), label='Digit Label')

    # Update function for animation: update point positions per frame.
    def update(frame):
        data = features[frame]
        scat.set_offsets(data)
        ax.set_title(f"Frame {frame+1}")
        return scat,

    # Create the animation using FuncAnimation.
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=2000, blit=True)
    
    # Save the animation as a video file (requires ffmpeg installed)
    Writer = animation.FFMpegWriter
    writer = Writer(fps=5, bitrate=1800)
    ani.save(save_path, writer=writer)

    plt.show()


def get_partial_dataloader(dataset, partial_classes, final_samples=10000):

    # Get indices of the dataset that correspond to the desired classes
    subset_indices = [i for i, (_, label) in enumerate(dataset) if label in partial_classes]

    # Create a subset dataset using the filtered indices
    subset_dataset = Subset(dataset, subset_indices)

    # Calculate class frequencies in the subset dataset
    labels = [dataset[i][1] for i in subset_indices]

    sample_weights = [1.0 for label in labels]

    # Create a WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=final_samples, replacement=True)

    # Create DataLoader with the sampler for oversampling
    partial_dataloader = DataLoader(subset_dataset, batch_size=64, sampler=sampler)

    return partial_dataloader


def animate_3d_features(features, label_frames, save_path='3d_features_animation.mp4'):
    """
    Animate 3D feature points with corresponding label frames using a custom colormap.

    Parameters:
    - features: list of NumPy arrays of shape [num_points, 3] representing feature coordinates per frame.
    - label_frames: list of NumPy arrays of shape [num_points] containing labels for each frame.
    - save_path: path to save the animation video (e.g., '3d_features_animation.mp4').

    Raises:
    - ValueError: if the number of feature frames and label frames are not equal.
    """
    num_frames = len(features)
    if num_frames != len(label_frames):
        raise ValueError("The number of feature frames and label frames must be the same.")

    # Define a discrete color mapping (example for up to 10 classes).
    discrete_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    num_labels = len(discrete_colors)

    # Create a discrete colormap and norm so that each integer label maps to its specific color.
    cmap = ListedColormap(discrete_colors)
    norm = BoundaryNorm(np.arange(-0.5, num_labels + 0.5, 1), num_labels)

    # Compute global axis limits from all feature points.
    all_data = np.concatenate(features, axis=0)
    x_min, x_max = np.min(all_data[:, 0]), np.max(all_data[:, 0])
    y_min, y_max = np.min(all_data[:, 1]), np.max(all_data[:, 1])
    z_min, z_max = np.min(all_data[:, 2]), np.max(all_data[:, 2])
    x_margin = 0.1 * (x_max - x_min)
    y_margin = 0.1 * (y_max - y_min)
    z_margin = 0.1 * (z_max - z_min)

    # Set up the figure and initial 3D scatter plot.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter(
        features[0][:, 0],
        features[0][:, 1],
        features[0][:, 2],
        c=label_frames[0],
        cmap=cmap,
        norm=norm,
        s=20
    )

    # Set axis limits.
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_zlim(z_min - z_margin, z_max + z_margin)
    ax.set_title("3D Feature Space")

    # Create a colorbar that always shows all label values.
    cbar = plt.colorbar(scat, ticks=np.arange(num_labels))
    cbar.set_label('Label')
    cbar.ax.set_yticklabels([str(i) for i in range(num_labels)])

    # Update function for the animation.
    def update(frame):
        data = features[frame]
        current_labels = label_frames[frame]
        # Update the positions of the points.
        scat._offsets3d = (data[:, 0], data[:, 1], data[:, 2])
        # Update the colors (labels) of the points.
        scat.set_array(current_labels)
        ax.set_title(f"Frame {frame + 1}")
        return scat,

    # Create and save the animation.
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=2000, blit=False)
    Writer = animation.FFMpegWriter
    writer = Writer(fps=5, bitrate=1800)
    ani.save(save_path, writer=writer)
    plt.show()
    

def filter_features(features, labels, selected_labels):
    """
    Filters features and labels based on a list of selected labels.

    Args:
        features (torch.Tensor or np.ndarray): Array of feature vectors with shape (N, feature_dim).
        labels (torch.Tensor or np.ndarray): Array of labels with shape (N,).
        selected_labels (list or set): The labels to keep.

    Returns:
        filtered_features, filtered_labels: The subset of features and labels corresponding to selected_labels.
    """
    if isinstance(features, torch.Tensor) and isinstance(labels, torch.Tensor):
        # Use torch.isin if available (requires PyTorch 1.10+)
        if hasattr(torch, 'isin'):
            mask = torch.isin(labels, torch.tensor(list(selected_labels), device=labels.device))
        else:
            # Fallback if torch.isin is not available
            mask = torch.zeros_like(labels, dtype=torch.bool)
            for lab in selected_labels:
                mask |= (labels == lab)
        return features[mask], labels[mask]
    
    elif isinstance(features, np.ndarray) and isinstance(labels, np.ndarray):
        mask = np.isin(labels, list(selected_labels))
        return features[mask], labels[mask]
    
    else:
        raise ValueError("features and labels must be of the same type (torch.Tensor or np.ndarray).")
    
def filter_labels(predicted_labels, true_labels, selected_labels):
    """
    Filters features and labels based on a list of selected labels.

    Args:
        features (torch.Tensor or np.ndarray): Array of feature vectors with shape (N, feature_dim).
        labels (torch.Tensor or np.ndarray): Array of labels with shape (N,).
        selected_labels (list or set): The labels to keep.

    Returns:
        filtered_features, filtered_labels: The subset of features and labels corresponding to selected_labels.
    """
    if isinstance([predicted_labels], torch.Tensor) and isinstance(true_labels, torch.Tensor):
        # Use torch.isin if available (requires PyTorch 1.10+)
        if hasattr(torch, 'isin'):
            mask = torch.isin(true_labels, torch.tensor(list(selected_labels), device=predicted_labels.device))
        else:
            # Fallback if torch.isin is not available
            mask = torch.zeros_like(predicted_labels, dtype=torch.bool)
            for lab in selected_labels:
                mask |= (true_labels == lab)
        return predicted_labels[mask], true_labels[mask]
    
    elif isinstance(predicted_labels, np.ndarray) and isinstance(true_labels, np.ndarray):
        mask = np.isin(true_labels, list(selected_labels))
        return predicted_labels[mask], true_labels[mask]
    
    else:
        raise ValueError("features and labels must be of the same type (torch.Tensor or np.ndarray).")

def visualize_3d_features(
    features,
    labels,
    discrete_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    save_path=None,
    title=None,
    point_size=20,
    alpha=0.7
):
    """
    3D scatter of feature vectors with a discrete colorbar for labels 0…N-1,
    using matplotlib's default 3D view angle (elev=30, azim=-60).

    Args:
        features (np.ndarray): shape (N, 3).
        labels (array-like of int): length N, values in [0, len(discrete_colors)-1].
        discrete_colors (list of str): hex codes in the desired label order.
        save_path (str, optional): directory to save '3d_features.pdf'.
        title (str, optional): figure title.
        point_size (int): scatter marker size.
        alpha (float): scatter transparency.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    num_labels = len(discrete_colors)

    # build colormap and norm from the given color order
    cmap = ListedColormap(discrete_colors)
    norm = BoundaryNorm(np.arange(-0.5, num_labels + 0.5, 1), num_labels)

    # norm = BoundaryNorm(np.arange(len(discrete_colors) + 1), cmap.N)

    # scatter points, mapping each label to its color by index
    sc = ax.scatter(
        features[:, 0], features[:, 1], features[:, 2],
        c=labels, cmap=cmap, norm=norm,
        s=point_size, alpha=alpha
    )

    # show all labels 0…len(discrete_colors)-1 on the colorbar
    full_range = np.arange(len(discrete_colors))
    cbar = fig.colorbar(
        sc,
        ax=ax,
        boundaries=np.arange(len(discrete_colors) + 1),
        ticks=full_range,
    )
    cbar.set_label('Label')
    cbar.set_ticklabels([str(l) for l in full_range])

    # axes labels and optional title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title:
        ax.set_title(title)

    plt.tight_layout()

    # save if requested
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(
            os.path.join(save_path, "3d_features.pdf"),
            format="pdf",
            bbox_inches="tight"
        )

    plt.show()


def visualize_3d_features_plotly(features, labels):
    """
    Visualize a 3D feature space using Plotly.
    
    Parameters:
      features: numpy array of shape (num_points, 3)
      labels: numpy array of shape (num_points,) with class labels
    """
    unique_labels = sorted(np.unique(labels))
    # Use a discrete color mapping (similar to matplotlib's tab10)
    discrete_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    

    colors = [discrete_colors[int(l) % len(discrete_colors)] for l in labels]

    # We'll use a continuous colorscale for the main trace but hide its colorbar
    # and disable its legend. Dummy traces with discrete colors will provide the legend.
    fig = go.Figure()
    
    # Main animated trace (not shown in legend)
    fig.add_trace(go.Scatter3d(
        x = features[:, 0],
        y = features[:, 1],
        z = features[:, 2],
        mode = 'markers',
        marker = dict(
            size = 5,
            color = colors,
            showscale = False         # Hide the colorbar
        ),
        name = "",                # Set empty name to avoid default "trace 0"
        showlegend = False        # Do not show legend for this trace
    ))
    
    # Add dummy traces for each unique label (for legend)
    
    for i, ul in enumerate(unique_labels):
        fig.add_trace(go.Scatter3d(
            x = [None],
            y = [None],
            z = [None],
            mode = 'markers',
            marker = dict(
                size = 8,
                color =discrete_colors[ul % len(discrete_colors)]
            ),
            name = f'Label {ul}',
            showlegend = True
        ))
    
    # Update layout and add a legend title.
    fig.update_layout(
        title = "3D Feature Space of MNIST",
        scene = dict(
            xaxis = dict(title="X"),
            yaxis = dict(title="Y"),
            zaxis = dict(title="Z")
        ),
        legend=dict(
            title=dict(text="Digit Labels"),
            x=0.85,
            y=0.95
        )
    )
    
    fig.show()

# visualize clean images and noisy images
def visualize_images(clean_images, labels, noisy_images, n_images=10):
    fig, axes = plt.subplots(n_images, 2, figsize=(10, 20))
    for i in range(n_images):
        axes[i, 0].imshow(clean_images[i].squeeze(), cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Label: {labels[i]} (Clean)")
        
        axes[i, 1].imshow(noisy_images[i][0].squeeze(), cmap='gray')
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f"Label: {labels[i]} (Noisy)")
    plt.tight_layout()
    plt.show()

def animate_3d_features_plotly(features, labels, title="MNIST 3D Feature Space"):
    """
    Create an interactive 3D animation of features using Plotly with a discrete color scheme,
    and add a legend mapping colors to labels.
    
    Parameters:
        features: list of numpy arrays, each of shape (num_points, 3) representing a frame.
        labels: numpy array of shape (num_points,) with integer class labels.
    """
    num_frames = len(features)
    
    # Compute global axis limits across all frames.
    all_data = np.concatenate(features, axis=0)
    x_min, x_max = np.min(all_data[:, 0]), np.max(all_data[:, 0])
    y_min, y_max = np.min(all_data[:, 1]), np.max(all_data[:, 1])
    z_min, z_max = np.min(all_data[:, 2]), np.max(all_data[:, 2])
    
    # Add margins (10% of the range).
    x_margin = 0.1 * (x_max - x_min)
    y_margin = 0.1 * (y_max - y_min)
    z_margin = 0.1 * (z_max - z_min)
    
    # Define a discrete color mapping (example for up to 10 classes).
    discrete_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # Map each label to its corresponding color.
    colors = [discrete_colors[int(l) % len(discrete_colors)] for l in labels[0]]
    
    # Create the initial 3D scatter trace for frame 0.
    scatter = go.Scatter3d(
        x=features[0][:, 0],
        y=features[0][:, 1],
        z=features[0][:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=colors,  # Use discrete colors based on labels.
        ),
        showlegend=False  # Legend will be added via dummy traces.
    )
    
    # Build frames for the animation: update the x, y, and z values.
    frames = []
    for i, data in enumerate(features):
        colors = [discrete_colors[int(l) % len(discrete_colors)] for l in labels[i]]
        frames.append(go.Frame(
            data=[go.Scatter3d(
                x=data[:, 0],
                y=data[:, 1],
                z=data[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=colors  # Colors remain the same across frames.
                )
            )],
            name=str(i)
        ))
    
    # Define layout with play/pause buttons and a slider.
    layout = go.Layout(
        title="MNIST 3D Feature Space",
        scene=dict(
            xaxis=dict(range=[x_min - x_margin, x_max + x_margin]),
            yaxis=dict(range=[y_min - y_margin, y_max + y_margin]),
            zaxis=dict(range=[z_min - z_margin, z_max + z_margin])
        ),
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 2000, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 500}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Frame:",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [{
                "args": [[str(k)], {"frame": {"duration": 300, "redraw": True},
                                     "mode": "immediate",
                                     "transition": {"duration": 300}}],
                "label": str(k),
                "method": "animate"
            } for k in range(num_frames)]
        }],
        legend=dict(
            title="Digit Label",
            x=0.85,
            y=0.95
        )
    )
    
    # Create the figure with the main animated trace.
    fig = go.Figure(data=[scatter], layout=layout, frames=frames)
    
    # Add dummy traces for the legend: one trace per unique label.
    unique_labels = [x for x in range(10)]
    for ul in unique_labels:
        fig.add_trace(go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode='markers',
            marker=dict(
                size=8,
                color=discrete_colors[int(ul) % len(discrete_colors)]
            ),
            name=f'Label {ul}',
            showlegend=True
        ))
    
    # Display the interactive figure. You can rotate, zoom, and pan while the animation plays.
    fig.show()


def animate_2d_features_plotly(features, labels):
    """
    Create an interactive @D animation of features using Plotly with a discrete color scheme,
    and add a legend mapping colors to labels.
    
    Parameters:
        features: list of numpy arrays, each of shape (num_points, 3) representing a frame.
        labels: numpy array of shape (num_points,) with integer class labels.
    """
    num_frames = len(features)
    
    # Compute global axis limits across all frames.
    all_data = np.concatenate(features, axis=0)
    x_min, x_max = np.min(all_data[:, 0]), np.max(all_data[:, 0])
    y_min, y_max = np.min(all_data[:, 1]), np.max(all_data[:, 1])
    
    # Add margins (10% of the range).
    x_margin = 0.1 * (x_max - x_min)
    y_margin = 0.1 * (y_max - y_min)
    
    # Define a discrete color mapping (example for up to 10 classes).
    discrete_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # Map each label to its corresponding color.
    colors = [discrete_colors[int(l) % len(discrete_colors)] for l in labels]
    
    # Create the initial 3D scatter trace for frame 0.
    scatter = go.Scatter(
        x=features[0][:, 0],
        y=features[0][:, 1],
        mode='markers',
        marker=dict(
            size=4,
            color=colors,  # Use discrete colors based on labels.
        ),
        showlegend=False  # Legend will be added via dummy traces.
    )
    
    # Build frames for the animation: update the x, y, and z values.
    frames = []
    for i, data in enumerate(features):
        frames.append(go.Frame(
            data=[go.Scatter(
                x=data[:, 0],
                y=data[:, 1],
                mode='markers',
                marker=dict(
                    size=4,
                    color=colors  # Colors remain the same across frames.
                )
            )],
            name=str(i)
        ))
    
    # Define layout with play/pause buttons and a slider.
    layout = go.Layout(
        title="MNIST 2D Feature Space",
        scene=dict(
            xaxis=dict(range=[x_min - x_margin, x_max + x_margin]),
            yaxis=dict(range=[y_min - y_margin, y_max + y_margin]),
        ),
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 2000, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 500}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Frame:",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [{
                "args": [[str(k)], {"frame": {"duration": 300, "redraw": True},
                                     "mode": "immediate",
                                     "transition": {"duration": 300}}],
                "label": str(k),
                "method": "animate"
            } for k in range(num_frames)]
        }],
        legend=dict(
            title="Digit Label",
            x=0.85,
            y=0.95
        )
    )
    
    # Create the figure with the main animated trace.
    fig = go.Figure(data=[scatter], layout=layout, frames=frames)
    
    # Add dummy traces for the legend: one trace per unique label.
    unique_labels = sorted(np.unique(labels))
    for ul in unique_labels:
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                size=8,
                color=discrete_colors[int(ul) % len(discrete_colors)]
            ),
            name=f'Label {ul}',
            showlegend=True
        ))
    
    # Display the interactive figure. You can rotate, zoom, and pan while the animation plays.
    fig.show()


def logit_adjusted_adaptation(model, adaptation_loader, test_loader, log_frequency=100, 
                              epochs=10, class_distribution=[0.1 for i in range(10)], 
                              tau=1.0, do_adjust_logits=True):


    class_distribution = torch.Tensor(class_distribution).to('cuda')

    print(f"test_class_distribution: {class_distribution}")

    test_acc_list = []
    logit_norm_list = []
    reduced_feature_list = []
    predicted_labels_list = []
    gradient_norm_dict = {i: [] for i in range(10)}

    deepcopy_model = deepcopy(model)
    # test accuracy
    test_accuracy,_ = logit_adjusted_validation(deepcopy_model, test_loader, class_distribution, tau=0.0)
    test_acc_list.append((test_accuracy, 0))

    steps = 0
    total_steps = len(adaptation_loader) * epochs


    #### Model Configuration ####
    configure_model(model)
    bn_params, bn_names = collect_params(model, freeze_layers=["bn100"])
    print(bn_names)
    tent_optimizer = optim.Adam(bn_params, lr=1e-3)


    for epoch in range(epochs):

        for x_data, y_data in adaptation_loader:
            tent_optimizer.zero_grad()
            x_data, y_data = x_data.to('cuda'), y_data.to('cuda')
            logits = model(x_data)

            logits.retain_grad()

            # adjust logits
            if do_adjust_logits:
                adjusted_logits = logits + tau * torch.log(class_distribution + 1e-12)
            else:
                adjusted_logits = logits

            # if (steps+1) % log_frequency == 0 or steps == 0:
            #     print(f"Original Logits: {[x.item() for x in logits[0]]}")
            #     print(f"Adjusted Logits: {[x.item() for x in adjusted_logits[0]]}")

            loss = entropy_loss(adjusted_logits).mean(0)
            loss.backward()
            tent_optimizer.step()


            # extract the gradient norm of specific logit w.r.t. loss
            for i in range(10):
                gradient_norm_dict[i].append(torch.norm(logits.grad[0, i]).item())
            
            if (steps+1) % log_frequency == 0:
                deepcopy_model = deepcopy(model)
                # test accuracy
                test_accuracy,_ = logit_adjusted_validation(deepcopy_model, test_loader, class_distribution, tau=0.0)
                test_acc_list.append((test_accuracy, steps))
                print(f"Adaptation step {steps}/{total_steps}, Entropy Loss: {loss.item()}")
                print(f"Test Accuracy: {test_accuracy}")

            steps += 1

    return test_acc_list, reduced_feature_list, predicted_labels_list, gradient_norm_dict