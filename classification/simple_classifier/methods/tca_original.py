from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import CNN


class TCA(nn.Module):
    """
    Topological Consistency Adaptation (TCA) — standalone wrapper.
    
    Based on: "Maintaining Consistent Inter-Class Topology in Continual Test-Time Adaptation"
    
    - No dependency on TTAMethod or registries.
    - Implements full TCA algorithm with Mean Teacher framework.
    - Updates norm-layer affine params by default.
    - Maintains class centroids and topology consistency.

    Parameters
    ----------
    model : nn.Module
        Pretrained classifier that outputs logits (B, C).
    num_classes : int
        Number of classes in the dataset.
    optimizer : torch.optim.Optimizer, optional
        If None, Adam on norm-layer affine params is created.
    device : str, optional
        "cuda" or "cpu". Defaults to available device.
    mixed_precision : bool
        Enable AMP (CUDA only).
    lr : float
        Learning rate if creating a default optimizer.
    lambda_align : float
        Weight for alignment loss (λ1 in paper).
    lambda_topology : float
        Weight for topology loss (λ2 in paper).
    alpha : float
        EMA coefficient for centroid updates.
    beta : float
        EMA coefficient for topology weight updates.
    t_param : float
        Temperature parameter for uniformity losses.
    teacher_momentum : float
        EMA momentum for teacher model updates.
    n_augmentations : int
        Number of augmentations for teacher averaging.
    confidence_threshold : float
        Threshold for augmentation averaging policy.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        device: str | None = None,
        mixed_precision: bool = False,
        lr: float = 1e-3,
        lambda_align: float = 0.025,
        lambda_topology: float = 0.15,
        alpha: float = 0.1,
        beta: float = 0.9,
        t_param: float = 2.0,
        teacher_momentum: float = 0.999,
        n_augmentations: int = 32,
        confidence_threshold: float = 0.72,
    ) -> None:
        super().__init__()
        
        # Store hyperparameters
        self.num_classes = num_classes
        self.lambda_align = lambda_align
        self.lambda_topology = lambda_topology
        self.alpha = alpha
        self.beta = beta
        self.t_param = t_param
        self.teacher_momentum = teacher_momentum
        self.n_augmentations = n_augmentations
        self.confidence_threshold = confidence_threshold
        
        # Device setup
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mixed_precision = bool(mixed_precision) and self.device.startswith("cuda")
        
        # Model setup
        self.original_model = deepcopy(model)
        self.model = model  # Student model
        self.teacher_model = deepcopy(model)  # Teacher model
        self.anchor_model = deepcopy(model)   # Anchor model (frozen)
        
        # Freeze teacher and anchor models
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        for param in self.anchor_model.parameters():
            param.requires_grad = False
            
        # Configure models
        self.configure_model()
        
        # Split model into feature extractor and classifier
        self.feature_extractor, self.classifier = self.split_model(self.model, 'CNN', 'mnist')
        print(f"Feature extractor: {self.feature_extractor} , Classifier: {self.classifier}")
        self.teacher_feature_extractor, self.teacher_classifier = self.split_model(self.teacher_model, 'CNN', 'mnist')
        
        # Default optimizer over norm params
        if optimizer is None:
            params = self.collect_norm_params(self.model)
            if not params:  # fallback if no norms are present
                params = list(self.model.parameters())
            optimizer = torch.optim.Adam(params, lr=lr)
        self.optimizer = optimizer
        
        # Mixed precision setup
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        
        # TCA state initialization
        self.class_centroids = {}
        self.prev_topology_weights = None
        self.first_batch = True
        
        # Loss functions
        self.symmetric_cross_entropy = SymmetricCrossEntropy()
        
        # Move to device
        self.to(self.device)

    # ----------------- Core API -----------------
    def loss_calculation(self, batch):
        """Calculate TCA loss following Algorithm 1."""
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(self.device, non_blocking=True)
        
        # Generate teacher predictions with optional augmentation averaging
        with torch.no_grad():
            teacher_logits = self._compute_teacher_predictions(x)
            pseudo_labels = teacher_logits.argmax(dim=1)
        
        # Extract student features and predictions
        features_clean = self._extract_features(x, self.feature_extractor)
        logits_clean = self.classifier(features_clean)
        
        # Extract features with augmentation
        x_aug1 = self._augment_data(x)
        features_aug1 = self._extract_features(x_aug1, self.feature_extractor)
        logits_aug1 = self.classifier(features_aug1)
        
        # Second augmentation for alignment loss
        x_aug2 = self._augment_data(x)
        features_aug2 = self._extract_features(x_aug2, self.feature_extractor)
        
        # Update class centroids (Equations 2-3)
        self._update_class_centroids(features_aug1, pseudo_labels)
        
        # Compute topology losses
        if self.first_batch:
            # First batch: standard uniformity loss (Algorithm 1, lines 4-6)
            inter_loss = self._compute_inter_class_uniformity_loss(use_weighting=False)
            intra_loss = self._compute_intra_class_compactness_loss(features_aug1, pseudo_labels)
            self.first_batch = False
        else:
            # Subsequent batches: weighted uniformity loss (Algorithm 1, lines 8-10)
            self._compute_batch_imbalance_weights(pseudo_labels)
            inter_loss = self._compute_inter_class_uniformity_loss(use_weighting=True)
            intra_loss = self._compute_intra_class_compactness_loss(features_aug1, pseudo_labels)
        
        # Compute alignment loss (Equation 12)
        align_loss = self._compute_alignment_loss(features_aug1, features_aug2)
        
        # Topology consistency loss
        topology_loss = inter_loss + intra_loss
        
        # Self-training loss (Symmetric Cross-Entropy, Equation 11)
        sce_loss1 = self.symmetric_cross_entropy(logits_clean, teacher_logits).mean()
        sce_loss2 = self.symmetric_cross_entropy(logits_aug1, teacher_logits).mean()
        sce_loss = 0.5 * sce_loss1 + 0.5 * sce_loss2
        
        # Total loss (Equation 13)
        total_loss = (sce_loss + 
                     self.lambda_align * align_loss + 
                     self.lambda_topology * topology_loss)
        
        # Combined predictions (weighted ensemble)
        with torch.no_grad():
            final_logits = 0.3 * logits_clean + 0.7 * teacher_logits
        
        return final_logits, total_loss

    @torch.enable_grad()
    def forward_and_adapt(self, batch):
        """Forward and adapt following TCA algorithm."""
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                outputs, loss = self.loss_calculation(batch)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            outputs, loss = self.loss_calculation(batch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        
        # Update teacher model with EMA
        self._update_teacher_model()
        
        return outputs

    # Convenience alias
    step = forward_and_adapt

    # ----------------- TCA Algorithm Implementation -----------------
    def _compute_teacher_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Generate teacher predictions with optional augmentation averaging."""
        with torch.no_grad():
            # Check confidence using anchor model
            anchor_logits = self.anchor_model(x)
            anchor_prob = F.softmax(anchor_logits, dim=1).max(1)[0]
            
            if anchor_prob.mean() < self.confidence_threshold:
                # Use augmentation averaging for low confidence
                teacher_outputs = []
                for _ in range(self.n_augmentations):
                    aug_x = self._augment_data(x)
                    output = self.teacher_model(aug_x)
                    teacher_outputs.append(output)
                return torch.stack(teacher_outputs).mean(0)
            else:
                # Single forward pass for high confidence
                return self.teacher_model(x)
    
    def _update_class_centroids(self, features: torch.Tensor, pseudo_labels: torch.Tensor) -> None:
        """Update class centroids using EMA (Equations 2-3)."""
        with torch.no_grad():
            for class_k in range(self.num_classes):
                class_mask = (pseudo_labels == class_k)
                
                if class_mask.sum() > 0:
                    # Compute current batch centroid (Equation 2)
                    current_centroid = features[class_mask].mean(dim=0)
                    
                    if class_k in self.class_centroids:
                        # Update with EMA (Equation 3)
                        self.class_centroids[class_k] = (
                            self.alpha * self.class_centroids[class_k] + 
                            (1 - self.alpha) * current_centroid
                        )
                    else:
                        self.class_centroids[class_k] = current_centroid
    
    def _compute_inter_class_uniformity_loss(self, use_weighting: bool = False) -> torch.Tensor:
        """Compute inter-class uniformity loss (Equations 4 and 10)."""
        if len(self.class_centroids) < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Get centroids
        centroid_indices = sorted(list(self.class_centroids.keys()))
        centroids = torch.stack([self.class_centroids[k] for k in centroid_indices])
        
        # Compute pairwise squared distances
        distance_matrix = torch.cdist(centroids, centroids, p=2).pow(2)
        
        # Get upper triangular indices (i < j pairs)
        num_centroids = len(centroid_indices)
        upper_triangular_indices = torch.triu_indices(
            num_centroids, num_centroids, offset=1, device=self.device
        )
        distances = distance_matrix[upper_triangular_indices[0], upper_triangular_indices[1]]
        
        if use_weighting and hasattr(self, 'current_class_weights'):
            # Weighted uniformity loss (Equation 10)
            weights_matrix = torch.zeros((num_centroids, num_centroids), device=self.device)
            for i, class_i in enumerate(centroid_indices):
                for j, class_j in enumerate(centroid_indices):
                    if i < j:
                        weights_matrix[i, j] = (
                            self.current_class_weights[class_i] + 
                            self.current_class_weights[class_j]
                        ) / 2
            
            weights = weights_matrix[upper_triangular_indices[0], upper_triangular_indices[1]]
            weighted_exp_dist = weights * torch.exp(-self.t_param * distances)
            uniformity_loss = torch.log(weighted_exp_dist.sum() / (weights.sum() + 1e-8) + 1e-8)
        else:
            # Standard uniformity loss (Equation 4)
            exp_distances = torch.exp(-self.t_param * distances)
            uniformity_loss = torch.log(
                2.0 / (num_centroids * (num_centroids - 1)) * exp_distances.sum() + 1e-8
            )
        
        return uniformity_loss
    
    def _compute_intra_class_compactness_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute intra-class compactness loss (Equation 5)."""
        total_loss = 0.0
        valid_classes = 0
        
        for class_k in range(self.num_classes):
            class_mask = (labels == class_k)
            class_features = features[class_mask]
            
            if len(class_features) >= 2:
                # Compute pairwise distances within class
                pairwise_distances = torch.pdist(class_features, p=2).pow(2)
                
                if len(pairwise_distances) > 0:
                    # Average intra-class distance
                    mean_intra_dist = pairwise_distances.mean()
                    # Compactness term
                    class_compactness = torch.exp(-self.t_param * mean_intra_dist)
                    total_loss += torch.log(class_compactness + 1e-8)
                    valid_classes += 1
        
        return total_loss / max(valid_classes, 1)
    
    def _compute_batch_imbalance_weights(self, pseudo_labels: torch.Tensor) -> None:
        """Compute batch imbalance topology weighting (Equations 7-9)."""
        with torch.no_grad():
            # Compute class frequencies (Equation 7)
            class_counts = torch.bincount(pseudo_labels, minlength=self.num_classes).float()
            sqrt_counts = torch.sqrt(class_counts + 1e-6)
            class_weights = sqrt_counts / (sqrt_counts.sum() + 1e-6)
            
            # Store current class weights
            self.current_class_weights = {k: class_weights[k] for k in range(self.num_classes)}
            
            # EMA update of weights (Equation 9)
            if self.prev_topology_weights is not None:
                for k in range(self.num_classes):
                    self.current_class_weights[k] = (
                        self.beta * self.prev_topology_weights[k] + 
                        (1 - self.beta) * self.current_class_weights[k]
                    )
            
            # Store for next iteration
            self.prev_topology_weights = {k: v.clone() for k, v in self.current_class_weights.items()}
    
    def _compute_alignment_loss(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """Compute alignment loss between augmented features (Equation 12)."""
        # L_align = -E[||f(x) - f(y)||_2^2] (note the negative sign)
        distances_squared = torch.norm(features1 - features2, dim=1, p=2).pow(2)
        return -distances_squared.mean()
    
    def _update_teacher_model(self) -> None:
        """Update teacher model using EMA."""
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher_model.parameters(), 
                                                  self.model.parameters()):
                teacher_param.data.mul_(self.teacher_momentum).add_(
                    student_param.data, alpha=(1 - self.teacher_momentum)
                )

    # ----------------- Helpers -----------------
    @staticmethod
    def _augment_data(x: torch.Tensor, std: float = 0.02) -> torch.Tensor:
        """Simple data augmentation with Gaussian noise."""
        noise = torch.randn_like(x) * std
        return torch.clamp(x + noise, 0, 1)
    
    @staticmethod
    def _extract_features(x: torch.Tensor, feature_extractor: nn.Module) -> torch.Tensor:
        """Extract features using the feature extractor."""
        features = feature_extractor(x)
        # Flatten if needed
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        return features
    
    @staticmethod
    def split_model(model, arch_name: str, dataset_name: str):
        """
        Split up the model into an encoder and a classifier.
        This is required for methods like RMT and AdaContrast
        Input:
            model: Model to be split up
            arch_name: Name of the network
            dataset_name: Name of the dataset
        Returns:
            encoder: The encoder of the model
            classifier The classifier of the model
        """
        if hasattr(model, "model") and hasattr(model.model, "pretrained_cfg") and hasattr(model.model, model.model.pretrained_cfg["classifier"]):
            # split up models loaded from timm
            classifier = deepcopy(getattr(model.model, model.model.pretrained_cfg["classifier"]))
            encoder = model
            encoder.model.reset_classifier(0)
            if isinstance(model, ImageNetXWrapper):
                encoder = nn.Sequential(encoder.normalize, encoder.model)

        elif arch_name == "CNN" or isinstance(model, CNN):
            # Custom CNN model splitting
            # Encoder: conv layers + batch norm + pooling + first FC layer
            encoder = nn.Sequential(
                model.conv1,
                model.bn1,
                nn.ReLU(),
                model.pool,
                model.conv2,
                model.bn2,
                nn.ReLU(),
                model.pool,
                nn.Flatten(),
                model.fc1,
                nn.ReLU()
            )
            # Classifier: final linear layer
            classifier = model.fc2
        elif arch_name == "Standard" and dataset_name in {"cifar10", "cifar10_c"}:
            encoder = nn.Sequential(*list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
            classifier = model.fc
        elif arch_name == "Hendrycks2020AugMix_WRN":
            normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
            encoder = nn.Sequential(normalization, *list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
            classifier = model.fc
        elif arch_name == "Hendrycks2020AugMix_ResNeXt":
            normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
            encoder = nn.Sequential(normalization, *list(model.children())[:2], nn.ReLU(), *list(model.children())[2:-1], nn.Flatten())
            classifier = model.classifier
        elif dataset_name == "domainnet126":
            encoder = model.encoder
            classifier = model.fc
        elif "resnet" in arch_name or "resnext" in arch_name or "wide_resnet" in arch_name or arch_name in {"Standard_R50", "Hendrycks2020AugMix", "Hendrycks2020Many", "Geirhos2018_SIN"}:
            encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.Flatten())
            classifier = model.model.fc
        elif "densenet" in arch_name:
            encoder = nn.Sequential(model.normalize, model.model.features, nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
            classifier = model.model.classifier
        elif "efficientnet" in arch_name:
            encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool, nn.Flatten())
            classifier = model.model.classifier
        elif "mnasnet" in arch_name:
            encoder = nn.Sequential(model.normalize, model.model.layers, nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
            classifier = model.model.classifier
        elif "shufflenet" in arch_name:
            encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
            classifier = model.model.fc
        elif "vit_" in arch_name and not "maxvit_" in arch_name:
            encoder = TransformerWrapper(model)
            classifier = model.model.heads.head
        elif "swin_" in arch_name:
            encoder = nn.Sequential(model.normalize, model.model.features, model.model.norm, model.model.permute, model.model.avgpool, model.model.flatten)
            classifier = model.model.head
        elif "convnext" in arch_name:
            encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool)
            classifier = model.model.classifier
        elif arch_name == "mobilenet_v2":
            encoder = nn.Sequential(model.normalize, model.model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
            classifier = model.model.classifier
        else:
            raise ValueError(f"The model architecture '{arch_name}' is not supported for dataset '{dataset_name}'.")

        # add a masking layer to the classifier
        if dataset_name in ["imagenet_a", "imagenet_r", "imagenet_v2", "imagenet_d109"]:
            mask = eval(f"{dataset_name.upper()}_MASK")
            classifier = nn.Sequential(classifier, ImageNetXMaskingLayer(mask))

        return encoder, classifier

    @staticmethod
    def collect_norm_params(model: nn.Module):
        """Collect normalization layer parameters."""
        params = []
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for name, p in m.named_parameters(recurse=False):
                    if name in ("weight", "bias"):
                        params.append(p)
        return params

    def configure_model(self) -> None:
        """Configure model for TCA adaptation."""
        self.model.eval()
        self.model.requires_grad_(False)
        
        # Enable gradients for normalization layers
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)

    def reset(self) -> None:
        """Reset TCA to initial state."""
        # Reset model parameters
        self.model.load_state_dict(self.original_model.state_dict())
        self.teacher_model.load_state_dict(self.original_model.state_dict())
        self.anchor_model.load_state_dict(self.original_model.state_dict())
        
        # Reset TCA state
        self.class_centroids = {}
        self.prev_topology_weights = None
        self.first_batch = True
        
        # Reset optimizer
        self.optimizer.state = {}
        
        # Configure model again
        self.configure_model()


class SymmetricCrossEntropy(nn.Module):
    """Symmetric Cross Entropy Loss for TCA."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        student_probs = F.softmax(student_logits, dim=1)
        teacher_probs = F.softmax(teacher_logits.detach(), dim=1)
        
        loss1 = -torch.sum(teacher_probs * torch.log(student_probs + 1e-8), dim=1)
        loss2 = -torch.sum(student_probs * torch.log(teacher_probs + 1e-8), dim=1)
        
        return loss1 + loss2