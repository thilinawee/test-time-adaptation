import torch
import torch.nn as nn

# Define the CNN model with two Batch Normalization layers
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn2   = nn.BatchNorm2d(64)
        
        # Fully connected layers
        # After two conv layers and pooling operations, the spatial dimensions reduce.
        # MNIST images are 28x28. After two rounds of 3x3 conv (without padding) and 2x2 pooling:
        #   After conv1: 28-3+1 = 26 -> after pool: 26/2 = 13 (floor division)
        #   After conv2: 13-3+1 = 11 -> after pool: 11/2 = 5 (floor division)
        # Thus, the feature map size is 64 x 5 x 5.
        self.fc1 = nn.Linear(64 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # First conv block: Conv -> BatchNorm -> ReLU -> Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        # Second conv block: Conv -> BatchNorm -> ReLU -> Pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    


class CNN3(nn.Module):
    # A model with 
    def __init__(self):
        super(CNN3, self).__init__()
    # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn2   = nn.BatchNorm2d(64)
        
        # Fully connected layers
        # From the comment: feature map size is 64 x 5 x 5 = 1600 features.
        self.fc1 = nn.Linear(64 * 5 * 5, 64)
        # Bottleneck layer: compress to 3 features (for direct 3D visualization)
        self.fc2 = nn.Linear(64, 3)
        # Final classification layer: map 3 features to 10 classes
        self.fc3 = nn.Linear(3, 10)
    
    def forward(self, x):
        # First conv block
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        # Second conv block
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        # Flatten
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        # Get 3D features
        features = self.fc2(x)
        # Classification logits
        logits = self.fc3(features)
        return logits