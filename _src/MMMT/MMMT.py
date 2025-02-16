import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalMultiTaskModel(nn.Module):
    """
    A generic multi-modal, multi-task model that accepts inputs from multiple modalities,
    encodes each modality separately, fuses the resulting latent representations, learns
    a joint representation, and then feeds the joint representation into multiple task-specific heads.
    
    Attributes:
        encoders (nn.ModuleDict): Mapping from modality names to modality-specific encoder modules.
        task_heads (nn.ModuleDict): Mapping from task names to task-specific head modules.
        fusion_method (callable): Function that fuses modality-specific features. Default concatenates along feature dimension.
        joint_network (nn.Module): Network to learn the joint representation from the fused features.
        variational (bool): If True, treats the joint representation as the mean of a Gaussian.
        sigma (float): Fixed standard deviation for variational reparameterization. Default is 1.0.
    """
    def __init__(self, encoders: dict, task_heads: dict, fusion_method: callable = None, 
                 joint_network: nn.Module = None, variational: bool = False, sigma: float = 1.0):
        """
        Initializes the MultiModalMultiTaskModel.
        
        Args:
            encoders (dict): Dictionary mapping modality names to encoder nn.Module objects.
            task_heads (dict): Dictionary mapping task names to task-specific head nn.Module objects.
            fusion_method (callable, optional): Function to fuse encoded representations. If None, defaults to concatenation.
            joint_network (nn.Module, optional): Network for joint representation learning. Defaults to identity mapping.
            variational (bool, optional): Whether to enable variational reparameterization. Defaults to False.
            sigma (float, optional): Standard deviation for the reparameterization noise if variational=True. Defaults to 1.0.
        """
        super(MultiModalMultiTaskModel, self).__init__()
        
        # Module dictionaries for encoders and task heads.
        self.encoders = nn.ModuleDict(encoders)
        self.task_heads = nn.ModuleDict(task_heads)
        
        # Set fusion method; default is to concatenate features along dimension 1.
        self.fusion_method = fusion_method if fusion_method is not None else self.default_fusion
        
        # Joint network to learn a shared representation.
        self.joint_network = joint_network if joint_network is not None else nn.Identity()
        
        # Variational parameters.
        self.variational = variational
        self.sigma = sigma

    def default_fusion(self, features: dict) -> torch.Tensor:
        """
        Default fusion method: concatenates all modality-specific features along dimension 1.
        
        Args:
            features (dict): A dictionary mapping modalities to their feature tensors.
            
        Returns:
            torch.Tensor: The concatenated tensor.
        """
        return torch.cat(list(features.values()), dim=1)

    def encode(self, inputs: dict) -> dict:
        """
        Encodes each modality using its respective encoder.
        
        Args:
            inputs (dict): A dictionary mapping modality names to input tensors.
            
        Returns:
            dict: A dictionary mapping modality names to encoded features.
        """
        features = {}
        for modality, encoder in self.encoders.items():
            if modality not in inputs:
                raise ValueError(f"Input for modality '{modality}' is missing.")
            features[modality] = encoder(inputs[modality])
        return features

    def fuse(self, features: dict) -> torch.Tensor:
        """
        Fuses the modality-specific features into a single latent representation.
        
        Args:
            features (dict): A dictionary mapping modality names to feature tensors.
            
        Returns:
            torch.Tensor: The fused representation.
        """
        return self.fusion_method(features)
    
    def compute_joint_representation(self, fused_representation: torch.Tensor) -> torch.Tensor:
        """
        Computes the joint representation from the fused modality features.
        
        Args:
            fused_representation (torch.Tensor): The fused latent representation.
            
        Returns:
            torch.Tensor: The joint representation.
        """
        return self.joint_network(fused_representation)
    
    def reparameterize(self, mean: torch.Tensor) -> torch.Tensor:
        """
        If variational, performs reparameterization using the provided sigma; otherwise, returns mean.
        
        Args:
            mean (torch.Tensor): The mean (joint representation) from the joint network.
            
        Returns:
            torch.Tensor: The latent representation after reparameterization.
        """
        if self.variational:
            noise = torch.randn_like(mean)
            return mean + self.sigma * noise
        else:
            return mean

    def decode(self, latent_representation: torch.Tensor) -> dict:
        """
        Decodes the latent representation into task-specific outputs.
        
        Args:
            latent_representation (torch.Tensor): The latent representation.
            
        Returns:
            dict: A dictionary mapping task names to their output tensors.
        """
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(latent_representation)
        return outputs

    def forward(self, inputs: dict) -> dict:
        """
        Forward pass that processes inputs from multiple modalities and computes task-specific outputs.
        
        Args:
            inputs (dict): A dictionary mapping modality names to input tensors.
            
        Returns:
            dict: A dictionary mapping task names to output tensors.
        """
        # Step 1: Encode each modality.
        features = self.encode(inputs)
        # Step 2: Fuse modality-specific features.
        fused_representation = self.fuse(features)
        # Step 3: Compute the joint representation.
        joint_representation = self.compute_joint_representation(fused_representation)
        # Step 4: Reparameterize if using variational encoding.
        latent_representation = self.reparameterize(joint_representation)
        # Step 5: Decode to get task-specific outputs.
        outputs = self.decode(latent_representation)
        return outputs

    # --- Additional utility methods ---
    
    def freeze_encoders(self):
        """
        Freezes the parameters of all modality-specific encoders.
        """
        for param in self.encoders.parameters():
            param.requires_grad = False
            
    def unfreeze_encoders(self):
        """
        Unfreezes the parameters of all modality-specific encoders.
        """
        for param in self.encoders.parameters():
            param.requires_grad = True
    
    def freeze_task_heads(self):
        """
        Freezes the parameters of all task-specific heads.
        """
        for param in self.task_heads.parameters():
            param.requires_grad = False
            
    def unfreeze_task_heads(self):
        """
        Unfreezes the parameters of all task-specific heads.
        """
        for param in self.task_heads.parameters():
            param.requires_grad = True

    def freeze_joint_network(self):
        """
        Freezes the parameters of the joint network.
        """
        for param in self.joint_network.parameters():
            param.requires_grad = False
        
    def unfreeze_joint_network(self):
        """
        Unfreezes the parameters of the joint network.
        """
        for param in self.joint_network.parameters():
            param.requires_grad = True

if __name__ == "__main__":
    # Define a simple MLP-based encoder for text.
    class TextEncoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int):
            super(TextEncoder, self).__init__()
            self.fc = nn.Linear(input_dim, hidden_dim)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.relu(self.fc(x))
    
    # Define a simple encoder for images.
    class ImageEncoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int):
            super(ImageEncoder, self).__init__()
            self.fc = nn.Linear(input_dim, hidden_dim)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.relu(self.fc(x))
    
    # Define a classification head that outputs logits.
    class ClassificationHead(nn.Module):
        def __init__(self, input_dim: int, num_classes: int):
            super(ClassificationHead, self).__init__()
            self.fc = nn.Linear(input_dim, num_classes)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)
    
    # Define a regression head.
    class RegressionHead(nn.Module):
        def __init__(self, input_dim: int, output_dim: int = 1):
            super(RegressionHead, self).__init__()
            self.fc = nn.Linear(input_dim, output_dim)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)
    
    # Hyperparameters for demonstration.
    text_input_dim = 50    # Dimensionality of text features.
    image_input_dim = 100  # Dimensionality of image features (flattened).
    hidden_dim = 64        # Feature dimension after encoding.
    num_classes = 10       # Number of classes for classification.
    
    # Define modality-specific encoders.
    encoders = {
        "text": TextEncoder(input_dim=text_input_dim, hidden_dim=hidden_dim),
        "image": ImageEncoder(input_dim=image_input_dim, hidden_dim=hidden_dim)
    }
    
    # Define a joint network that processes the fused representation.
    # Here, the fused representation has dimension 2 * hidden_dim.
    joint_network = nn.Sequential(
        nn.Linear(2 * hidden_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 2 * hidden_dim)
    )
    
    # Define task-specific heads.
    task_heads = {
        "classification": ClassificationHead(input_dim=2 * hidden_dim, num_classes=num_classes),
        "regression": RegressionHead(input_dim=2 * hidden_dim)
    }
    
    # Create the multi-modal, multi-task model with variational encoding enabled.
    model = MultiModalMultiTaskModel(
        encoders=encoders, 
        task_heads=task_heads, 
        joint_network=joint_network, 
        variational=True,  # Enable reparameterization.
        sigma=1.0        # User-defined sigma for noise scaling.
    )
    
    # Create dummy inputs (batch size of 8).
    dummy_text = torch.randn(8, text_input_dim)
    dummy_image = torch.randn(8, image_input_dim)
    
    # Package inputs as a dictionary.
    inputs = {
        "text": dummy_text,
        "image": dummy_image
    }
    
    # Forward pass.
    outputs = model(inputs)
    
    # Display the shapes of outputs for each task.
    for task, output in outputs.items():
        print(f"Task: {task}, Output Shape: {output.shape}")

