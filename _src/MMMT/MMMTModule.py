from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import pickle
from pathlib import Path

# Import our modules (assumed to be available in the same directory)
from DFTransformerEncoderDecoder import DFTransformerEncoder, DFTransformerDecoder
from MMMT import MultiModalMultiTaskModel
import MLheads
from TabularReconstructionLoss import TabularReconstructionLoss

class TaskType(Enum):
    """Enumeration of supported task types"""
    REGRESSION = auto()
    CLASSIFICATION = auto()
    BINOMIAL = auto()
    NEGATIVE_BINOMIAL = auto()
    POISSON = auto()
    TIME_TO_EVENT = auto()
    CLUSTERING = auto()
    UMAP = auto()
    TSNE = auto()
    PCA = auto()
    RECONSTRUCTION = auto()

@dataclass
class TaskConfig:
    """Configuration for a specific task"""
    task_type: TaskType
    num_classes: Optional[int] = None
    time_bins: Optional[int] = None
    num_events: Optional[int] = 1
    r: Optional[float] = 1.0  # For negative binomial
    n_components: Optional[int] = 2  # For dimension reduction
    loss_weights: Optional[Dict[str, float]] = None  # For reconstruction

class TabularEncoder(nn.Module):
    """Wraps DFTransformerEncoder as an nn.Module for use in MMMT"""
    def __init__(self, df_encoder: DFTransformerEncoder):
        super().__init__()
        self.df_encoder = df_encoder

    def forward(self, x: pd.DataFrame) -> torch.Tensor:
        return self.df_encoder.forward(x, return_tokens=False)

class MultiTaskModel:
    """
    A unified model for multi-modal, multi-task learning.
    Currently supports tabular data with various supervised and unsupervised tasks.
    """
    def __init__(
        self,
        numeric_cols: List[str],
        categorical_cols: List[str],
        task_configs: Dict[str, TaskConfig],
        embed_dim: int = 32,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.1,
        use_missing_indicator: bool = True,
        handle_unknown: bool = True,
        joint_hidden_dim: int = 64,
        variational: bool = False,
        sigma: float = 1.0,
        device: torch.device = torch.device("cpu"),
        learning_rate: float = 1e-3
    ):
        self.device = device
        self.task_configs = task_configs
        
        # Initialize tabular encoder
        self.df_encoder = DFTransformerEncoder(
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            embed_dim=embed_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            dropout=dropout,
            use_missing_indicator=use_missing_indicator,
            handle_unknown=handle_unknown,
            device=self.device
        )
        self.encoder_module = TabularEncoder(self.df_encoder)
        
        # Initialize task heads
        self.task_heads = self._create_task_heads(embed_dim * 2)  # 2x for joint representation
        
        # Joint network for representation learning
        self.joint_network = nn.Sequential(
            nn.Linear(embed_dim, joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(joint_hidden_dim, embed_dim * 2)
        )
        
        # Initialize MMMT model
        self.model = MultiModalMultiTaskModel(
            encoders={"tabular": self.encoder_module},
            task_heads=self.task_heads,
            joint_network=self.joint_network,
            variational=variational,
            sigma=sigma
        ).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.reconstruction_loss = None  # Will be set up after encoder is fitted

    def _is_differentiable_task(self, task_type: TaskType) -> bool:
        """Determines if a task type is differentiable"""
        return task_type not in {TaskType.UMAP, TaskType.TSNE, TaskType.PCA, TaskType.RECONSTRUCTION}

    def _create_task_heads(self, input_dim: int) -> Dict[str, nn.Module]:
        """Creates task-specific heads for differentiable tasks only"""
        heads = {}
        self.dim_reduction_heads = {}  # Store non-differentiable heads separately
        
        for name, config in self.task_configs.items():
            if not self._is_differentiable_task(config.task_type):
                if config.task_type in {TaskType.UMAP, TaskType.TSNE, TaskType.PCA}:
                    n_components = config.n_components or 2
                    if config.task_type == TaskType.UMAP:
                        self.dim_reduction_heads[name] = MLheads.UMAPHead(n_components=n_components)
                    elif config.task_type == TaskType.TSNE:
                        self.dim_reduction_heads[name] = MLheads.TSNEHead(n_components=n_components)
                    else:  # PCA
                        self.dim_reduction_heads[name] = MLheads.PCAHead(n_components=n_components)
                continue
                
            if config.task_type == TaskType.REGRESSION:
                heads[name] = MLheads.RegressionHead(input_dim=input_dim)
            elif config.task_type == TaskType.CLASSIFICATION:
                heads[name] = MLheads.ClassificationHead(
                    input_dim=input_dim,
                    num_classes=config.num_classes or 2
                )
            elif config.task_type == TaskType.BINOMIAL:
                heads[name] = MLheads.BinomialClassificationHead(input_dim=input_dim)
            elif config.task_type == TaskType.NEGATIVE_BINOMIAL:
                heads[name] = MLheads.NegativeBinomialHead(
                    input_dim=input_dim,
                    r=config.r or 1.0
                )
            elif config.task_type == TaskType.POISSON:
                heads[name] = MLheads.PoissonRegressionHead(input_dim=input_dim)
            elif config.task_type == TaskType.TIME_TO_EVENT:
                heads[name] = MLheads.DeepHitHead(
                    input_dim=input_dim,
                    time_bins=config.time_bins or 10,
                    num_events=config.num_events or 1
                )
            elif config.task_type == TaskType.CLUSTERING:
                heads[name] = MLheads.ClusteringHead(
                    input_dim=input_dim,
                    num_clusters=config.num_classes or 3
                )
                    
        return heads

    def _setup_reconstruction_loss(self) -> Optional[TabularReconstructionLoss]:
        """Sets up reconstruction loss if configured"""
        recon_config = next(
            (cfg for name, cfg in self.task_configs.items() 
             if cfg.task_type == TaskType.RECONSTRUCTION),
            None
        )
        
        if recon_config is None:
            return None
            
        return TabularReconstructionLoss(
            encoder=self.df_encoder,
            decoder=DFTransformerDecoder(self.df_encoder),
            **(recon_config.loss_weights or {})
        )

    def fit_encoder(self, df: pd.DataFrame) -> None:
        """Fits the tabular encoder on training data and sets up reconstruction loss if needed"""
        self.df_encoder.fit(df)
        # Now that encoder is fitted, we can set up reconstruction loss
        self.reconstruction_loss = self._setup_reconstruction_loss()

    def fit(
        self,
        df: pd.DataFrame,
        target_data: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
        epochs: int = 10,
        batch_size: int = 32,
        targets_to_train: Optional[List[str]] = None,
        verbose: bool = True
    ) -> List[float]:
        """
        Trains the model on provided data.
        
        Args:
            df: Input DataFrame
            target_data: Dictionary mapping target names to their values
            epochs: Number of training epochs
            batch_size: Training batch size
            targets_to_train: Optional list of targets to train on
            verbose: Whether to print training progress
            
        Returns:
            List of epoch losses
        """
        self.model.train()
        num_samples = len(df)
        epoch_losses = []
        
        active_targets = targets_to_train or list(self.model.task_heads.keys())
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_df = df.iloc[start_idx:end_idx]
                
                # Forward pass
                outputs = self.model({"tabular": batch_df})
                
                # Compute loss
                batch_loss = self._compute_batch_loss(
                    outputs, target_data, start_idx, end_idx, active_targets
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                
                epoch_loss += batch_loss.item()
                
            avg_epoch_loss = epoch_loss / ((num_samples + batch_size - 1) // batch_size)
            epoch_losses.append(avg_epoch_loss)
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f}")
                
        return epoch_losses

    def _compute_batch_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target_data: Dict[str, Any],
        start_idx: int,
        end_idx: int,
        active_targets: List[str]
    ) -> torch.Tensor:
        """Computes the combined loss for a batch"""
        total_loss = torch.tensor(0.0, device=self.device)
        
        for target in active_targets:
            if target not in self.model.task_heads:
                continue
                
            config = self.task_configs[target]
            if config.task_type in {TaskType.UMAP, TaskType.TSNE, TaskType.PCA}:
                continue
                
            head = self.model.task_heads[target]
            
            if config.task_type == TaskType.TIME_TO_EVENT:
                times, events = target_data[target]
                batch_target = (
                    times[start_idx:end_idx].to(self.device),
                    events[start_idx:end_idx].to(self.device)
                )
            else:
                batch_target = target_data[target][start_idx:end_idx].to(self.device)
                
            total_loss += head.loss(outputs[target], batch_target)
            
        # Add reconstruction loss if configured and "reconstruction" is in active_targets
        if (self.reconstruction_loss is not None and 
            ("reconstruction" in active_targets or "reconstruction" in self.task_configs)):
            batch_df = target_data["reconstruction"].iloc[start_idx:end_idx]  # Get the correct batch slice
            token_seq = self.df_encoder.forward(batch_df, return_tokens=True)
            total_loss += self.reconstruction_loss(token_seq, batch_df)  # Pass the batch_df instead of full df
                
        return total_loss

    def predict(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Makes predictions for all tasks"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model({"tabular": df})
            return {
                name: output for name, output in outputs.items()
                if self.task_configs[name].task_type not in 
                {TaskType.UMAP, TaskType.TSNE, TaskType.PCA}
            }

    def reduce_dimensionality(
        self,
        df: pd.DataFrame,
        target: str
    ) -> np.ndarray:
        """Applies dimensionality reduction to the data"""
        config = self.task_configs[target]
        if config.task_type not in {TaskType.UMAP, TaskType.TSNE, TaskType.PCA}:
            raise ValueError(f"Target {target} is not a dimensionality reduction task")
            
        self.model.eval()
        with torch.no_grad():
            features = self.encoder_module(df)
            joint = self.model.joint_network(features)
            latent = self.model.reparameterize(joint)
            
        # Retrieve the appropriate head from the non-differentiable heads
        if target not in self.dim_reduction_heads:
            raise KeyError(f"Dimensionality reduction head for {target} not found.")
        head = self.dim_reduction_heads[target]
        
        latent_np = latent.cpu().numpy()
        
        if config.task_type == TaskType.UMAP:
            head.fit(latent_np)
            return head.transform(latent_np)
        elif config.task_type == TaskType.TSNE:
            return head.fit_transform(latent_np)
        else:  # PCA
            head.fit(latent_np)
            return head.transform(latent_np)

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves the model to disk"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: Union[str, Path]) -> MultiTaskModel:
        """Loads a saved model from disk"""
        with open(filepath, "rb") as f:
            return pickle.load(f)

if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Sample data
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "income": [50, 60, 70, 80, 90],
        "gender": ["M", "F", "F", "M", "F"]
    })
    
    # Configure tasks - separate differentiable and non-differentiable tasks
    task_configs = {
        # Differentiable tasks
        "income": TaskConfig(TaskType.REGRESSION),
        "gender": TaskConfig(TaskType.CLASSIFICATION, num_classes=2),
        "readmission": TaskConfig(TaskType.BINOMIAL),
        "count": TaskConfig(TaskType.POISSON),
        "survival": TaskConfig(TaskType.TIME_TO_EVENT, time_bins=10),
        
        # Non-differentiable tasks (dimensionality reduction)
        "dim_reduction": TaskConfig(TaskType.UMAP, n_components=2),
        
        # Reconstruction task
        "reconstruction": TaskConfig(
            TaskType.RECONSTRUCTION,
            loss_weights={
                "numeric_loss_weight": 1.0,
                "categorical_loss_weight": 1.0,
                "missing_loss_weight": 1.0
            }
        )
    }
    
    # Create dummy target data
    target_data = {
        "income": torch.tensor([50000, 60000, 70000, 80000, 90000], dtype=torch.float32),
        "gender": torch.tensor([0, 1, 1, 0, 1], dtype=torch.long),
        "readmission": torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32),
        "count": torch.tensor([3, 2, 4, 5, 3], dtype=torch.float32),
        "survival": (
            torch.tensor([2, 3, 4, 3, 2], dtype=torch.long),  # times
            torch.tensor([1, 0, 1, 1, 0], dtype=torch.long)   # events
        ),
        "reconstruction": df  # Original data for reconstruction task
    }
    
    # Initialize model
    model = MultiTaskModel(
        numeric_cols=["age", "income"],
        categorical_cols=["gender"],
        task_configs=task_configs,
        embed_dim=32,
        transformer_layers=2,
        transformer_heads=4,
        dropout=0.1,
        use_missing_indicator=True,
        handle_unknown=True,
        joint_hidden_dim=64,
        variational=False,
        sigma=1.0,
        device=torch.device("cpu"),
        learning_rate=1e-3
    )
    
    # Fit encoder first
    print("Fitting encoder...")
    model.fit_encoder(df)
    
    # Train model in stages
    print("\nTraining regression and classification tasks...")
    model.fit(
        df,
        target_data,
        epochs=3,
        batch_size=2,
        targets_to_train=["income", "gender"],
        verbose=True
    )
    
    print("\nTraining additional tasks...")
    model.fit(
        df,
        target_data,
        epochs=2,
        batch_size=2,
        targets_to_train=["readmission", "count", "survival"],
        verbose=True
    )
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(df)
    for target, output in predictions.items():
        print(f"{target}:")
        print(output)
    
    # Apply dimensionality reduction
    print("\nApplying UMAP dimensionality reduction...")
    embedding = model.reduce_dimensionality(df, "dim_reduction")
    print("UMAP embedding:")
    print(embedding)
    
    # Save and load model
    print("\nSaving model...")
    model.save("multitask_model.pkl")
    
    print("Loading model...")
    loaded_model = MultiTaskModel.load("multitask_model.pkl")
    
    # Verify loaded model works
    print("\nVerifying loaded model...")
    new_predictions = loaded_model.predict(df)
    if all((predictions[k] == new_predictions[k]).all() for k in predictions):
        print("Loaded model produces identical predictions!")