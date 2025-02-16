from __future__ import annotations
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Tuple

# Additional imports for splitting and evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Import our custom modules (assumed to be available in the same directory)
from DFTransformerEncoderDecoder import DFTransformerEncoder, DFTransformerDecoder
from MMMT import MultiModalMultiTaskModel
import MLheads
from TabularReconstructionLoss import TabularReconstructionLoss

# =============================================================================
# Task and Model Definitions (consistent with MLheads.py)
# =============================================================================

class TaskType(Enum):
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
    task_type: TaskType
    num_classes: Optional[int] = None
    time_bins: Optional[int] = None
    num_events: Optional[int] = 1
    r: Optional[float] = 1.0  # For negative binomial
    n_components: Optional[int] = 2  # For dimensionality reduction
    loss_weights: Optional[Dict[str, float]] = None  # For reconstruction
    trials: Optional[int] = None  # For binomial regression head (counts of successes)


class TabularEncoder(nn.Module):
    """Wraps DFTransformerEncoder as an nn.Module for use in MMMT."""
    def __init__(self, df_encoder: DFTransformerEncoder):
        super().__init__()
        self.df_encoder = df_encoder

    def forward(self, x: pd.DataFrame) -> torch.Tensor:
        # The encoder returns a tensor representation of the DataFrame.
        return self.df_encoder.forward(x, return_tokens=False)

class MultiTaskModel:
    """
    A unified model for multi-modal, multi-task learning.
    Supports tabular data with supervised tasks (regression, classification)
    as well as unsupervised tasks like UMAP-based dimensionality reduction
    and reconstruction.
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
        
        # Initialize the DFTransformer encoder for tabular data.
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
        
        # Create task heads for differentiable tasks.
        self.task_heads = self._create_task_heads(embed_dim * 2)  # Joint representation is 2×embed_dim
        
        # Joint network for combining representations.
        self.joint_network = nn.Sequential(
            nn.Linear(embed_dim, joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(joint_hidden_dim, embed_dim * 2)
        )
        
        # Initialize the multi-modal multi-task model.
        self.model = MultiModalMultiTaskModel(
            encoders={"tabular": self.encoder_module},
            task_heads=self.task_heads,
            joint_network=self.joint_network,
            variational=variational,
            sigma=sigma
        ).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.reconstruction_loss = None  # Will be set after encoder fitting

    def _is_differentiable_task(self, task_type: TaskType) -> bool:
        return task_type not in {TaskType.UMAP, TaskType.TSNE, TaskType.PCA, TaskType.RECONSTRUCTION}

    def _create_task_heads(self, input_dim: int) -> Dict[str, nn.Module]:
        heads = {}
        self.dim_reduction_heads = {}  # For non-differentiable tasks

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
                # Ensure the trials parameter is provided for the binomial task.
                if config.trials is None:
                    raise ValueError("For TaskType.BINOMIAL, 'trials' must be specified in TaskConfig")
                heads[name] = MLheads.BinomialRegressionHead(input_dim=input_dim, trials=config.trials)
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
        """Fits the encoder (which may, for example, learn embeddings for categorical features)
        and sets up reconstruction loss if configured."""
        self.df_encoder.fit(df)
        self.reconstruction_loss = self._setup_reconstruction_loss()

    def fit(
        self,
        df: pd.DataFrame,
        target_data: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
        epochs: int = 10,
        batch_size: int = 32,
        targets_to_train: Optional[List[str]] = None,
        verbose: bool = True,
        val_df: Optional[pd.DataFrame] = None,
        val_target_data: Optional[Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]] = None
    ) -> Tuple[List[float], Optional[List[float]]]:
        """
        Trains the multi-task model on the provided training data. Optionally, if
        validation data is supplied, computes and tracks the validation loss after each epoch.

        Parameters:
            df (pd.DataFrame): The training data.
            target_data (dict): Dictionary mapping task names to target data for training.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size used during training.
            targets_to_train (List[str], optional): List of task names to train. Defaults to all differentiable tasks.
            verbose (bool): If True, prints loss information during training.
            val_df (pd.DataFrame, optional): The validation data.
            val_target_data (dict, optional): Dictionary mapping task names to target data for validation.

        Returns:
            Tuple containing:
                - List of average training losses per epoch.
                - List of average validation losses per epoch (if validation data is provided; otherwise, None).
        """
        self.model.train()
        num_samples = len(df)
        epoch_train_losses = []
        epoch_val_losses = [] if val_df is not None and val_target_data is not None else None
        active_targets = targets_to_train or list(self.model.task_heads.keys())

        for epoch in range(epochs):
            epoch_loss = 0.0
            # Iterate over mini-batches from the training data.
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_df = df.iloc[start_idx:end_idx]

                # Forward pass through the model.
                outputs = self.model({"tabular": batch_df})
                
                # Compute the combined loss for the current batch.
                batch_loss = self._compute_batch_loss(
                    outputs, target_data, start_idx, end_idx, active_targets
                )
                
                # Backpropagation and optimizer step.
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                
                epoch_loss += batch_loss.item()

            # Compute the average training loss for this epoch.
            avg_epoch_loss = epoch_loss / ((num_samples + batch_size - 1) // batch_size)
            epoch_train_losses.append(avg_epoch_loss)

            # If validation data is provided, evaluate the model on the validation set.
            if val_df is not None and val_target_data is not None:
                self.model.eval()  # Set the model to evaluation mode.
                val_loss = 0.0
                num_val_samples = len(val_df)
                with torch.no_grad():
                    for start_idx in range(0, num_val_samples, batch_size):
                        end_idx = min(start_idx + batch_size, num_val_samples)
                        batch_val_df = val_df.iloc[start_idx:end_idx]
                        
                        # Forward pass for validation data.
                        outputs_val = self.model({"tabular": batch_val_df})
                        
                        # Compute loss for the validation batch.
                        batch_val_loss = self._compute_batch_loss(
                            outputs_val, val_target_data, start_idx, end_idx, active_targets
                        )
                        val_loss += batch_val_loss.item()
                avg_val_loss = val_loss / ((num_val_samples + batch_size - 1) // batch_size)
                epoch_val_losses.append(avg_val_loss)
                self.model.train()  # Revert back to training mode.

            # Print progress if verbose is enabled.
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                if epoch_val_losses is not None:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_epoch_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f}")

        return epoch_train_losses, epoch_val_losses

    def _compute_batch_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target_data: Dict[str, Any],
        start_idx: int,
        end_idx: int,
        active_targets: List[str]
    ) -> torch.Tensor:
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
            
        # Add reconstruction loss if applicable.
        if (self.reconstruction_loss is not None and 
            ("reconstruction" in active_targets or "reconstruction" in self.task_configs)):
            batch_df = target_data["reconstruction"].iloc[start_idx:end_idx]
            token_seq = self.df_encoder.forward(batch_df, return_tokens=True)
            total_loss += self.reconstruction_loss(token_seq, batch_df)
                
        return total_loss

    def predict(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Generates predictions for all differentiable tasks."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model({"tabular": df})
            return {
                name: output for name, output in outputs.items()
                if self.task_configs[name].task_type not in {TaskType.UMAP, TaskType.TSNE, TaskType.PCA}
            }

    def reduce_dimensionality(
        self,
        df: pd.DataFrame,
        target: str
    ) -> np.ndarray:
        """Applies a dimensionality reduction head (e.g. UMAP) to the latent space."""
        config = self.task_configs[target]
        if config.task_type not in {TaskType.UMAP, TaskType.TSNE, TaskType.PCA}:
            raise ValueError(f"Target {target} is not a dimensionality reduction task")
            
        self.model.eval()
        with torch.no_grad():
            features = self.encoder_module(df)
            joint = self.model.joint_network(features)
            latent = self.model.reparameterize(joint)
            
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
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: Union[str, Path]) -> MultiTaskModel:
        with open(filepath, "rb") as f:
            return pickle.load(f)

# =============================================================================
# Real Data Example with Training/Test Split and Plots
# =============================================================================

if __name__ == "__main__":
    # Load Titanic dataset using Seaborn.
    df = sns.load_dataset("titanic")
    print(df.columns)
    # Select relevant columns and drop rows with missing values.
    df = df[["age", "fare", "sex", "class", "adult_male", "deck", "embarked", "survived"]].reset_index(drop=True)
    # Ensure that categorical columns are strings.
    df["sex"] = df["sex"].astype(str)
    df["embarked"] = df["embarked"].astype(str)
    df["class"] = df["class"].astype(str)
    df["deck"] = df["deck"].astype(str)
    df["adult_male"] = df["adult_male"].astype(str)

    # standardize the fare column
    df["fare"] = (df["fare"] - df["fare"].mean()) / df["fare"].std()
    
    # Split the data into 80% train and 20% test.
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    # split train into train and validation
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)
    # Prepare target data for training.
    target_data_train = {
        "fare": torch.tensor(df_train["fare"].values, dtype=torch.float32),
        "survived": torch.tensor(df_train["survived"].values, dtype=torch.long),
        "reconstruction": df_train  # Reconstruction uses the full training DataFrame.
    }
    # validation data
    target_data_val = {
        "fare": torch.tensor(df_val["fare"].values, dtype=torch.float32),
        "survived": torch.tensor(df_val["survived"].values, dtype=torch.long),
        "reconstruction": df_val  # Reconstruction uses the full training DataFrame.
    }
    
    # Prepare target data for testing (no reconstruction target here).
    target_data_test = {
        "fare": torch.tensor(df_test["fare"].values, dtype=torch.float32),
        "survived": torch.tensor(df_test["survived"].values, dtype=torch.long)
    }
    
    # Define task configurations:
    # - "fare": regression task (uses MSELoss by MLheads.RegressionHead).
    # - "survived": classification task (uses CrossEntropyLoss by MLheads.ClassificationHead).
    # - "umap": for dimensionality reduction (post-hoc analysis).
    # - "reconstruction": to learn a robust joint representation.
    task_configs = {
        "fare": TaskConfig(TaskType.REGRESSION),
        "survived": TaskConfig(TaskType.CLASSIFICATION, num_classes=2),
        "umap": TaskConfig(TaskType.UMAP, n_components=2),
        "reconstruction": TaskConfig(
            TaskType.RECONSTRUCTION,
            loss_weights={
                "numeric_loss_weight": 1.0,
                "categorical_loss_weight": 1.0,
                "missing_loss_weight": 1.0
            }
        )
    }
    
    # Specify numeric and categorical columns.
    numeric_cols = ["age", "fare"]
    categorical_cols = ["sex", "embarked", "class", "adult_male", "deck"]
    
    # Initialize the multi-task model.
    model = MultiTaskModel(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        task_configs=task_configs,
        embed_dim=32,
        transformer_layers=2,
        transformer_heads=4,
        dropout=0.1,
        use_missing_indicator=True,
        handle_unknown=True,
        joint_hidden_dim=64,
        variational=True,
        sigma=1.0,
        device=torch.device("cpu"),
        learning_rate=1e-4
    )
    
    # Fit the encoder on the training data.
    print("Fitting encoder on training data...")
    model.fit_encoder(df_train)
    
    # Train the model on the differentiable tasks ("fare" and "survived").
    print("Training model on training data...")
    num_epochs = 500
    # validation data
    train_losses, val_losses = model.fit(
        df_train,
        target_data_train,
        val_df=df_val,
        val_target_data=target_data_val,
        epochs=num_epochs,
        batch_size=64,
        targets_to_train=["fare", "survived"],
        verbose=True
    )
    
    # -------------------------------------------------------------------------
    # Plot the training loss curve.
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-')
    if val_losses is not None:
        plt.plot(range(1, num_epochs + 1), val_losses, marker='o', linestyle='-', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Curve")
    if val_losses is not None:
        plt.legend(["Train Loss", "Validation Loss"])
    plt.grid(True)
    plt.show()
    
    num_epochs = 500
    model.lr = 1e-2
    train_losses, val_losses = model.fit(
        df_train,
        target_data_train,
        val_df=df_val,
        val_target_data=target_data_val,
        epochs=num_epochs,
        batch_size=64,
        targets_to_train=["survived"],
        verbose=True
    )

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-')
    if val_losses is not None:
        plt.plot(range(1, num_epochs + 1), val_losses, marker='o', linestyle='-', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Curve")
    if val_losses is not None:
        plt.legend(["Train Loss", "Validation Loss"])
    plt.grid(True)
    plt.show()
    # -------------------------------------------------------------------------
    # Evaluate on the Test Set.
    model.model.eval()
    with torch.no_grad():
        predictions_test = model.predict(df_test)
    
    # --- Regression Evaluation (Fare) ---
    # Assume that the regression head outputs a tensor of shape (n,) or (n, 1)
    fare_pred = predictions_test["fare"].squeeze()
    fare_true = target_data_test["fare"]
    rmse = torch.sqrt(torch.mean((fare_pred - fare_true) ** 2)).item()
    print(f"Test RMSE for fare regression: {rmse:.4f}")
    
    # Plot True vs Predicted Fare.
    plt.figure(figsize=(8, 5))
    plt.scatter(fare_true.numpy(), fare_pred.numpy(), c='blue', alpha=0.6)
    plt.plot([fare_true.min(), fare_true.max()], [fare_true.min(), fare_true.max()], 'r--')
    plt.xlabel("True Fare")
    plt.ylabel("Predicted Fare")
    plt.title("Fare Regression: True vs Predicted")
    plt.grid(True)
    plt.show()
    
    # --- Classification Evaluation (Survived) ---
    survived_logits = predictions_test["survived"]
    # ClassificationHead outputs logits of shape (n, num_classes)
    survived_pred = survived_logits.argmax(dim=1)
    survived_true = target_data_test["survived"]
    accuracy = (survived_pred == survived_true).float().mean().item()
    print(f"Test Accuracy for survival classification: {accuracy:.4f}")
    
    # Compute and plot the confusion matrix.
    cm = confusion_matrix(survived_true.numpy(), survived_pred.numpy())
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Not Survived", "Survived"],
                yticklabels=["Not Survived", "Survived"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix for Survival Classification")
    plt.show()
    
    # -------------------------------------------------------------------------
    # UMAP Embedding Visualization.
    print("Computing UMAP embedding for test data...")
    umap_embedding = model.reduce_dimensionality(df_test, "umap")
    plt.figure(figsize=(8, 6))
    # Color the points by the survival outcome.
    survived_test = target_data_test["survived"].numpy()
    scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1],
                          c=survived_test, cmap="coolwarm", alpha=0.7)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.title("UMAP Embedding of Test Data")
    plt.colorbar(scatter, ticks=[0, 1], label="Survival")
    plt.show()
    
    # -------------------------------------------------------------------------
    # Save and reload the model (optional).
    print("Saving model to 'titanic_multitask_model.pkl'...")
    model.save("titanic_multitask_model.pkl")
    print("Loading model from 'titanic_multitask_model.pkl'...")
    loaded_model = MultiTaskModel.load("titanic_multitask_model.pkl")
    
    # Verify that the loaded model produces similar predictions.
    with torch.no_grad():
        new_predictions = loaded_model.predict(df_test)
    for task, output in new_predictions.items():
        print(f"Task: {task}, output shape: {output.shape}")

    # compare the accuracy of xgboost on the same data for comparison
    import xgboost as xgb
    from sklearn.metrics import accuracy_score
    # prepare the data
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()
    df_train
    # encode the categorical columns
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in categorical_cols:
        df_train[col] = le.fit_transform(df_train[col])
        df_val[col] = le.transform(df_val[col])
        df_test[col] = le.transform(df_test[col])

    # prepare the data
    X_train = df_train.drop(columns=["survived"])
    y_train = df_train["survived"]
    X_val = df_val.drop(columns=["survived"])
    y_val = df_val["survived"]
    X_test = df_test.drop(columns=["survived"])
    y_test = df_test["survived"]

    # train the xgboost model
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    # make predictions
    y_pred = model.predict(X_test)
    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    # compare the accuracy of the two models
    print(f"Test Accuracy for survival classification: {accuracy:.4f}")


    from sklearn.ensemble import BaggingClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    # Define grid search parameters for the decision tree base estimator.
    params = [{'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}]

    # Initialize the BaggingClassifier with GridSearchCV wrapping a DecisionTreeClassifier.
    b_clf = BaggingClassifier(
        GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=3, verbose=0),
        n_estimators=100,
        max_samples=100,
        bootstrap=True,
        n_jobs=-1
    )

    # Fit the bagging classifier on the training data.
    b_clf.fit(X_train, y_train)

    # Predict the test set labels.
    y_pred = b_clf.predict(X_test)

    # Compute and print the accuracy score.
    print(accuracy_score(y_test, y_pred))
