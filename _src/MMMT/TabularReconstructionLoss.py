import os
import urllib.request
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, classification_report

# Add _src to sys.path so that our custom encoder/decoder can be imported.
sys.path.append(str(Path.cwd()))
from _src.MMMT.DFTransformerEncoderDecoder import DFTransformerEncoder, DFTransformerDecoder


# --- Reconstruction Loss as defined earlier ---
class TabularReconstructionLoss(nn.Module):
    r"""
    Generic reconstruction loss for tabular data that combines numeric and categorical losses.

    For numeric columns, the loss is computed as:
      - Mean squared error (MSE) between the predicted scaled value and the ground truth scaled value,
        computed only on non-missing entries.
      - If a missing indicator is used, a binary cross entropy (BCE) loss is computed on the missingness.

    For categorical columns, the loss is the cross-entropy loss between predicted logits and
    the ground truth label indices.

    The total loss is a weighted sum of the numeric and categorical losses.
    """
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 numeric_loss_weight: float = 1.0,
                 categorical_loss_weight: float = 1.0,
                 missing_loss_weight: float = 1.0) -> None:
        super(TabularReconstructionLoss, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.numeric_loss_weight = numeric_loss_weight
        self.categorical_loss_weight = categorical_loss_weight
        self.missing_loss_weight = missing_loss_weight

        # Loss functions used internally.
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')
    
    def forward(self, token_seq: torch.Tensor, df: pd.DataFrame) -> torch.Tensor:
        r"""
        Computes the reconstruction loss.
        """
        device = token_seq.device
        total_numeric_loss = 0.0
        total_categorical_loss = 0.0

        # ------------------------------
        # Process numeric features.
        # ------------------------------
        for idx, col in enumerate(self.encoder.numeric_columns):
            # Extract token corresponding to the numeric column.
            token = token_seq[:, idx, :]  # shape: (B, embed_dim)
            
            # Predict the (scaled) numeric value.
            pred_value = self.decoder.num_decoders[col](token)  # shape: (B, 1)
            pred_value = pred_value.squeeze(1)  # shape: (B,)
            
            # Retrieve ground truth values.
            gt_values_np = df[col].to_numpy().astype(np.float32)
            gt_tensor = torch.tensor(gt_values_np, dtype=torch.float32, device=device)
            non_missing_mask = ~torch.isnan(gt_tensor)
            
            if non_missing_mask.sum() > 0:
                scaling_mean, scaling_std = self.encoder.num_scalers[col]
                gt_scaled = (gt_tensor[non_missing_mask] - scaling_mean) / scaling_std
                mse = self.mse_loss(pred_value[non_missing_mask], gt_scaled)
            else:
                mse = torch.tensor(0.0, device=device)
            
            total_numeric_loss += mse

            if self.encoder.use_missing_indicator:
                missing_key = col + "_missing"
                pred_missing_logit = self.decoder.num_decoders[missing_key](token)  # shape: (B, 1)
                pred_missing_prob = torch.sigmoid(pred_missing_logit).squeeze(1)  # shape: (B,)
                gt_missing = torch.tensor(np.isnan(gt_values_np).astype(np.float32),
                                          dtype=torch.float32, device=device)
                bce = self.bce_loss(pred_missing_prob, gt_missing)
                total_numeric_loss += self.missing_loss_weight * bce

        # ------------------------------
        # Process categorical features.
        # ------------------------------
        num_numeric = len(self.encoder.numeric_columns)
        for j, col in enumerate(self.encoder.categorical_columns):
            token = token_seq[:, num_numeric + j, :]  # shape: (B, embed_dim)
            logits = self.decoder.cat_decoders[col](token)  # shape: (B, num_classes)
            
            gt_series = df[col].fillna("missing").astype(str)
            mapping = self.encoder.cat_token2idx[col]
            gt_indices = [mapping[label] if label in mapping else mapping.get("<UNK>", 0)
                          for label in gt_series.to_numpy()]
            gt_indices_tensor = torch.tensor(gt_indices, dtype=torch.long, device=device)
            
            ce = F.cross_entropy(logits, gt_indices_tensor)
            total_categorical_loss += ce

        total_loss = (self.numeric_loss_weight * total_numeric_loss +
                      self.categorical_loss_weight * total_categorical_loss)
        return total_loss


# --- Anomaly Detection Example using the Arrhythmia Dataset ---
def anomaly_detection_example() -> None:
    # Step 1: Download the Arrhythmia dataset to the data folder.
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    dataset_path = data_dir / "arrhythmia.data"
    
    if not dataset_path.exists():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data"
        print("Downloading Arrhythmia dataset...")
        urllib.request.urlretrieve(url, dataset_path)
        print("Download complete.")
    
    # Step 2: Load the dataset.
    # Note: The dataset has no header; missing values are marked as "?".
    df = pd.read_csv(dataset_path, header=None, na_values="?")
    num_columns = df.shape[1]
    # Assume the last column is the target (class label).
    feature_cols = [f"feature_{i}" for i in range(num_columns - 1)]
    target_col = "target"
    df.columns = feature_cols + [target_col]
    
    # For anomaly detection, assume label '1' is normal and all others are anomalies.
    df["anomaly"] = (df[target_col] != 1).astype(int)
    
    # Train autoencoder only on normal samples.
    df_features = df[feature_cols]
    normal_df = df_features[df["anomaly"] == 0].reset_index(drop=True)
    test_df = df_features.reset_index(drop=True)
    test_labels = df["anomaly"].reset_index(drop=True)
    
    # Step 3: Define numeric/categorical columns.
    # (The Arrhythmia dataset is entirely numeric.)
    numeric_cols = feature_cols
    categorical_cols = []  # no categorical features in this dataset
    
    # Step 4: Initialize and fit the encoder on normal data.
    encoder = DFTransformerEncoder(
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        use_missing_indicator=True,
        handle_unknown=True,
        device=torch.device("cpu")
    )
    encoder.fit(normal_df)
    
    # Step 5: Initialize the decoder and reconstruction loss module.
    decoder = DFTransformerDecoder(encoder)
    loss_module = TabularReconstructionLoss(encoder, decoder,
                                            numeric_loss_weight=1.0,
                                            categorical_loss_weight=1.0,
                                            missing_loss_weight=1.0)
    
    # Step 6: Train the autoencoder on normal data.
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    num_epochs = 50
    batch_size = 32
    num_train = len(normal_df)
    train_losses = []  # To record training loss for each epoch
    
    print("Training autoencoder on normal data...")
    encoder.train()
    decoder.train()
    for epoch in range(num_epochs):
        permutation = np.random.permutation(num_train)
        epoch_loss = 0.0
        for i in range(0, num_train, batch_size):
            indices = permutation[i:i + batch_size]
            batch_df = normal_df.iloc[indices]
            token_seq = encoder.forward(batch_df, return_tokens=True)
            loss = loss_module(token_seq, batch_df)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(indices)
        epoch_loss /= num_train
        train_losses.append(epoch_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d}: Loss = {epoch_loss:.4f}")
    
    # Plot training loss curve.
    plt.figure(figsize=(8, 5))
    plt.plot(range(num_epochs), train_losses, marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.title("Training Loss over Epochs")
    plt.grid(True)
    plt.show()
    
    # Step 7: Compute per-sample reconstruction errors on the test set.
    def compute_reconstruction_errors(df_input: pd.DataFrame) -> np.ndarray:
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            token_seq = encoder.forward(df_input, return_tokens=True)  # shape: (B, T, embed_dim)
            B = token_seq.size(0)
            device = token_seq.device
            sample_losses = torch.zeros(B, device=device)
            
            # Process numeric features.
            for idx, col in enumerate(encoder.numeric_columns):
                token = token_seq[:, idx, :]
                pred_value = decoder.num_decoders[col](token).squeeze(1)  # (B,)
                
                gt_values_np = df_input[col].to_numpy().astype(np.float32)
                gt_tensor = torch.tensor(gt_values_np, dtype=torch.float32, device=device)
                
                # Create a mask for non-missing values.
                non_missing_mask = ~torch.isnan(gt_tensor)
                
                if non_missing_mask.any():
                    scaling_mean, scaling_std = encoder.num_scalers[col]
                    gt_scaled = (gt_tensor[non_missing_mask] - scaling_mean) / scaling_std
                    se = torch.zeros_like(pred_value)
                    se[non_missing_mask] = (pred_value[non_missing_mask] - gt_scaled) ** 2
                    sample_losses += se
                
                if encoder.use_missing_indicator:
                    missing_key = col + "_missing"
                    pred_missing_logit = decoder.num_decoders[missing_key](token).squeeze(1)
                    pred_missing_prob = torch.sigmoid(pred_missing_logit)
                    gt_missing = torch.tensor(np.isnan(gt_values_np).astype(np.float32),
                                              dtype=torch.float32, device=device)
                    bce = F.binary_cross_entropy(pred_missing_prob, gt_missing, reduction='none')
                    sample_losses += loss_module.missing_loss_weight * bce
            
            # Process categorical features (if any).
            num_numeric = len(encoder.numeric_columns)
            for j, col in enumerate(encoder.categorical_columns):
                token = token_seq[:, num_numeric + j, :]
                logits = decoder.cat_decoders[col](token)
                gt_series = df_input[col].fillna("missing").astype(str)
                mapping = encoder.cat_token2idx[col]
                gt_indices = [mapping[label] if label in mapping else mapping.get("<UNK>", 0)
                              for label in gt_series.to_numpy()]
                gt_indices_tensor = torch.tensor(gt_indices, dtype=torch.long, device=device)
                ce = F.cross_entropy(logits, gt_indices_tensor, reduction='none')
                sample_losses += ce
            
            return sample_losses.cpu().numpy()

    test_errors = compute_reconstruction_errors(test_df)
    
    # Step 8: Evaluate anomaly detection performance.
    auc = roc_auc_score(test_labels, test_errors)
    print("\nDetailed Evaluation Report:")
    print(f"Anomaly Detection ROC AUC: {auc:.4f}")

    # Compute classification report.
    threshold = np.percentile(test_errors, 25)
    pred_labels = (test_errors > threshold).astype(int)
    print("\nClassification Report:")
    print(classification_report(test_labels, pred_labels, target_names=["Normal", "Anomaly"]))

    
    # Compute summary statistics.
    normal_errors = test_errors[test_labels == 0]
    anomaly_errors = test_errors[test_labels == 1]
    
    stats = {
        "Normal": {
            "Mean": np.mean(normal_errors),
            "Median": np.median(normal_errors),
            "Std": np.std(normal_errors)
        },
        "Anomaly": {
            "Mean": np.mean(anomaly_errors),
            "Median": np.median(anomaly_errors),
            "Std": np.std(anomaly_errors)
        }
    }
    
    for label, metrics in stats.items():
        print(f"\n{label} Samples:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Step 9: Plot side-by-side boxplots of reconstruction errors.
    plt.figure(figsize=(8, 6))
    data_to_plot = [normal_errors, anomaly_errors]
    plt.boxplot(data_to_plot, tick_labels=["Normal", "Anomaly"], patch_artist=True,
                boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red'))
    plt.ylabel("Reconstruction Error")
    plt.title("Boxplot of Reconstruction Errors by Class")
    plt.grid(axis='y')
    plt.show()


if __name__ == "__main__":
    anomaly_detection_example()
