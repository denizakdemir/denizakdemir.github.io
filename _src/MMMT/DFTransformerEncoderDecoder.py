import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Callable, Optional, Any

class DFTransformerEncoder(nn.Module):
    """
    Transformer-based encoder for tabular data with numeric and categorical features.

    Preprocessing steps:
      - **Numeric features:** Missing values are imputed (default: mean) and then scaled.
      - **Categorical features:** Missing values are replaced with "missing", then label encoded.
    
    Embedding:
      - Numeric features: Projected via a linear layer.
      - Categorical features: Embedded via an nn.Embedding layer.
    
    Transformer:
      - Stacks all feature embeddings into a token sequence, adds learned positional embeddings,
        and processes them with a transformer encoder.
      
    Aggregation:
      - Mean pooling is applied over the tokens unless token-level output is requested.
    """
    def __init__(self,
                 numeric_columns: List[str],
                 categorical_columns: List[str],
                 embed_dim: int = 32, 
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 use_missing_indicator: bool = False,
                 handle_unknown: bool = False,
                 device: Optional[torch.device] = None) -> None:
        super(DFTransformerEncoder, self).__init__()
        self.numeric_columns: List[str] = numeric_columns
        self.categorical_columns: List[str] = categorical_columns
        self.embed_dim: int = embed_dim
        self.num_layers: int = num_layers
        self.num_heads: int = num_heads
        self.dropout: float = dropout
        self.use_missing_indicator: bool = use_missing_indicator
        self.handle_unknown: bool = handle_unknown
        self.device: torch.device = device if device is not None else torch.device("cpu")
        
        # Containers for imputation values, scaling parameters, and label encoders (populated during fit)
        self.num_imputers: Dict[str, float] = {}
        self.num_scalers: Dict[str, tuple] = {}  # Stores (mean, std) for each numeric column
        self.cat_label_encoders: Dict[str, LabelEncoder] = {}
        self.cat_token2idx: Dict[str, Dict[str, int]] = {}  # For efficient token lookup

        # For numeric features, input dimension is 1 (value) plus 1 if using missing indicator.
        num_input_dim: int = 1 + int(use_missing_indicator)
        self.num_embeddings: nn.ModuleDict = nn.ModuleDict({
            col: nn.Linear(num_input_dim, embed_dim) for col in numeric_columns
        })
        self.cat_embeddings: nn.ModuleDict = nn.ModuleDict()  # Initialized in fit
        
        # Total number of tokens equals number of numeric plus categorical features.
        self.total_tokens: int = len(numeric_columns) + len(categorical_columns)
        # Learned positional embeddings: one vector per token.
        self.positional_embedding: nn.Parameter = nn.Parameter(
            torch.randn(self.total_tokens, embed_dim, device=self.device)
        )
        
        # Define transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=self.num_heads, 
            dropout=self.dropout,
            batch_first=True  # Batch-first mode.
        )
        self.transformer_encoder: nn.TransformerEncoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.fitted: bool = False

    def fit(self, df: pd.DataFrame, impute_func: Optional[Callable[[pd.Series], float]] = None) -> "DFTransformerEncoder":
        """
        Fits the encoder on the provided DataFrame.
        
        Args:
            df (pd.DataFrame): Training data.
            impute_func (Callable, optional): Function to compute imputation value (defaults to mean).
            
        Returns:
            DFTransformerEncoder: Fitted encoder.
        """
        expected_cols = set(self.numeric_columns + self.categorical_columns)
        if not expected_cols.issubset(set(df.columns)):
            missing_cols = expected_cols - set(df.columns)
            raise ValueError(f"DataFrame is missing expected columns: {missing_cols}")
        
        # Fit numeric imputers.
        impute_func = impute_func if impute_func is not None else (lambda s: s.mean())
        for col in self.numeric_columns:
            self.num_imputers[col] = impute_func(df[col])
            # Compute scaling parameters (mean and std) using non-missing values.
            col_values = df[col].dropna().astype(np.float32)
            scaling_mean = col_values.mean()
            scaling_std = col_values.std()
            if scaling_std == 0:
                scaling_std = 1.0  # Avoid division by zero.
            self.num_scalers[col] = (scaling_mean, scaling_std)
        
        # Fit label encoders and initialize embeddings for categorical features.
        for col in self.categorical_columns:
            col_data: pd.Series = df[col].fillna("missing").astype(str)
            le = LabelEncoder()
            le.fit(col_data)
            if self.handle_unknown:
                # Append an unknown token if handling unknowns.
                le.classes_ = np.concatenate([le.classes_, np.array(["<UNK>"])])
            self.cat_label_encoders[col] = le
            self.cat_token2idx[col] = {token: idx for idx, token in enumerate(le.classes_)}
            num_classes: int = len(le.classes_)
            self.cat_embeddings[col] = nn.Embedding(num_classes, self.embed_dim)
        
        self.fitted = True
        return self

    def forward(self, df: pd.DataFrame, return_tokens: bool = False) -> torch.Tensor:
        """
        Transforms the input DataFrame into a latent representation.
        
        Args:
            df (pd.DataFrame): Input data.
            return_tokens (bool): If True, returns the full token sequence (shape: (B, T, embed_dim));
                                  if False, returns a pooled representation (shape: (B, embed_dim)).
            
        Returns:
            torch.Tensor: Latent representation.
        """
        if not self.fitted:
            raise ValueError("Encoder has not been fitted. Call 'fit' with training data first.")

        batch_size: int = df.shape[0]
        tokens: List[torch.Tensor] = []

        # Process numeric features.
        for col in self.numeric_columns:
            # Create a mask of missing values.
            missing_mask: np.ndarray = df[col].isna().astype(np.float32).to_numpy()
            # Impute missing values.
            imputed_values: np.ndarray = df[col].fillna(self.num_imputers[col]).to_numpy(dtype=np.float32)
            # Retrieve scaling parameters.
            scaling_mean, scaling_std = self.num_scalers[col]
            # Scale the numeric values.
            scaled_values: np.ndarray = (imputed_values - scaling_mean) / scaling_std

            if self.use_missing_indicator:
                # If using a missing indicator, concatenate the scaled value and the missing mask.
                combined: np.ndarray = np.stack([scaled_values, missing_mask], axis=1)
                col_tensor: torch.Tensor = torch.from_numpy(combined).to(self.device)
            else:
                col_tensor = torch.from_numpy(scaled_values.reshape(-1, 1)).to(self.device)
            emb: torch.Tensor = self.num_embeddings[col](col_tensor)
            tokens.append(emb)

        # Process categorical features.
        for col in self.categorical_columns:
            col_values: np.ndarray = df[col].fillna("missing").astype(str).to_numpy()
            mapping: Dict[str, int] = self.cat_token2idx[col]
            unknown_index: Optional[int] = mapping.get("<UNK>") if self.handle_unknown else None
            encoded: List[int] = []
            for val in col_values:
                if val in mapping:
                    encoded.append(mapping[val])
                elif unknown_index is not None:
                    encoded.append(unknown_index)
                else:
                    raise ValueError(f"Unseen category '{val}' in column '{col}'. "
                                     "Set handle_unknown=True to enable unknown token handling.")
            col_tensor = torch.tensor(encoded, dtype=torch.long, device=self.device)
            emb = self.cat_embeddings[col](col_tensor)
            tokens.append(emb)
        
        # Stack tokens: shape (batch_size, total_tokens, embed_dim)
        token_seq: torch.Tensor = torch.stack(tokens, dim=1)
        # Add learned positional embeddings.
        token_seq = token_seq + self.positional_embedding.unsqueeze(0)
        
        # Process sequence with the transformer encoder.
        transformer_output: torch.Tensor = self.transformer_encoder(token_seq)
        
        if return_tokens:
            return transformer_output  # Return token-level representation for decoding.
        
        # Otherwise, aggregate via mean pooling.
        representation: torch.Tensor = transformer_output.mean(dim=1)
        return representation

    def fit_transform(self, df: pd.DataFrame, impute_func: Optional[Callable[[pd.Series], float]] = None) -> torch.Tensor:
        """
        Fits the encoder and transforms the input data.
        
        Args:
            df (pd.DataFrame): Training data.
            impute_func (Callable, optional): Custom imputation function. Defaults to mean.
            
        Returns:
            torch.Tensor: Latent representation.
        """
        self.fit(df, impute_func=impute_func)
        return self.forward(df)

class DFTransformerDecoder(nn.Module):
    """
    Decoder that reconstructs the original tabular data from the token-level representation.
    
    The decoder assumes that the token sequence is ordered so that the first
    |numeric_columns| tokens correspond to numeric features and the remaining tokens
    correspond to categorical features. It uses the saved information from the encoder (e.g.,
    label encoders, scaling parameters) to map back to the original DataFrame format.
    """
    def __init__(self, encoder: DFTransformerEncoder):
        """
        Initializes the decoder.
        
        Args:
            encoder (DFTransformerEncoder): A fitted encoder whose saved state will be used for decoding.
        """
        super(DFTransformerDecoder, self).__init__()
        self.encoder = encoder
        
        # Create decoder heads for numeric features.
        self.num_decoders = nn.ModuleDict()
        for col in encoder.numeric_columns:
            # Linear layer to decode the numeric value (in scaled space).
            self.num_decoders[col] = nn.Linear(encoder.embed_dim, 1)
            if encoder.use_missing_indicator:
                # An extra head to predict missing indicator (using a sigmoid later).
                self.num_decoders[col + "_missing"] = nn.Linear(encoder.embed_dim, 1)
        
        # Create decoder heads for categorical features.
        self.cat_decoders = nn.ModuleDict()
        for col in encoder.categorical_columns:
            num_classes = len(encoder.cat_label_encoders[col].classes_)
            self.cat_decoders[col] = nn.Linear(encoder.embed_dim, num_classes)
            
    def forward(self, token_seq: torch.Tensor) -> Dict[str, Any]:
        """
        Decodes the token sequence into predictions for numeric and categorical features.
        
        Args:
            token_seq (torch.Tensor): Token sequence of shape (batch_size, total_tokens, embed_dim)
            
        Returns:
            Dict[str, Any]: A dictionary with keys 'numeric' and 'categorical' containing decoded outputs.
        """
        batch_size = token_seq.size(0)
        num_cols = self.encoder.numeric_columns
        cat_cols = self.encoder.categorical_columns
        
        numeric_outputs = {}
        categorical_outputs = {}
        
        # Decode numeric features.
        for i, col in enumerate(num_cols):
            token = token_seq[:, i, :]  # Token for this numeric column.
            val = self.num_decoders[col](token)  # Predicted (scaled) value.
            numeric_outputs[col] = val.squeeze(1)
            if self.encoder.use_missing_indicator:
                missing_logit = self.num_decoders[col + "_missing"](token)
                missing_prob = torch.sigmoid(missing_logit)
                numeric_outputs[col + "_missing"] = missing_prob.squeeze(1)
                
        # Decode categorical features.
        for j, col in enumerate(cat_cols):
            token = token_seq[:, len(num_cols) + j, :]
            logits = self.cat_decoders[col](token)
            probs = F.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1)
            categorical_outputs[col] = pred_idx  # These indices will be mapped back to labels.
        
        return {'numeric': numeric_outputs, 'categorical': categorical_outputs}
    
    def decode_to_dataframe(self, token_seq: torch.Tensor) -> pd.DataFrame:
        """
        Decodes the token sequence and returns a pandas DataFrame with reconstructed data.
        
        Args:
            token_seq (torch.Tensor): Token sequence from the encoder.
            
        Returns:
            pd.DataFrame: Reconstructed DataFrame.
        """
        decoded = self.forward(token_seq)
        numeric_data = {}
        for col, vals in decoded['numeric'].items():
            # Skip missing indicator outputs here.
            if col.endswith("_missing"):
                continue
            # Retrieve the scaling parameters.
            scaling_mean, scaling_std = self.encoder.num_scalers[col]
            # Inverse transform: unscale the predicted value.
            unscaled_vals = vals.cpu().detach().numpy() * scaling_std + scaling_mean
            if self.encoder.use_missing_indicator:
                missing_col = col + "_missing"
                # Use 0.5 as the threshold for missingness.
                missing_flags = (decoded['numeric'][missing_col] > 0.5).cpu().numpy()
                unscaled_vals[missing_flags] = np.nan
            numeric_data[col] = unscaled_vals
        
        categorical_data = {}
        for col, indices in decoded['categorical'].items():
            idx_array = indices.cpu().detach().numpy()
            le = self.encoder.cat_label_encoders[col]
            # Inverse transform to get the original labels.
            decoded_labels = le.inverse_transform(idx_array)
            categorical_data[col] = decoded_labels
        
        # Combine numeric and categorical data into one DataFrame.
        data = {**numeric_data, **categorical_data}
        df_decoded = pd.DataFrame(data)
        return df_decoded

# --- Example Usage ---
def main() -> None:
    # Sample training data.
    data_train: Dict[str, List[Any]] = {
        'age': [25, np.nan, 35, 45],
        'income': [50000, 60000, np.nan, 80000],
        'gender': ['M', 'F', np.nan, 'F'],
        'occupation': ['engineer', 'doctor', 'lawyer', np.nan]
    }
    df_train: pd.DataFrame = pd.DataFrame(data_train)

    # Sample test data.
    data_test: Dict[str, List[Any]] = {
        'age': [30, 40],
        'income': [55000, np.nan],
        'gender': ['F', 'M'],
        'occupation': ['doctor', 'scientist']
    }
    df_test: pd.DataFrame = pd.DataFrame(data_test)

    # Define feature columns.
    numeric_cols: List[str] = ['age', 'income']
    categorical_cols: List[str] = ['gender', 'occupation']

    # Initialize the encoder with missing indicator and unknown token handling.
    encoder: DFTransformerEncoder = DFTransformerEncoder(
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

    # Fit the encoder on training data.
    encoder.fit(df_train)
    
    # When decoding, we require the full token sequence. Set return_tokens=True.
    token_seq: torch.Tensor = encoder.forward(df_test, return_tokens=True)
    
    # Initialize the decoder with the fitted encoder.
    decoder: DFTransformerDecoder = DFTransformerDecoder(encoder)
    
    # Decode the token sequence into a reconstructed DataFrame.
    df_decoded: pd.DataFrame = decoder.decode_to_dataframe(token_seq)
    print("Reconstructed DataFrame:")
    print(df_decoded)

if __name__ == "__main__":
    main()
