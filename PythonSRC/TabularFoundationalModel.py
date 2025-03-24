#!/usr/bin/env python
"""
A realistic tabular autoencoder that integrates per–variable metadata 
and an overall dataset text description. This extended example demonstrates
training on two separate datasets that are related but not identical,
and it has been adjusted to handle missingness (both structural/informative
and random) in categorical and numeric variables.

Key components:
1. A pretrained BERT encoder for generating fixed–length metadata embeddings.
2. MLP modules (MLPNum and MLPCat) to fuse a cell’s raw value, its missingness flag,
   and its metadata.
3. A transformer encoder (with sinusoidal positional encoding) that processes 
   a token sequence comprising a global overall–description token followed by cell tokens.
4. Decoding modules (a shared numeric decoder, per–column categorical decoders, and a 
   missingness decoder) to reconstruct original inputs and predict missingness.
5. A self–supervised training loop that applies MSE loss for numeric features, cross–entropy loss 
   for categorical features (only on observed values), and binary cross–entropy loss for missingness.
6. Demonstrates batch-based training with DataLoader, GPU support, and precomputation of metadata embeddings.
7. Dummy CSV files for demonstration, with datasets sharing some columns (e.g. age, gender)
   while differing on others (income vs. salary).

Author: [Your Name]
Date: [Today's Date]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import math
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split

# Set device for GPU usage if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Pretrained BERT for Metadata Encoding
# ------------------------------
class PretrainedBertEncoder:
    """
    Wraps a pretrained BERT model and tokenizer to produce fixed–length embeddings
    for text descriptions. The BERT [CLS] token representation is projected to a lower dimension.
    BERT parameters are frozen for efficiency and caching is recommended.
    """
    def __init__(self, embedding_dim, pretrained_model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        # Freeze BERT parameters to avoid costly backpropagation.
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert_hidden_size = self.bert.config.hidden_size  # typically 768
        self.projection = nn.Linear(self.bert_hidden_size, embedding_dim).to(device)
        self.embedding_dim = embedding_dim

    def encode(self, text):
        """
        Encodes input text into a fixed–length embedding.
        
        Args:
            text (str): Text description to encode.
            
        Returns:
            torch.Tensor: Embedding of size [embedding_dim].
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = self.bert(**inputs)
        cls_output = outputs.last_hidden_state[:, 0, :]  # shape: [1, bert_hidden_size]
        embedding = self.projection(cls_output)  # shape: [1, embedding_dim]
        return embedding.squeeze(0)  # shape: [embedding_dim]

# ------------------------------
# MLP Modules for Cell Encoding
# ------------------------------
class MLPNum(nn.Module):
    """
    MLP for encoding numeric features.
    Expects an input vector comprising [value, missing_flag, metadata_embedding].
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPNum, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class MLPCat(nn.Module):
    """
    MLP for encoding categorical features.
    Expects an input vector comprising [value, missing_flag, metadata_embedding].
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPCat, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------
# Sinusoidal Positional Encoding
# ------------------------------
def positional_encoding(seq_len, d_model):
    """
    Computes sinusoidal positional encodings.
    
    Args:
        seq_len (int): Sequence length.
        d_model (int): Embedding dimension.
        
    Returns:
        torch.Tensor: Positional encodings with shape [seq_len, 1, d_model].
    """
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(1).to(device)

# ------------------------------
# Transformer Encoder with Positional Encoding
# ------------------------------
class TransformerEncoderModel(nn.Module):
    """
    A transformer encoder that adds sinusoidal positional encodings 
    to a token sequence and outputs updated embeddings.
    """
    def __init__(self, cell_embedding_dim, nhead, num_layers, dim_feedforward):
        super(TransformerEncoderModel, self).__init__()
        self.d_model = cell_embedding_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cell_embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, token_sequence, src_key_padding_mask=None):
        # token_sequence: [seq_len, batch_size, embedding_dim]
        seq_len = token_sequence.size(0)
        pe = positional_encoding(seq_len, self.d_model)
        token_sequence = token_sequence + pe
        output = self.transformer_encoder(token_sequence, src_key_padding_mask=src_key_padding_mask)
        return output

# ------------------------------
# Tabular Decoder for Reconstruction and Missingness Prediction
# ------------------------------
class TabularDecoder(nn.Module):
    """
    A decoder that reconstructs original cell values from transformer embeddings.
    For numeric features, outputs a scalar value.
    For categorical features, outputs logits over the category set.
    Additionally, a missingness decoder predicts a missingness logit for each cell.
    """
    def __init__(self, cell_embedding_dim, categorical_info):
        super(TabularDecoder, self).__init__()
        self.numeric_decoder = nn.Linear(cell_embedding_dim, 1)
        self.categorical_decoders = nn.ModuleDict()
        for var, info in categorical_info.items():
            num_classes = len(info['values']) + 1  # +1 for unknown token
            self.categorical_decoders[var] = nn.Linear(cell_embedding_dim, num_classes)
        self.missingness_decoder = nn.Linear(cell_embedding_dim, 1)

    def forward(self, cell_embeddings, sample_vars):
        """
        Decodes each cell embedding based on variable type and predicts missingness.
        
        Args:
            cell_embeddings (torch.Tensor): [num_cells, embedding_dim]
            sample_vars (list of str): Variable identifiers for each cell.
            
        Returns:
            predictions (dict): Mapping variable identifier to reconstruction prediction.
            missingness_predictions (dict): Mapping variable identifier to missingness logit.
        """
        predictions = {}
        missingness_predictions = {}
        for idx, var in enumerate(sample_vars):
            # Skip decoding for the overall token.
            if var.startswith("OVERALL_TOKEN"):
                continue
            emb = cell_embeddings[idx]  # [embedding_dim]
            if var.startswith("num:"):
                predictions[var] = self.numeric_decoder(emb.unsqueeze(0)).squeeze(0)
            elif var.startswith("cat:"):
                col_name = var[4:]  # remove 'cat:' prefix
                predictions[var] = self.categorical_decoders[col_name](emb.unsqueeze(0)).squeeze(0)
            # Predict missingness for every cell.
            missing_logit = self.missingness_decoder(emb.unsqueeze(0)).squeeze(0)
            missingness_predictions[var] = missing_logit
        return predictions, missingness_predictions

# ------------------------------
# Data Preprocessing Functions
# ------------------------------
def preprocess_data(df, numeric_info, categorical_info, normalization="standard"):
    """
    Preprocesses the dataframe:
      - Numeric columns: standardizes (or min–max scales) based on provided info.
      - Categorical columns: maps values to indices (with a reserved unknown index).
      - Generates missingness masks (1 if missing, 0 if observed).
    
    Returns:
        processed_features: dict mapping column name to preprocessed Series.
        missing_masks: dict mapping column name to missingness mask Series.
    """
    processed_features = {}
    missing_masks = {}
    for col in df.columns:
        if col in numeric_info:
            # Use standardization if possible; otherwise fall back to min–max scaling.
            if normalization == "standard":
                # Compute mean and std from the provided min and max (for demo, we use midpoint/std approx)
                min_val = numeric_info[col]['min']
                max_val = numeric_info[col]['max']
                mean_val = (min_val + max_val) / 2.0
                std_val = (max_val - min_val) / 4.0  # assume 95% within 2 std dev
                processed_features[col] = (df[col] - mean_val) / std_val
            else:
                min_val = numeric_info[col]['min']
                max_val = numeric_info[col]['max']
                processed_features[col] = (df[col] - min_val) / (max_val - min_val)
            missing_masks[col] = df[col].isnull().astype(float)
        elif col in categorical_info:
            mapping = {str(val): idx for idx, (val, _) in enumerate(categorical_info[col]['values'])}
            # Reserve index for unknown categories.
            mapping["unknown"] = len(mapping)
            processed_features[col] = df[col].astype(str).map(lambda x: mapping.get(x, mapping["unknown"]))
            missing_masks[col] = df[col].isnull().astype(float)
    return processed_features, missing_masks

def parse_data_dictionary(data_dict):
    """
    Parses the data dictionary to extract numeric and categorical variable information 
    and the overall dataset description. Expects a row with type 'overall' containing 
    an 'overall_description' column.
    
    Returns:
        numeric_info: dict mapping variable name to its info.
        categorical_info: dict mapping variable name to its info.
        overall_description: str or None.
    """
    numeric_info = {}
    categorical_info = {}
    overall_description = None

    for _, row in data_dict.iterrows():
        if row['type'] == 'overall':
            if 'overall_description' in row and pd.notnull(row['overall_description']):
                overall_description = row['overall_description']
            continue

        var_name = row['name']
        var_type = row['type']
        if var_type in ['continuous', 'integer']:
            numeric_info[var_name] = {
                'type': var_type,
                'min': row['min'],
                'max': row['max'],
                'description': row['description']
            }
        elif var_type == 'categorical':
            if var_name not in categorical_info:
                categorical_info[var_name] = {
                    'values': [],
                    'description': row['description']
                }
            categorical_info[var_name]['values'].append((row['value'], row['label']))
    return numeric_info, categorical_info, overall_description

def encode_metadata(variable_info, encoder):
    """
    Uses the pretrained encoder to produce metadata embeddings for each variable.
    Detaches the embeddings to avoid backpropagating through BERT repeatedly.
    
    Args:
        variable_info (dict): Mapping of variable names to info (including description).
        encoder: An encoder instance with an `encode` method.
        
    Returns:
        dict: Mapping from variable name to metadata embedding.
    """
    metadata_embeddings = {}
    for var, info in variable_info.items():
        description = info['description']
        metadata_embeddings[var] = encoder.encode(description).detach()
    return metadata_embeddings

# ------------------------------
# Custom PyTorch Dataset and Collate Function for Batch Processing
# ------------------------------
class TabularDataset(Dataset):
    """
    A dataset for preprocessed tabular data. Expects preprocessed features and missing masks.
    """
    def __init__(self, processed_features, missing_masks, columns):
        self.processed_features = processed_features
        self.missing_masks = missing_masks
        self.columns = columns  # list of column names in order

    def __len__(self):
        return len(next(iter(self.processed_features.values())))

    def __getitem__(self, idx):
        sample = {col: self.processed_features[col].iloc[idx] for col in self.columns}
        mask = {col: self.missing_masks[col].iloc[idx] for col in self.columns}
        return {"sample": sample, "mask": mask}

def collate_fn(batch):
    """
    Custom collate function to combine samples into a batch.
    Returns a dict with keys 'sample' and 'mask' mapping to lists.
    """
    samples = [b["sample"] for b in batch]
    masks = [b["mask"] for b in batch]
    return {"sample": samples, "mask": masks}

# ------------------------------
# Batch Encoder: Vectorized Encoding of a Batch
# ------------------------------
def encode_sample_batch(batch_samples, metadata_embeddings, mlp_num, mlp_cat, numeric_info, categorical_info):
    """
    Encodes a batch of samples by fusing preprocessed values, missingness flags,
    and metadata embeddings.
    
    Args:
        batch_samples (list of dict): Each dict maps column names to preprocessed value.
        metadata_embeddings (dict): Mapping from variable name to metadata embedding.
        mlp_num, mlp_cat: MLP modules for numeric and categorical variables.
        numeric_info, categorical_info: Dictionaries with variable metadata.
        
    Returns:
        token_sequence: Tensor of shape [seq_len, batch_size, cell_embedding_dim]
        sample_vars: List of strings identifying each cell (same across batch)
    """
    batch_size = len(batch_samples)
    cell_representations = []
    sample_vars = []
    # Use the keys from the first sample; assume order is consistent.
    for var in batch_samples[0].keys():
        col_vals = []
        missing_flags = []
        for sample in batch_samples:
            val = sample[var]
            # Determine variable type.
            if var in numeric_info:
                missing_flag = 1.0 if pd.isnull(val) else 0.0
                actual_value = float(val) if pd.notnull(val) else 0.0
                col_vals.append(actual_value)
                missing_flags.append(missing_flag)
            elif var in categorical_info:
                missing_flag = 1.0 if pd.isnull(val) else 0.0
                actual_value = float(val)  # Already encoded as index
                col_vals.append(actual_value)
                missing_flags.append(missing_flag)
            else:
                # Fallback: use zero.
                col_vals.append(0.0)
                missing_flags.append(0.0)
        # Convert lists to tensors.
        col_vals_tensor = torch.tensor(col_vals, dtype=torch.float32, device=device).unsqueeze(1)  # shape [B, 1]
        missing_tensor = torch.tensor(missing_flags, dtype=torch.float32, device=device).unsqueeze(1)  # shape [B, 1]
        meta_emb = metadata_embeddings[var].unsqueeze(0).repeat(batch_size, 1)  # shape [B, embedding_dim]
        combined = torch.cat([col_vals_tensor, missing_tensor, meta_emb], dim=1)  # shape [B, 2+embedding_dim]
        # Process with appropriate MLP.
        if var in numeric_info:
            cell_repr = mlp_num(combined)  # shape [B, cell_embedding_dim]
            sample_vars.append(f"num:{var}")
        elif var in categorical_info:
            cell_repr = mlp_cat(combined)
            sample_vars.append(f"cat:{var}")
        else:
            cell_repr = meta_emb
            sample_vars.append(var)
        cell_representations.append(cell_repr)
    # Stack cell representations to form a tensor of shape [num_cells, B, cell_embedding_dim]
    cell_tensor = torch.stack(cell_representations, dim=0)
    return cell_tensor, sample_vars

# ------------------------------
# Batched Loss Computation
# ------------------------------
def compute_reconstruction_loss_batch(predictions, missingness_predictions, batch_samples, batch_masks, sample_vars,
                                      numeric_info, categorical_info, missing_loss_weight=1.0):
    """
    Computes the total loss for a batch.
    
    Args:
        predictions: dict mapping variable id to reconstruction predictions (tensor of shape [B] or [B, num_classes]).
        missingness_predictions: dict mapping variable id to missingness logits (tensor of shape [B]).
        batch_samples: list of sample dicts.
        batch_masks: list of mask dicts.
        sample_vars: list of variable ids (strings).
    
    Returns:
        total_loss: scalar loss.
    """
    mse_loss = nn.MSELoss(reduction='sum')
    ce_loss = nn.CrossEntropyLoss(reduction='sum')
    bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
    
    total_recon_loss = 0.0
    total_missing_loss = 0.0
    observed_count = 0
    B = len(batch_samples)
    # For each variable (cell) across batch.
    for var_id in sample_vars:
        if var_id.startswith("OVERALL_TOKEN"):
            continue
        var = var_id.split(":", 1)[-1]
        # Create target tensor for reconstruction loss.
        # For numeric variables:
        if var_id.startswith("num:"):
            targets = []
            valid = []
            for mask, sample in zip(batch_masks, batch_samples):
                if mask[var] == 0.0 and pd.notnull(sample[var]):
                    targets.append(float(sample[var]))
                    valid.append(1.0)
                else:
                    targets.append(0.0)
                    valid.append(0.0)
            targets = torch.tensor(targets, dtype=torch.float32, device=device).unsqueeze(1)
            valid = torch.tensor(valid, dtype=torch.float32, device=device).unsqueeze(1)
            if valid.sum() > 0:
                pred = predictions[var_id].unsqueeze(1)
                total_recon_loss += mse_loss(pred * valid, targets * valid)
                observed_count += valid.sum().item()
        elif var_id.startswith("cat:"):
            targets = []
            valid = []
            for mask, sample in zip(batch_masks, batch_samples):
                if mask[var] == 0.0 and pd.notnull(sample[var]):
                    targets.append(int(sample[var]))
                    valid.append(1)
                else:
                    targets.append(0)
                    valid.append(0)
            targets = torch.tensor(targets, dtype=torch.long, device=device)
            valid = torch.tensor(valid, dtype=torch.float32, device=device)
            if valid.sum() > 0:
                pred_logits = predictions[var_id]
                total_recon_loss += ce_loss(pred_logits, targets)
                observed_count += valid.sum().item()
        # Missingness loss for all cells.
        missing_targets = []
        for mask in batch_masks:
            missing_targets.append(mask[var])
        missing_targets = torch.tensor(missing_targets, dtype=torch.float32, device=device)
        pred_missing = missingness_predictions[var_id]
        total_missing_loss += bce_loss(pred_missing.unsqueeze(1), missing_targets.unsqueeze(1))
    
    total_cells = (len(sample_vars) - 1) * B  # exclude overall token
    if observed_count == 0:
        observed_count = 1
    total_loss = (total_recon_loss / observed_count) + (missing_loss_weight * total_missing_loss / total_cells)
    return total_loss

# ------------------------------
# Training Loop with Batch Processing and Multiple Rounds
# ------------------------------
def train_foundational_model(dataset_paths, dictionary_paths, encoder,
                             transformer_model, mlp_num, mlp_cat,
                             num_rounds=1, num_epochs=5, batch_size=4):
    # Initialize decoder with an empty categorical decoders dict.
    initial_categorical_info = {}
    decoder = TabularDecoder(cell_embedding_dim=mlp_num.net[-1].out_features,
                              categorical_info=initial_categorical_info).to(device)
    
    optimizer = optim.Adam(
        list(transformer_model.parameters()) +
        list(mlp_num.parameters()) +
        list(mlp_cat.parameters()) +
        list(decoder.parameters()),
        lr=1e-3
    )
    
    # Outer loop: iterate over multiple rounds
    for round in range(num_rounds):
        print(f"Starting training round {round+1}/{num_rounds}")
        for data_path, dict_path in zip(dataset_paths, dictionary_paths):
            df, data_dict = load_dataset(data_path, dict_path)
            numeric_info, categorical_info, overall_description = parse_data_dictionary(data_dict)
            
            # Update decoder for new categorical columns.
            for col, info in categorical_info.items():
                if col not in decoder.categorical_decoders:
                    num_classes = len(info['values']) + 1  # +1 for unknown
                    decoder.categorical_decoders[col] = nn.Linear(mlp_num.net[-1].out_features, num_classes).to(device)
                    optimizer.add_param_group({'params': decoder.categorical_decoders[col].parameters()})
            
            # Preprocess the data.
            processed_features, missing_masks = preprocess_data(df, numeric_info, categorical_info, normalization="standard")
            columns = list(df.columns)
            dataset = TabularDataset(processed_features, missing_masks, columns)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            
            # Precompute metadata embeddings for numeric and categorical variables.
            metadata_numeric = encode_metadata(numeric_info, encoder)
            metadata_categorical = encode_metadata(categorical_info, encoder)
            metadata_embeddings = {**metadata_numeric, **metadata_categorical}
            
            # Encode overall dataset description once.
            if overall_description is not None:
                dataset_context = encoder.encode(overall_description).detach().unsqueeze(0)  # shape: [1, embedding_dim]
            else:
                dataset_context = torch.zeros(encoder.embedding_dim, device=device).unsqueeze(0)
            
            transformer_model.train()
            mlp_num.train()
            mlp_cat.train()
            decoder.train()
            
            # For each epoch over this dataset.
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                num_batches = 0
                for batch in dataloader:
                    batch_samples = batch["sample"]
                    batch_masks = batch["mask"]
                    # Vectorized encoding for the batch.
                    cell_encoding, sample_vars = encode_sample_batch(batch_samples, metadata_embeddings,
                                                                       mlp_num, mlp_cat, numeric_info, categorical_info)
                    # Prepend overall token to each sample.
                    # overall_token: shape [1, batch_size, embedding_dim]
                    overall_token = dataset_context.repeat(1, cell_encoding.size(1), 1)
                    token_sequence = torch.cat([overall_token, cell_encoding], dim=0)
                    # Append overall token id for tracking.
                    sample_vars_full = ["OVERALL_TOKEN"] + sample_vars

                    # Pass through transformer.
                    encoded_sequence = transformer_model(token_sequence)
                    # Remove overall token from output.
                    cell_outputs = encoded_sequence[1:, :, :]  # shape: [num_cells, batch_size, cell_embedding_dim]
                    # For simplicity, process each cell independently (flatten batch dimension).
                    # Then, decoder expects a tensor of shape [num_cells * batch_size, cell_embedding_dim]
                    B = cell_outputs.size(1)
                    num_cells = cell_outputs.size(0)
                    cell_outputs_flat = cell_outputs.view(num_cells * B, -1)
                    
                    # Create predictions for each variable over the batch.
                    predictions = {}
                    missingness_predictions = {}
                    for i, var in enumerate(sample_vars):  # Use sample_vars instead of sample_vars_full
                        # Extract embeddings for cell i across the batch.
                        cell_emb = cell_outputs[i, :, :]  # shape: [B, cell_embedding_dim]
                        if var.startswith("num:"):
                            predictions[var] = decoder.numeric_decoder(cell_emb)  # [B, 1] (numeric prediction)
                        elif var.startswith("cat:"):
                            col_name = var[4:]
                            predictions[var] = decoder.categorical_decoders[col_name](cell_emb)  # [B, num_classes]
                        missingness_predictions[var] = decoder.missingness_decoder(cell_emb).squeeze(1)  # [B]
                            
                    loss = compute_reconstruction_loss_batch(predictions, missingness_predictions,
                                                             batch_samples, batch_masks, sample_vars_full,
                                                             numeric_info, categorical_info, missing_loss_weight=1.0)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                print(f"Round {round+1}, Epoch {epoch+1} on dataset {data_path}: Avg Loss: {epoch_loss/num_batches:.4f}")
    return transformer_model

# ------------------------------
# Data Ingestion
# ------------------------------
def load_dataset(data_path, dictionary_path):
    """
    Loads a dataset and its data dictionary from CSV files.
    """
    data = pd.read_csv(data_path)
    data_dict = pd.read_csv(dictionary_path)
    return data, data_dict

# ------------------------------
# Dummy Dataset and Dictionary Generation (with Missing Values)
# ------------------------------
def create_dummy_dataset1():
    """
    Generates the first dummy dataset and its data dictionary.
    Columns: 'age', 'gender', 'income'.
    Introduces missing values in 'income' and 'gender'.
    """
    df = pd.DataFrame({
         'age': [25, 30, 22, 28, np.nan],
         'gender': ['0', '1', '0', np.nan, '0'],
         'income': [50000, np.nan, 45000, 52000, 58000]
    })
    dict_rows = [
       {'name': 'age', 'type': 'integer', 'description': 'Age of the individual', 'min': 18, 'max': 100},
       {'name': 'income', 'type': 'continuous', 'description': 'Annual income in USD', 'min': 0, 'max': 1000000},
       {'name': 'gender', 'type': 'categorical', 'description': 'Gender of the individual', 'value': '0', 'label': 'Male'},
       {'name': 'gender', 'type': 'categorical', 'description': 'Gender of the individual', 'value': '1', 'label': 'Female'},
       {'name': 'dataset', 'type': 'overall', 'description': 'Overall dataset info', 
        'overall_description': 'This dataset contains demographic information including age, income, and gender.'}
    ]
    data_dict = pd.DataFrame(dict_rows)
    return df, data_dict

def create_dummy_dataset2():
    """
    Generates the second dummy dataset and its data dictionary.
    Columns: 'age', 'gender', 'salary' (instead of income).
    Introduces missing values in 'salary' and 'age'.
    """
    df = pd.DataFrame({
         'age': [40, 45, np.nan, 50, 42],
         'gender': ['1', '0', '1', np.nan, '1'],
         'salary': [65000, 72000, 68000, np.nan, 71000]
    })
    dict_rows = [
       {'name': 'age', 'type': 'integer', 'description': 'Age in years', 'min': 18, 'max': 100},
       {'name': 'salary', 'type': 'continuous', 'description': 'Annual salary in USD', 'min': 0, 'max': 200000},
       {'name': 'gender', 'type': 'categorical', 'description': 'Reported gender', 'value': '0', 'label': 'Male'},
       {'name': 'gender', 'type': 'categorical', 'description': 'Reported gender', 'value': '1', 'label': 'Female'},
       {'name': 'dataset', 'type': 'overall', 'description': 'Overall dataset info', 
        'overall_description': 'This second dataset contains demographic information including age, salary, and gender from a different population.'}
    ]
    data_dict = pd.DataFrame(dict_rows)
    return df, data_dict

def save_dummy_datasets():
    """
    Saves the two dummy datasets and their dictionaries as CSV files.
    """
    df1, data_dict1 = create_dummy_dataset1()
    df1.to_csv("data/dummy_dataset1.csv", index=False)
    data_dict1.to_csv("data/dummy_dict1.csv", index=False)
    
    df2, data_dict2 = create_dummy_dataset2()
    df2.to_csv("data/dummy_dataset2.csv", index=False)
    data_dict2.to_csv("data/dummy_dict2.csv", index=False)

# ------------------------------
# Main Execution with Demonstration
# ------------------------------
def main():
    # Create and save dummy CSV datasets.
    save_dummy_datasets()
    
    # Hyperparameters.
    embedding_dim = 16       # Dimension for metadata embeddings.
    mlp_hidden_dim = 32      # Hidden dimension for MLPs.
    cell_embedding_dim = 16  # Output dimension for cell encoders.
    nhead = 2                # Number of transformer attention heads.
    num_layers = 2           # Number of transformer encoder layers.
    dim_feedforward = 64     # Dimension of the transformer feedforward network.
    num_epochs = 50
    batch_size = 4
    num_rounds = 3         # New parameter: number of rounds to visit each dataset
    
    # Initialize the pretrained BERT encoder.
    pretrained_encoder = PretrainedBertEncoder(embedding_dim=embedding_dim)
    
    # Initialize MLPs for numeric and categorical features.
    # Input dimension: 2 (value, missing_flag) + embedding_dim.
    mlp_num = MLPNum(input_dim=2 + embedding_dim, hidden_dim=mlp_hidden_dim, output_dim=cell_embedding_dim).to(device)
    mlp_cat = MLPCat(input_dim=2 + embedding_dim, hidden_dim=mlp_hidden_dim, output_dim=cell_embedding_dim).to(device)
    
    # Initialize the transformer encoder.
    transformer_model = TransformerEncoderModel(
        cell_embedding_dim=cell_embedding_dim,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward
    ).to(device)
    
    # File paths for the two dummy datasets.
    dataset_paths = ["data/dummy_dataset1.csv", "data/dummy_dataset2.csv"]
    dictionary_paths = ["data/dummy_dict1.csv", "data/dummy_dict2.csv"]
    
    # Train the autoencoder model on both datasets over multiple rounds.
    trained_model = train_foundational_model(
        dataset_paths, dictionary_paths, pretrained_encoder, transformer_model, mlp_num, mlp_cat,
        num_rounds=num_rounds, num_epochs=num_epochs, batch_size=batch_size
    )
    
    # Save the trained transformer's state.
    torch.save(trained_model.state_dict(), "transformer_model.pth")
    print("Training complete and model saved.")
    
    # ---------------
    # Demonstration: Handling Missing Cells on a Sample from Dataset 1
    # ---------------
    df_demo, data_dict_demo = create_dummy_dataset1()
    numeric_info, categorical_info, overall_description = parse_data_dictionary(data_dict_demo)
    processed_features, missing_masks = preprocess_data(df_demo, numeric_info, categorical_info, normalization="standard")
    metadata_numeric = encode_metadata(numeric_info, pretrained_encoder)
    metadata_categorical = encode_metadata(categorical_info, pretrained_encoder)
    metadata_embeddings = {**metadata_numeric, **metadata_categorical}
    
    # For demonstration, select the first sample.
    sample = {col: processed_features[col].iloc[0] for col in df_demo.columns}
    mask = {col: missing_masks[col].iloc[0] for col in df_demo.columns}
    batch_samples = [sample]  # Batch of 1.
    batch_masks = [mask]
    
    # Encode the sample.
    cell_encoding, sample_vars = encode_sample_batch(batch_samples, metadata_embeddings, mlp_num, mlp_cat,
                                                       numeric_info, categorical_info)
    if overall_description is not None:
        dataset_context = pretrained_encoder.encode(overall_description).detach().unsqueeze(0)
    else:
        dataset_context = torch.zeros(pretrained_encoder.embedding_dim, device=device).unsqueeze(0)
    overall_token = dataset_context.unsqueeze(1)  # shape: [1, 1, embedding_dim]
    token_sequence = torch.cat([overall_token, cell_encoding], dim=0)  # [seq_len, 1, embedding_dim]
    
    transformer_model.eval()
    with torch.no_grad():
        encoded_sequence = transformer_model(token_sequence)
        cell_outputs = encoded_sequence[1:, 0, :]
    
    # Initialize a decoder for demonstration.
    demo_decoder = TabularDecoder(cell_embedding_dim=cell_embedding_dim, categorical_info=categorical_info).to(device)
    predictions, missingness_predictions = demo_decoder(cell_outputs, sample_vars)
    
    print("\nDemonstration of Missing Cell Handling:")
    for var_id in sample_vars:
        pred_val = predictions[var_id].detach().cpu().numpy()
        miss_logit = missingness_predictions[var_id].item()
        print(f"{var_id} - Reconstruction Prediction: {pred_val}, Missingness Logit: {miss_logit:.4f}")

if __name__ == "__main__":
    main()
