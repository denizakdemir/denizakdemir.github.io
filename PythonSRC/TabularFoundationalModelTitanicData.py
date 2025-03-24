#!/usr/bin/env python
"""
A realistic tabular autoencoder that integrates per–variable metadata 
and an overall dataset text description.

Key components:
1. A pretrained BERT encoder for generating fixed–length metadata embeddings.
2. MLP modules (MLPNum and MLPCat) to fuse a cell’s raw value with its metadata.
3. A transformer encoder (with sinusoidal positional encoding) that processes 
   a token sequence comprising a global overall–description token followed by cell tokens.
4. Decoding modules (a shared numeric decoder and per–column categorical decoders)
   to reconstruct original inputs.
5. A self–supervised training loop that applies MSE loss for numeric features 
   and cross–entropy loss for categorical features, ignoring missing values.
6. Demonstration using the Titanic dataset (real data).

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

# ------------------------------
# Pretrained BERT for Metadata Encoding
# ------------------------------
class PretrainedBertEncoder:
    """
    Wraps a pretrained BERT model and tokenizer to produce fixed–length embeddings 
    for text descriptions. The [CLS] token representation is projected to a lower dimension.
    """
    def __init__(self, embedding_dim, pretrained_model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size  # typically 768
        self.projection = nn.Linear(self.bert_hidden_size, embedding_dim)
        self.embedding_dim = embedding_dim

    def encode(self, text):
        """
        Encodes the input text into a fixed–length embedding.
        Args:
          text (str): The text description to encode.
        Returns:
          torch.Tensor: An embedding of size [embedding_dim].
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.bert(**inputs)
        cls_output = outputs.last_hidden_state[:, 0, :]  # shape: [1, bert_hidden_size]
        embedding = self.projection(cls_output)  # shape: [1, embedding_dim]
        return embedding.squeeze(0)  # shape: [embedding_dim]

# ------------------------------
# MLP Modules for Cell Encoding
# ------------------------------
class MLPNum(nn.Module):
    """
    MLP for encoding numeric features. Expects an input vector that is a concatenation 
    of a scalar value and its metadata embedding.
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
    MLP for encoding categorical features. Expects an input vector that is a concatenation 
    of the categorical index (as a float) and its metadata embedding.
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
    return pe.unsqueeze(1)

# ------------------------------
# Transformer Encoder with Positional Encoding
# ------------------------------
class TransformerEncoderModel(nn.Module):
    """
    A transformer encoder that adds sinusoidal positional encodings to a token sequence 
    and outputs updated embeddings.
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
        pe = positional_encoding(seq_len, self.d_model).to(token_sequence.device)
        token_sequence = token_sequence + pe
        output = self.transformer_encoder(token_sequence, src_key_padding_mask=src_key_padding_mask)
        return output

# ------------------------------
# Tabular Decoder for Reconstruction
# ------------------------------
class TabularDecoder(nn.Module):
    """
    A decoder that reconstructs original cell values from transformer embeddings.
    For numeric features, outputs a scalar. For categorical features, outputs logits.
    """
    def __init__(self, cell_embedding_dim, categorical_info):
        super(TabularDecoder, self).__init__()
        self.numeric_decoder = nn.Linear(cell_embedding_dim, 1)
        self.categorical_decoders = nn.ModuleDict()
        for var, info in categorical_info.items():
            num_classes = len(info['values'])
            self.categorical_decoders[var] = nn.Linear(cell_embedding_dim, num_classes)

    def forward(self, cell_embeddings, sample_vars):
        """
        Decodes each cell embedding based on variable type.
        Args:
          cell_embeddings (torch.Tensor): [num_cells, embedding_dim]
          sample_vars (list of str): List of variable names (prefixed with 'num:' or 'cat:').
        Returns:
          dict: Mapping variable name to prediction tensor.
        """
        predictions = {}
        for idx, var in enumerate(sample_vars):
            emb = cell_embeddings[idx]  # [embedding_dim]
            if var.startswith("OVERALL_TOKEN"):
                continue
            if var.startswith("num:"):
                predictions[var] = self.numeric_decoder(emb.unsqueeze(0)).squeeze(0)
            elif var.startswith("cat:"):
                col_name = var[4:]
                predictions[var] = self.categorical_decoders[col_name](emb.unsqueeze(0)).squeeze(0)
            else:
                predictions[var] = emb
        return predictions

# ------------------------------
# Data Ingestion and Parsing
# ------------------------------
def load_dataset(data_path, dictionary_path):
    """
    Loads a dataset and its data dictionary from CSV files.
    """
    data = pd.read_csv(data_path)
    data_dict = pd.read_csv(dictionary_path)
    return data, data_dict

def parse_data_dictionary(data_dict):
    """
    Parses the data dictionary to extract numeric and categorical variable information 
    and the overall dataset description.
    Returns:
      numeric_info: dict mapping variable name to its info.
      categorical_info: dict mapping variable name to its info.
      overall_description: string or None.
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

def preprocess_data(df, numeric_info, categorical_info):
    """
    Preprocesses the dataframe:
      - For numeric columns: min–max normalization.
      - For categorical columns: maps values to indices.
      - Generates missingness masks (1 if missing, 0 if observed).
    Returns:
      processed_features: dict mapping column name to preprocessed Series.
      missing_masks: dict mapping column name to missing mask Series.
    """
    processed_features = {}
    missing_masks = {}
    for col in df.columns:
        if col in numeric_info:
            min_val = numeric_info[col]['min']
            max_val = numeric_info[col]['max']
            processed_features[col] = (df[col] - min_val) / (max_val - min_val)
            missing_masks[col] = df[col].isnull().astype(float)
        elif col in categorical_info:
            mapping = {str(val): idx for idx, (val, _) in enumerate(categorical_info[col]['values'])}
            processed_features[col] = df[col].astype(str).map(mapping)
            missing_masks[col] = df[col].isnull().astype(float)
    return processed_features, missing_masks

def encode_metadata(variable_info, encoder):
    """
    Uses the pretrained encoder to produce metadata embeddings for each variable.
    Args:
      variable_info (dict): Mapping of variable names to their info.
      encoder: An encoder instance with an `encode` method.
    Returns:
      dict: Mapping from variable name to metadata embedding.
    """
    metadata_embeddings = {}
    for var, info in variable_info.items():
        description = info['description']
        metadata_embeddings[var] = encoder.encode(description).detach()
    return metadata_embeddings

def encode_sample(sample, metadata_embeddings, mlp_num, mlp_cat, numeric_info, categorical_info):
    """
    Encodes each cell in a sample by fusing its preprocessed value with metadata.
    
    Args:
      sample (dict): Mapping of column names to preprocessed values.
      metadata_embeddings (dict): Mapping variable name to metadata embedding.
      mlp_num, mlp_cat: MLP modules for numeric and categorical variables.
      numeric_info (dict): Dictionary of numeric variable metadata.
      categorical_info (dict): Dictionary of categorical variable metadata.
      
    Returns:
      cell_tensor: Tensor of shape [num_cells, embedding_dim].
      sample_vars: List of strings identifying each cell, prefixed with 'num:' or 'cat:'.
    """
    cell_representations = []
    sample_vars = []
    for var, value in sample.items():
        # Check the variable type based on the data dictionary.
        if var in numeric_info:
            # For numeric variables, even if value is missing, use a default (e.g., 0.0)
            numeric_value = float(value) if pd.notnull(value) else 0.0
            numeric_tensor = torch.tensor([numeric_value], dtype=torch.float32)
            combined = torch.cat([numeric_tensor, metadata_embeddings[var]])
            cell_repr = mlp_num(combined)
            sample_vars.append(f"num:{var}")
        elif var in categorical_info:
            # For categorical variables, use a default index (e.g., 0) if missing.
            cat_value = float(value) if pd.notnull(value) else 0.0
            cat_tensor = torch.tensor([cat_value], dtype=torch.float32)
            combined = torch.cat([cat_tensor, metadata_embeddings[var]])
            cell_repr = mlp_cat(combined)
            sample_vars.append(f"cat:{var}")
        else:
            # Fallback for unexpected variables.
            default_tensor = torch.tensor([0.0], dtype=torch.float32)
            combined = torch.cat([default_tensor, metadata_embeddings.get(var, default_tensor)])
            cell_repr = mlp_num(combined)
            sample_vars.append(f"num:{var}")
        cell_representations.append(cell_repr.unsqueeze(0))
    return torch.cat(cell_representations, dim=0), sample_vars

def compute_reconstruction_loss(predictions, sample, missing_masks, sample_vars, numeric_info, categorical_info):
    """
    Computes the reconstruction loss. For numeric variables, uses MSE loss; for categorical, cross–entropy loss.
    Missing values (mask==1 or sample is NaN) are skipped.
    Returns:
      loss: Scalar tensor representing total loss.
    """
    loss = 0.0
    num_count = 0
    cat_count = 0
    mse_loss = nn.MSELoss(reduction='sum')
    ce_loss = nn.CrossEntropyLoss(reduction='sum')
    
    for cell_id in sample_vars:
        if cell_id.startswith("num:"):
            var = cell_id[4:]
            if missing_masks[var].iloc[0] == 1.0 or pd.isnull(sample[var]):
                continue
            target = torch.tensor([[sample[var]]], dtype=torch.float32)
            pred = predictions[cell_id].unsqueeze(0)
            loss += mse_loss(pred, target)
            num_count += 1
        elif cell_id.startswith("cat:"):
            var = cell_id[4:]
            if missing_masks[var].iloc[0] == 1.0 or pd.isnull(sample[var]):
                continue
            target = torch.tensor([int(sample[var])], dtype=torch.long)
            pred_logits = predictions[cell_id].unsqueeze(0)
            loss += ce_loss(pred_logits, target)
            cat_count += 1
    total_count = num_count + cat_count if (num_count + cat_count) > 0 else 1
    return loss / total_count

# ------------------------------
# Training Loop
# ------------------------------
def train_foundational_model(dataset_paths, dictionary_paths, encoder,
                             transformer_model, mlp_num, mlp_cat, num_epochs=1):
    """
    Sequentially ingests datasets and trains the transformer–based autoencoder 
    using a multi–task reconstruction loss.
    """
    # Load the dictionary from the first file (assumes all share the same dictionary)
    _, categorical_info, _ = parse_data_dictionary(pd.read_csv(dictionary_paths[0]))
    decoder = TabularDecoder(cell_embedding_dim=mlp_num.net[-1].out_features,
                              categorical_info=categorical_info)
    
    optimizer = optim.Adam(
        list(transformer_model.parameters()) +
        list(mlp_num.parameters()) +
        list(mlp_cat.parameters()) +
        list(decoder.parameters()),
        lr=1e-3
    )
    
    for data_path, dict_path in zip(dataset_paths, dictionary_paths):
        df, data_dict = load_dataset(data_path, dict_path)
        numeric_info, categorical_info, overall_description = parse_data_dictionary(data_dict)
        processed_features, missing_masks = preprocess_data(df, numeric_info, categorical_info)
        
        # Generate metadata embeddings for both numeric and categorical variables.
        metadata_numeric = encode_metadata(numeric_info, encoder)
        metadata_categorical = encode_metadata(categorical_info, encoder)
        metadata_embeddings = {**metadata_numeric, **metadata_categorical}
        
        # Encode the overall dataset description.
        dataset_context = encoder.encode(overall_description).detach() if overall_description is not None else torch.zeros(encoder.embedding_dim)
        dataset_context = dataset_context.unsqueeze(0)  # shape: [1, embedding_dim]

        for epoch in range(num_epochs):
            for idx in range(len(df)):
                sample = {col: processed_features[col].iloc[idx] for col in processed_features}
                sample_mask = {col: missing_masks[col].iloc[[idx]] for col in missing_masks}
                
                # Encode cell–level representations with type information.
                cell_encoding, sample_vars = encode_sample(sample, metadata_embeddings, mlp_num, mlp_cat, numeric_info, categorical_info)
                overall_token = dataset_context  # [1, embedding_dim]
                token_sequence = torch.cat([overall_token, cell_encoding], dim=0)  # [seq_len, embedding_dim]
                sample_vars_full = ["OVERALL_TOKEN"] + sample_vars
                
                token_sequence = token_sequence.unsqueeze(1)  # add batch dimension
                
                encoded_sequence = transformer_model(token_sequence)
                cell_outputs = encoded_sequence[1:, 0, :]  # remove overall token
                
                predictions = decoder(cell_outputs, sample_vars)
                
                loss = compute_reconstruction_loss(predictions, sample, sample_mask, sample_vars, numeric_info, categorical_info)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if idx % 10 == 0:
                    print(f"Epoch {epoch}, Sample {idx}, Loss: {loss.item():.4f}")
    return transformer_model

# ------------------------------
# Real Data: Titanic Dataset and Dictionary Generation
# ------------------------------
def save_titanic_dataset():
    """
    Downloads the Titanic dataset from a public URL, selects relevant columns,
    and creates a corresponding data dictionary.
    The chosen columns are:
      - Age (numeric, continuous)
      - Fare (numeric, continuous)
      - Sex (categorical)
    An overall description is also provided.
    """
    # Download Titanic dataset
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    # Select only the relevant columns
    df = df[['Age', 'Fare', 'Sex']]
    df.to_csv("data/titanic_dataset.csv", index=False)

    # Create data dictionary
    age_info = {'name': 'Age', 'type': 'continuous', 'description': 'Age of the passenger', 'min': 0, 'max': 80}
    fare_info = {'name': 'Fare', 'type': 'continuous', 'description': 'Fare paid by the passenger', 'min': 0, 'max': 300}
    # For the categorical column "Sex", create one row per category.
    sex_info1 = {'name': 'Sex', 'type': 'categorical', 'description': 'Gender of the passenger', 'value': 'male', 'label': 'Male'}
    sex_info2 = {'name': 'Sex', 'type': 'categorical', 'description': 'Gender of the passenger', 'value': 'female', 'label': 'Female'}
    overall_info = {'name': 'dataset', 'type': 'overall', 'description': 'Titanic dataset overall description',
                    'overall_description': 'This dataset contains information on Titanic passengers including age, fare, and gender.'}
    dict_rows = [age_info, fare_info, sex_info1, sex_info2, overall_info]
    data_dict = pd.DataFrame(dict_rows)
    data_dict.to_csv("data/titanic_dict.csv", index=False)
    print("Titanic dataset and dictionary saved as 'data/titanic_dataset.csv' and 'data/titanic_dict.csv'.")

# ------------------------------
# Main Execution
# ------------------------------
def main():
    # Uncomment the following line to use the dummy dataset:
    # save_dummy_dataset()

    # For a real data example, we generate and save the Titanic dataset and its dictionary.
    save_titanic_dataset()
    
    # Hyperparameters.
    embedding_dim = 16       # Dimension for metadata embeddings.
    mlp_hidden_dim = 32      # Hidden layer size for MLPs.
    cell_embedding_dim = 16  # Output dimension for cell encoders.
    nhead = 2                # Transformer attention heads.
    num_layers = 2           # Number of transformer encoder layers.
    dim_feedforward = 64     # Feedforward network dimension in transformer.
    
    # Initialize the pretrained BERT encoder.
    pretrained_encoder = PretrainedBertEncoder(embedding_dim=embedding_dim)
    
    # Initialize MLPs for numeric and categorical features.
    mlp_num = MLPNum(input_dim=1 + embedding_dim, hidden_dim=mlp_hidden_dim, output_dim=cell_embedding_dim)
    mlp_cat = MLPCat(input_dim=1 + embedding_dim, hidden_dim=mlp_hidden_dim, output_dim=cell_embedding_dim)
    
    # Initialize the transformer encoder.
    transformer_model = TransformerEncoderModel(
        cell_embedding_dim=cell_embedding_dim,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward
    )
    
    # Use the Titanic dataset files.
    dataset_paths = ["titanic_dataset.csv"]
    dictionary_paths = ["titanic_dict.csv"]
    
    # Train the autoencoder model.
    trained_model = train_foundational_model(
        dataset_paths, dictionary_paths, pretrained_encoder, transformer_model, mlp_num, mlp_cat, num_epochs=5
    )
    
    # Save the trained transformer's state.
    torch.save(trained_model.state_dict(), "transformer_model.pth")
    print("Training complete and model saved as 'transformer_model.pth'.")

if __name__ == "__main__":
    main()
