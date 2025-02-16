import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# For real survival data, we use the lung dataset from lifelines.
# Install lifelines via: pip install lifelines
from lifelines.datasets import load_lung

# ===============================================
# 1. Custom Dataset Definition (unchanged)
# ===============================================
class LungSurvivalDataset(Dataset):
    """
    A PyTorch Dataset for survival data.

    Assumes the dataframe has:
      - a time column (observed time)
      - an event column (1 if event occurred, 0 if censored)
      - several predictor columns (features)

    Note: In the lung dataset, 'status' is coded as 1 (censored) and 2 (event).
    We convert it to 0/1.
    """
    def __init__(self, df, feature_cols, time_col='time', event_col='status'):
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.time_col = time_col
        self.event_col = event_col

        # Convert status: assume status==2 means event occurred, so delta = status - 1.
        self.df[self.event_col] = self.df[self.event_col] - 1

        # Standardize features (optional but often recommended)
        self.features = self.df[feature_cols].values.astype(np.float32)
        self.times = self.df[time_col].values.astype(np.float32)
        self.events = self.df[event_col].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.features[idx]
        t = self.times[idx]
        delta = self.events[idx]
        return torch.tensor(x), torch.tensor(t), torch.tensor(delta)

# ===============================================
# 2. Autoregressive Deep Survival Model
# ===============================================
class AutoregressiveDeepSurvivalModel(nn.Module):
    """
    Autoregressive deep survival model.

    The hazard function \(\lambda(x,t)\) is modeled as evolving over time via a GRU cell.
    The model works as follows:
      1. A static encoder maps the covariates \(x\) to an initial hidden state \(h_0\).
      2. Over \(n\) discrete time steps, a GRU cell updates the hidden state:
         \[
         h_{i+1} = \text{GRUCell}(dt, h_i)
         \]
         where the input is a small time increment \(dt = t/n\).
      3. At each step, the hazard is computed as:
         \[
         \lambda_i = \text{softplus}(W h_i + b)
         \]
      4. The cumulative hazard is approximated as:
         \[
         H(x,t) \approx \sum_{i=1}^{n} \lambda_i \, dt.
         \]
    """
    def __init__(self, input_dim, hidden_dim=32):
        """
        Parameters:
          input_dim: number of features in x.
          hidden_dim: dimension of the recurrent hidden state.
        """
        super(AutoregressiveDeepSurvivalModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Encoder: maps static covariates to the initial hidden state
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # GRU cell: input dimension is 1 (the time increment dt) and hidden_dim
        self.gru_cell = nn.GRUCell(input_size=1, hidden_size=hidden_dim)
        
        # Hazard output layer: maps hidden state to a hazard value (ensured positive via softplus)
        self.hazard_layer = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()

    def forward_integration(self, x, t, n_steps=100):
        """
        Compute the integrated hazard and the hazard at observed time t in an autoregressive manner.

        Parameters:
          x: tensor of shape (batch, num_features)
          t: tensor of shape (batch,) representing observed times
          n_steps: number of discrete time steps for integration

        Returns:
          integrated: tensor of shape (batch,) containing the cumulative hazard \(\int_0^t \lambda(x,s) ds\)
          hazard_t: tensor of shape (batch,) containing \(\lambda(x,t)\) computed at time t.
        """
        batch_size = x.shape[0]
        # Encode static covariates into an initial hidden state h0
        h = self.encoder(x)  # shape: (batch, hidden_dim)
        
        # Create tensor to accumulate the integrated hazard for each sample
        integrated = torch.zeros(batch_size, device=x.device)
        
        # Each sample can have its own observed time; compute dt per sample: dt = t/n_steps.
        # dt has shape (batch, 1) so it can be fed to the GRU cell.
        dt = (t / n_steps).unsqueeze(1)  # shape: (batch, 1)
        
        # Iterate over n_steps to simulate the autoregressive evolution of the hazard
        for i in range(n_steps):
            # Update hidden state with dt as the input.
            h = self.gru_cell(dt, h)  # shape: (batch, hidden_dim)
            # Compute hazard at current time step
            hazard = self.softplus(self.hazard_layer(h)).squeeze(1)  # shape: (batch,)
            # Accumulate the integrated hazard using a Riemann sum
            integrated = integrated + hazard * dt.squeeze(1)
        
        # The hazard at the observed time is taken as the hazard from the final update
        hazard_t = hazard
        return integrated, hazard_t

    def integrated_hazard(self, x, t, n_steps=100):
        """
        Compute the cumulative hazard H(x,t) = ∫₀ᵗ λ(x,s) ds.

        Returns:
          integrated: tensor of shape (batch, 1)
        """
        integrated, _ = self.forward_integration(x, t, n_steps)
        return integrated.unsqueeze(1)

    def hazard_at_time(self, x, t, n_steps=100):
        """
        Compute the hazard λ(x,t) at time t.

        Returns:
          hazard_t: tensor of shape (batch, 1)
        """
        _, hazard_t = self.forward_integration(x, t, n_steps)
        return hazard_t.unsqueeze(1)

    def predict_survival(self, x, time_grid, n_steps=100):
        """
        Predict survival probability curves for new samples.

        For each sample, the survival function is computed as:
          S(t) = exp(-∫₀ᵗ λ(x, s) ds)

        Parameters:
          x: tensor of shape (batch, num_features)
          time_grid: 1D tensor of time points at which to evaluate S(t)
          n_steps: number of time steps for approximating the integral for each t

        Returns:
          survival_curves: tensor of shape (batch, len(time_grid))
        """
        batch_size = x.shape[0]
        survival_curves = []
        # For each sample in the batch, simulate the autoregressive updates
        for i in range(batch_size):
            s_i = []
            for t_val in time_grid:
                t_tensor = t_val.unsqueeze(0)  # shape: (1,)
                integrated, _ = self.forward_integration(x[i].unsqueeze(0), t_tensor, n_steps)
                s_val = torch.exp(-integrated)  # s_val has shape (1,)
                # Use squeeze() to remove all dimensions of size 1 safely.
                s_i.append(s_val.squeeze())
            survival_curves.append(torch.stack(s_i))
        return torch.stack(survival_curves)


# ===============================================
# 3. Negative Log-Likelihood Loss Function for AR Model
# ===============================================
def negative_log_likelihood(model, x, t, delta, n_steps=100):
    """
    Computes the negative log-likelihood for survival data using the autoregressive model.

    For each sample i:
      loss_i = ∫₀^(t_i) λ(x_i, s) ds - δ_i * log(λ(x_i, t_i))

    Parameters:
      model: an instance of AutoregressiveDeepSurvivalModel
      x: tensor of shape (batch, num_features)
      t: tensor of shape (batch,)
      delta: tensor of shape (batch,) (1 if event observed, 0 if censored)
      n_steps: number of time steps for approximating the integral

    Returns:
      loss: scalar tensor representing the average loss over the batch.
    """
    integrated, hazard_t = model.forward_integration(x, t, n_steps=n_steps)
    # Clamp hazard values for numerical stability
    hazard_t = torch.clamp(hazard_t, min=1e-7)
    loss = integrated - delta * torch.log(hazard_t)
    return torch.mean(loss)

# ===============================================
# 4. Data Loading and Preprocessing (Using Lung Dataset)
# ===============================================
def load_and_preprocess_data():
    # Load lung dataset from lifelines
    df = load_lung()
    # Select a subset of columns: using 'age' and 'ph.ecog' as predictors.
    cols = ['age', 'ph.ecog', 'time', 'status']
    df = df[cols].dropna().reset_index(drop=True)

    # Standardize features (recommended)
    feature_cols = ['age', 'ph.ecog']
    for col in feature_cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    return df, feature_cols

# ===============================================
# 5. Training Loop (unchanged except for model type)
# ===============================================
def train_model(model, dataloader, n_epochs=100, lr=1e-3, n_steps=100):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for x_batch, t_batch, delta_batch in dataloader:
            x_batch = x_batch.to(device)
            t_batch = t_batch.to(device)
            delta_batch = delta_batch.to(device)

            optimizer.zero_grad()
            loss = negative_log_likelihood(model, x_batch, t_batch, delta_batch, n_steps=n_steps)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_batch.size(0)
        epoch_loss /= len(dataloader.dataset)
        losses.append(epoch_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")
    return losses

# ===============================================
# 6. Main Implementation
# ===============================================
if __name__ == "__main__":
    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess the data
    df, feature_cols = load_and_preprocess_data()
    print("Data sample:")
    print(df.head())

    # Create dataset and dataloader
    dataset = LungSurvivalDataset(df, feature_cols, time_col='time', event_col='status')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the autoregressive survival model
    input_dim = len(feature_cols)
    model = AutoregressiveDeepSurvivalModel(input_dim=input_dim, hidden_dim=32).to(device)

    # Train the model
    n_epochs = 200
    losses = train_model(model, dataloader, n_epochs=n_epochs, lr=1e-3, n_steps=50)

    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log-Likelihood Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.show()

    # ------------------------------
    # Predict Survival Curves
    # ------------------------------
    # Create a time grid for prediction (from time 0 to max observed time)
    max_time = df['time'].max()
    time_grid = torch.linspace(0, max_time, steps=100).to(device)

    model.eval()
    with torch.no_grad():
        # Get a batch of samples from the dataloader (or use new samples)
        x_sample, t_sample, delta_sample = next(iter(dataloader))
        x_sample = x_sample.to(device)
        # Predict survival curves for each sample in the batch
        survival_curves = model.predict_survival(x_sample, time_grid, n_steps=50)

    # Plot survival curves for the first 5 samples
    plt.figure(figsize=(8, 6))
    for i in range(min(5, survival_curves.shape[0])):
        plt.plot(time_grid.cpu().numpy(), survival_curves[i].cpu().numpy(), label=f"Sample {i}")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title("Predicted Survival Curves (Autoregressive Model)")
    plt.legend()
    plt.show()


    # ------------------------------
    # Predict and Plot Hazard Function Curves
    # ------------------------------
    # Create a time grid for hazard prediction (from time 0 to max observed time)
    time_grid = torch.linspace(0, max_time, steps=100).to(device)

    model.eval()
    hazard_curves = []
    with torch.no_grad():
        # We use the same x_sample obtained earlier from the dataloader.
        for i in range(min(5, x_sample.shape[0])):  # For the first 5 samples
            hazard_vals = []
            for t_val in time_grid:
                # t_val is a scalar; unsqueeze to shape (1,) for compatibility
                t_tensor = t_val.unsqueeze(0)
                # Compute the hazard at time t for sample i.
                # hazard_at_time returns a tensor of shape (1,1); we squeeze it to a scalar.
                hazard_val = model.hazard_at_time(x_sample[i].unsqueeze(0), t_tensor, n_steps=50)
                hazard_vals.append(hazard_val.squeeze())
            # Stack hazard values along the time dimension for sample i.
            hazard_curves.append(torch.stack(hazard_vals))
    # Stack curves for all selected samples: shape (n_samples, len(time_grid))
    hazard_curves = torch.stack(hazard_curves)

    # Plot hazard curves for the first 5 samples
    plt.figure(figsize=(8, 6))
    for i in range(hazard_curves.shape[0]):
        plt.plot(time_grid.cpu().numpy(), hazard_curves[i].cpu().numpy(), label=f"Sample {i}")
    plt.xlabel("Time")
    plt.ylabel("Hazard $\lambda(x,t)$")
    plt.title("Predicted Hazard Curves (Autoregressive Model)")
    plt.legend()
    plt.show()
