import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

class AnomalyDetector:
    def __init__(self, input_dim, latent_dim=8):
        """
        Initializes the AnomalyDetector with:
        - An autoencoder for dimensionality reduction and reconstruction.
        - A StandardScaler to normalize data before feeding into the model.
        - A threshold for determining whether a data point is an outlier.
        """
        self.model = Autoencoder(input_dim, latent_dim)
        self.scaler = StandardScaler()
        self.threshold = None
    
    def fit(self, X_train, epochs=100, batch_size=32):
        """
        Trains the autoencoder on the provided training data.
        
        Args:
            X_train (numpy.ndarray): Training data.
            epochs (int): Number of training epochs.
            batch_size (int): Size of each training batch.
        """
        # Standardize the data
        X_scaled = self.scaler.fit_transform(X_train)
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Set up the optimizer and loss function
        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                
                # Forward pass
                reconstructed = self.model(batch)
                loss = criterion(reconstructed, batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Once trained, calculate and store the reconstruction error threshold
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            # Set threshold as mean + 4 standard deviations
            self.threshold = errors.mean() + 4 * errors.std()
    
    def predict(self, X):
        """
        Predicts whether each data point in X is normal (0) or anomalous (1).
        
        Args:
            X (numpy.ndarray): Data to classify.
        
        Returns:
            numpy.ndarray: Binary labels (1 for anomaly, 0 for normal).
        """
        # Standardize the data
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Calculate reconstruction error
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            
        # Return anomaly labels (1 for anomaly, 0 for normal)
        return (errors > self.threshold).numpy().astype(int)
    
    def get_reconstruction_error(self, X):
        """
        Computes the per-sample reconstruction error for input X.
        
        Args:
            X (numpy.ndarray): Data for which to compute reconstruction error.
        
        Returns:
            numpy.ndarray: Array of reconstruction errors.
        """
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
        
        return errors.numpy()

# Example usage
if __name__ == "__main__":
    # Generate synthetic normal training data
    np.random.seed(42)
    normal_data = np.random.normal(loc=0, scale=1, size=(10000, 10))
    
    # Create and train the AnomalyDetector
    detector = AnomalyDetector(input_dim=10)
    detector.fit(normal_data)
    
    # Generate test data: mostly normal with a small number of anomalies
    test_normal = np.random.normal(loc=0, scale=1, size=(1000, 10))
    test_anomalies = np.random.normal(loc=3, scale=2, size=(20, 10))
    test_data = np.vstack([test_normal, test_anomalies])
    
    # Compute predictions
    predictions = detector.predict(test_data)
    errors = detector.get_reconstruction_error(test_data)
    
    # Create ground-truth labels: 0 for normal, 1 for anomaly
    y_true = np.concatenate([np.zeros(len(test_normal)), np.ones(len(test_anomalies))])
    
    # Evaluate performance
    print(f"Detected {sum(predictions)} anomalies")
    print(f"Mean reconstruction error: {errors.mean():.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, predictions, target_names=['Normal', 'Anomaly']))
