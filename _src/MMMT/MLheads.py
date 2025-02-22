"""
MLheads.py

This module defines various neural network heads for biostatistics-related tasks.
Each head includes an optional default loss function, which the user may override.
The supervised tasks include:
    - Regression (using MSE loss)
    - Multi-class Classification (using Cross-Entropy loss)
    - Binomial Classification (using BCEWithLogits loss)
    - Negative Binomial Regression (using a custom negative log likelihood)
    - Poisson Regression (using PyTorch's PoissonNLLLoss)
    - DeepHit-style Time-to-Event (for censored competing risks)
    - Binomial Regression (for modeling counts of successes out of a fixed number of trials)
    - Cox Proportional Hazards (for survival analysis)

The unsupervised tasks include:
    - Clustering (producing soft cluster assignments using KL-divergence loss)
    - Dimension Reduction using UMAP, t-SNE, and PCA.
      (These are provided as wrappers around non-differentiable algorithms for post-hoc analysis.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# =============================================================================
# Supervised Heads with Customizable Architectures
# =============================================================================

class RegressionHead(nn.Module):
    """
    A regression head that outputs a continuous value.
    
    Default loss: Mean Squared Error (MSELoss)
    
    Parameters:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output. Default is 1.
        custom_nn (nn.Module, optional): A user-defined neural network module.
        default_loss (nn.Module, optional): Loss function. Defaults to MSELoss.
    """
    def __init__(self, input_dim: int, output_dim: int = 1, custom_nn: nn.Module = None, default_loss: nn.Module = None):
        super(RegressionHead, self).__init__()
        # Use user-provided module or default to a single linear layer.
        self.model = custom_nn if custom_nn is not None else nn.Linear(input_dim, output_dim)
        self.default_loss = default_loss if default_loss is not None else nn.MSELoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return output.squeeze(-1)  # Remove last dimension if it's 1

    
    def loss(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            predictions = predictions[mask]
            targets = targets[mask]
        return self.default_loss(predictions, targets)


class ClassificationHead(nn.Module):
    """
    A multi-class classification head that outputs logits.
    
    Default loss: CrossEntropyLoss
    
    Parameters:
        input_dim (int): Dimension of the input features.
        num_classes (int): Number of classes.
        custom_nn (nn.Module, optional): A user-defined neural network module.
        default_loss (nn.Module, optional): Loss function. Defaults to CrossEntropyLoss.
    """
    def __init__(self, input_dim: int, num_classes: int, custom_nn: nn.Module = None, default_loss: nn.Module = None):
        super(ClassificationHead, self).__init__()
        self.model = custom_nn if custom_nn is not None else nn.Linear(input_dim, num_classes)
        self.default_loss = default_loss if default_loss is not None else nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def loss(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            predictions = predictions[mask]
            targets = targets[mask]
        return self.default_loss(predictions, targets)


class BinomialClassificationHead(nn.Module):
    """
    A binomial (binary) classification head that outputs a single logit.
    
    Default loss: BCEWithLogitsLoss
    
    Parameters:
        input_dim (int): Dimension of the input features.
        custom_nn (nn.Module, optional): A user-defined neural network module.
        default_loss (nn.Module, optional): Loss function. Defaults to BCEWithLogitsLoss.
    """
    def __init__(self, input_dim: int, custom_nn: nn.Module = None, default_loss: nn.Module = None):
        super(BinomialClassificationHead, self).__init__()
        self.model = custom_nn if custom_nn is not None else nn.Linear(input_dim, 1)
        self.default_loss = default_loss if default_loss is not None else nn.BCEWithLogitsLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze the last dimension so that output shape matches the target's shape.
        return self.model(x).squeeze(-1)
    
    def loss(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # If a mask is provided, filter predictions and targets.
        if mask is not None:
            predictions = predictions[mask]
            targets = targets[mask]
        return self.default_loss(predictions, targets.float())


class NegativeBinomialHead(nn.Module):
    """
    A negative binomial regression head for modeling overdispersed count data.
    The head outputs log(mu) so that mu = exp(log_mu) is strictly positive.
    
    Default loss: Custom negative log likelihood for the negative binomial distribution.
    
    Parameters:
        input_dim (int): Dimension of the input features.
        r (float): Dispersion parameter (assumed fixed); higher r reduces overdispersion.
        custom_nn (nn.Module, optional): A user-defined neural network module.
        default_loss (callable, optional): Loss function. Defaults to the internal nb_loss.
    """
    def __init__(self, input_dim: int, r: float = 1.0, custom_nn: nn.Module = None, default_loss=None):
        super(NegativeBinomialHead, self).__init__()
        self.model = custom_nn if custom_nn is not None else nn.Linear(input_dim, 1)
        self.r = r  # Fixed dispersion parameter.
        self.default_loss = default_loss if default_loss is not None else self.nb_loss
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Predict log(mu); mu = exp(log_mu)
        log_mu = self.model(x)
        return log_mu
    
    def nb_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the negative log likelihood for the negative binomial distribution.
        
        The negative binomial likelihood (up to an additive constant) is:
            L = r * log(r) - r * log(r + mu) + y * log(mu) - y * log(r + mu)
        where mu = exp(predictions) and y are the observed counts.
        
        Terms that do not depend on predictions are omitted.
        """
        mu = torch.exp(predictions)
        r = self.r
        eps = 1e-8  # Avoid log(0)
        loss = r * torch.log(r + eps) - r * torch.log(r + mu + eps) \
               + targets * torch.log(mu + eps) - targets * torch.log(r + mu + eps)
        return -loss.mean()
    
    def loss(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            predictions = predictions[mask]
            targets = targets[mask]
        return self.default_loss(predictions, targets)


class PoissonRegressionHead(nn.Module):
    """
    A Poisson regression head for count data.
    The head outputs log(lambda) so that lambda = exp(log_lambda) is positive.
    
    Default loss: PoissonNLLLoss from PyTorch.
    
    Parameters:
        input_dim (int): Dimension of the input features.
        custom_nn (nn.Module, optional): A user-defined neural network module.
        default_loss (nn.Module, optional): Loss function. Defaults to PoissonNLLLoss.
    """
    def __init__(self, input_dim: int, custom_nn: nn.Module = None, default_loss: nn.Module = None):
        super(PoissonRegressionHead, self).__init__()
        self.model = custom_nn if custom_nn is not None else nn.Linear(input_dim, 1)
        self.default_loss = default_loss if default_loss is not None else nn.PoissonNLLLoss(log_input=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def loss(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            predictions = predictions[mask]
            targets = targets[mask]
        return self.default_loss(predictions, targets)

class BinomialRegressionHead(nn.Module):
    """
    A binomial regression head for modeling counts of successes out of a fixed number of trials.
    
    The head outputs a single logit which is transformed via a sigmoid into a probability of success.
    Given the number of trials (assumed fixed for all samples), the default loss is the negative log
    likelihood of the observed count (ignoring the constant binomial coefficient).
    
    The likelihood (up to an additive constant) is:
        L = k * log(p) + (trials - k) * log(1 - p)
    where:
        - p = sigmoid(logit) is the predicted probability of success,
        - k is the observed count of successes, and
        - trials is the fixed number of trials.
    
    Parameters:
        input_dim (int): Dimension of the input features.
        trials (int): The fixed number of trials for each sample.
        custom_nn (nn.Module, optional): A user-defined neural network module. Defaults to a single linear layer.
        default_loss (callable, optional): A loss function. Defaults to the internal binomial_loss.
    """
    def __init__(self, input_dim: int, trials: int, custom_nn: nn.Module = None, default_loss=None):
        super(BinomialRegressionHead, self).__init__()
        self.model = custom_nn if custom_nn is not None else nn.Linear(input_dim, 1)
        self.trials = trials  # Fixed number of trials per sample.
        self.default_loss = default_loss if default_loss is not None else self.binomial_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: computes the logit for the probability of success.
        
        Returns:
            torch.Tensor: A tensor of shape (batch_size,) representing the logits.
        """
        return self.model(x).squeeze(-1)
    
    def binomial_loss(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the negative log likelihood for binomial outcomes.
        
        Args:
            predictions (torch.Tensor): Logits of shape (batch_size,).
            targets (torch.Tensor): Observed counts of successes (0 <= k <= trials) of shape (batch_size,).
            mask (torch.Tensor, optional): Boolean tensor of shape (batch_size,) for selecting valid samples.
        
        Returns:
            torch.Tensor: The average negative log likelihood loss.
        """
        if mask is not None:
            predictions = predictions[mask]
            targets = targets[mask]
        eps = 1e-8  # To avoid log(0)
        # Convert logits to probabilities.
        p = torch.sigmoid(predictions)
        # Compute negative log likelihood (ignoring the binomial coefficient which is constant w.r.t. p)
        loss = -(targets * torch.log(p + eps) + (self.trials - targets) * torch.log(1 - p + eps))
        return loss.mean()

    def loss(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        A convenience method to compute the loss.
        """
        return self.default_loss(predictions, targets, mask=mask)



class DeepHitHead(nn.Module):
    """
    A DeepHit-style time-to-event head for censored competing risks.

    The head outputs discrete-time probability distributions over pre-specified time bins.
    For multiple competing risks, the output shape is (batch_size, num_events, time_bins);
    for a single event, the output shape is (batch_size, time_bins).

    The default loss is a likelihood loss computed as follows:
      - For an event (event > 0): 
            loss = -log(predicted probability at the event's time bin for the observed event type).
      - For a censored observation (event == 0): 
            loss = -log(sum of predicted probabilities over time bins later than the censoring time).

    Parameters:
        input_dim (int): Dimension of the input features.
        time_bins (int): Number of discrete time intervals.
        num_events (int): Number of competing risks. For a single event, use num_events=1.
        custom_nn (nn.Module, optional): A user-defined neural network module.
        default_loss (callable, optional): Loss function. Defaults to the internal deep_hit_loss.
    """
    def __init__(self, input_dim: int, time_bins: int, num_events: int = 1, custom_nn: nn.Module = None, default_loss=None):
        super(DeepHitHead, self).__init__()
        self.time_bins = time_bins
        self.num_events = num_events
        # For multiple events, the output dimension is num_events * time_bins.
        output_dim = num_events * time_bins if num_events > 1 else time_bins
        self.model = custom_nn if custom_nn is not None else nn.Linear(input_dim, output_dim)
        self.default_loss = default_loss if default_loss is not None else self.deep_hit_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if self.num_events > 1:
            # Reshape to (batch_size, num_events, time_bins)
            out = out.view(-1, self.num_events, self.time_bins)
            # Apply softmax over time bins for each event type.
            out = F.softmax(out, dim=2)
        else:
            # For single event, shape: (batch_size, time_bins)
            out = F.softmax(out, dim=1)
        return out

    def deep_hit_loss(self, predictions: torch.Tensor, targets: tuple, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the likelihood loss for DeepHit using vectorized tensor operations.

        Args:
            predictions: Tensor of shape (batch_size, time_bins) for single event or 
                         (batch_size, num_events, time_bins) for competing risks.
            targets: A tuple (times, events)
                     - times: LongTensor of shape (batch_size,) indicating the time bin index.
                     - events: LongTensor of shape (batch_size,) where 0 indicates censoring and
                               a positive integer indicates the event type.
            mask: Optional Boolean tensor of shape (batch_size,) indicating valid samples.

        Returns:
            The average negative log-likelihood loss.
        """
        eps = 1e-8  # Small constant to prevent log(0)
        times, events = targets  # Unpack target information

        # If a mask is provided, select only valid samples.
        if mask is not None:
            times = times[mask]
            events = events[mask]
            predictions = predictions[mask]

        batch_size = times.shape[0]

        if self.num_events > 1:
            # Multi-event case: predictions shape (batch_size, num_events, time_bins)

            # Create masks for observed events and censored data
            observed_mask = (events > 0)
            censored_mask = (events == 0)

            # Initialize loss tensor
            loss = torch.zeros(batch_size, device=predictions.device)

            # --- Observed Events ---
            if observed_mask.sum() > 0:
                # Identify indices of observed events
                observed_indices = torch.nonzero(observed_mask, as_tuple=True)[0]
                # Adjust event type to 0-indexing
                event_indices = events[observed_mask] - 1  
                time_indices = times[observed_mask]
                # Gather predicted probabilities for each observed event
                p_event = predictions[observed_indices, event_indices, time_indices]
                loss[observed_mask] = -torch.log(p_event + eps)

            # --- Censored Observations ---
            if censored_mask.sum() > 0:
                # For censored data, compute the survival probability as the sum of probabilities 
                # over all event types and over time bins greater than the censoring time.
                times_cens = times[censored_mask]
                # Create a time grid [0, 1, ..., time_bins-1]
                time_grid = torch.arange(self.time_bins, device=predictions.device).unsqueeze(0)  # Shape: (1, time_bins)
                # Create a mask: True for time bins after the censoring time.
                mask_time = (time_grid > times_cens.unsqueeze(1)).float()  # Shape: (n_cens, time_bins)
                # Multiply the predictions by the time mask (unsqueeze along event dim) and sum over events and time.
                p_survival = (predictions[censored_mask] * mask_time.unsqueeze(1)).sum(dim=(1, 2))
                # In cases where no time bins are available (e.g., t == time_bins-1), ensure non-zero value.
                p_survival = torch.where(p_survival > 0, p_survival, torch.full_like(p_survival, eps))
                loss[censored_mask] = -torch.log(p_survival + eps)

            return loss.mean()

        else:
            # Single event case: predictions shape (batch_size, time_bins)

            # Create masks for observed events and censored data.
            observed_mask = (events > 0)
            censored_mask = (events == 0)

            loss = torch.zeros(batch_size, device=predictions.device)

            # --- Observed Events ---
            if observed_mask.sum() > 0:
                observed_indices = torch.nonzero(observed_mask, as_tuple=True)[0]
                time_indices = times[observed_mask]
                # Gather the predicted probability for each observed event.
                p_event = predictions[observed_indices, time_indices]
                loss[observed_mask] = -torch.log(p_event + eps)

            # --- Censored Observations ---
            if censored_mask.sum() > 0:
                times_cens = times[censored_mask]
                time_grid = torch.arange(self.time_bins, device=predictions.device).unsqueeze(0)  # Shape: (1, time_bins)
                mask_time = (time_grid > times_cens.unsqueeze(1)).float()  # Shape: (n_cens, time_bins)
                # Sum probabilities over time bins after the censoring time.
                p_survival = (predictions[censored_mask] * mask_time).sum(dim=1)
                p_survival = torch.where(p_survival > 0, p_survival, torch.full_like(p_survival, eps))
                loss[censored_mask] = -torch.log(p_survival + eps)

            return loss.mean()

    def loss(self, predictions: torch.Tensor, targets: tuple, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Wrapper to call the default loss function.
        """
        return self.default_loss(predictions, targets, mask=mask)


class CoxPHHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # Your initialization code here.
        self.last_baseline = None  # To store (baseline_times, baseline_hazards)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass implementation.
        return x  # Example placeholder

    def loss(self, predictions: torch.Tensor, targets: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the negative log partial likelihood loss and the baseline hazards.
        Parameters:
            predictions: Risk scores (shape: (batch_size,)).
            targets: Tuple (times, events) with times and binary event indicators.
        Returns:
            A tuple (loss, baseline_times, baseline_hazards).
        """
        times, events = targets

        # (Assuming mask and reshaping are handled as in your provided code)
        times = times.view(-1)
        events = events.view(-1)
        predictions = predictions.view(-1)

        # Sort in descending order by time.
        sorted_indices = torch.argsort(times, descending=True)
        sorted_times = times[sorted_indices]
        sorted_events = events[sorted_indices]
        sorted_preds = predictions[sorted_indices]

        # Compute exponentiated risk scores.
        exp_preds = torch.exp(sorted_preds)

        # Compute the cumulative risk set sums.
        risk_set = torch.cumsum(exp_preds, dim=0)

        # Only consider subjects where an event occurred.
        event_mask = (sorted_events == 1)
        if event_mask.sum() == 0:
            loss_value = torch.tensor(0.0, device=predictions.device)
        else:
            loss_value = -torch.sum(sorted_preds[event_mask] - torch.log(risk_set[event_mask] + 1e-8))
            loss_value = loss_value / event_mask.sum()

        # Compute baseline hazards using the Breslow estimator.
        event_times = sorted_times[sorted_events == 1]
        baseline_times = torch.unique(event_times)  # Sorted in ascending order.
        baseline_hazard_list = []
        for t in baseline_times:
            at_risk = (sorted_times >= t)
            risk_sum = torch.sum(exp_preds[at_risk])
            d_t = torch.sum((sorted_times == t) & (sorted_events == 1)).float()
            baseline_hazard_t = d_t / (risk_sum + 1e-8)
            baseline_hazard_list.append(baseline_hazard_t)
        
        baseline_hazards = torch.stack(baseline_hazard_list)

        # Store the last computed baseline hazards and times.
        self.last_baseline = (baseline_times, baseline_hazards)
        
        return loss_value, baseline_times, baseline_hazards


# =============================================================================
# Unsupervised Heads
# =============================================================================

class ClusteringHead(nn.Module):
    """
    A clustering head that produces soft cluster assignments.
    
    It uses a linear layer to map inputs to logits over cluster centers,
    applies log-softmax, and by default uses KL divergence loss.
    
    Parameters:
        input_dim (int): Dimension of the input features.
        num_clusters (int): Number of clusters.
        custom_nn (nn.Module, optional): A user-defined neural network module.
        default_loss (nn.Module, optional): Loss function. Defaults to KLDivLoss.
    """
    def __init__(self, input_dim: int, num_clusters: int, custom_nn: nn.Module = None, default_loss: nn.Module = None):
        super(ClusteringHead, self).__init__()
        self.model = custom_nn if custom_nn is not None else nn.Linear(input_dim, num_clusters)
        # Default loss: KLDivLoss (expects log probabilities and target probabilities)
        self.default_loss = default_loss if default_loss is not None else nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        # Return log probabilities to be used with KLDivLoss.
        return F.log_softmax(logits, dim=1)
    
    def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.default_loss(predictions, targets)


# =============================================================================
# Unsupervised Dimension Reduction Heads
# =============================================================================

try:
    import umap
except ImportError:
    umap = None

try:
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None


class UMAPHead(nn.Module):
    """
    A wrapper for UMAP dimension reduction.
    
    Note: UMAP is not differentiable. Use the fit() and transform() methods
    for post-hoc analysis.
    """
    def __init__(self, n_components: int = 2, **umap_kwargs):
        super(UMAPHead, self).__init__()
        if umap is None:
            raise ImportError("Please install the 'umap-learn' package to use UMAPHead.")
        self.n_components = n_components
        self.umap_model = umap.UMAP(n_components=n_components, **umap_kwargs)
        self.default_loss = None

    def fit(self, X):
        self.umap_model.fit_transform(X)
    
    def transform(self, X):
        return self.umap_model.transform(X)
    
    def forward(self, x):
        raise NotImplementedError("UMAPHead does not support forward(). Use fit() and transform() instead.")


class TSNEHead(nn.Module):
    """
    A wrapper for t-SNE dimension reduction.
    
    Note: t-SNE is not differentiable. Use the fit_transform() method
    for post-hoc analysis.
    """
    def __init__(self, n_components: int = 2, **tsne_kwargs):
        super(TSNEHead, self).__init__()
        if TSNE is None:
            raise ImportError("Please install scikit-learn to use TSNEHead.")
        self.n_components = n_components
        self.tsne_model = TSNE(n_components=n_components, **tsne_kwargs)
        self.default_loss = None

    def fit_transform(self, X):
        return self.tsne_model.fit_transform(X)
    
    def forward(self, x):
        raise NotImplementedError("TSNEHead does not support forward(). Use fit_transform() instead.")


class PCAHead(nn.Module):
    """
    A wrapper for PCA dimension reduction.
    
    Note: PCA here is used as a post-hoc analysis tool and is not differentiable.
    """
    def __init__(self, n_components: int = 2, **pca_kwargs):
        super(PCAHead, self).__init__()
        if PCA is None:
            raise ImportError("Please install scikit-learn to use PCAHead.")
        self.n_components = n_components
        self.pca_model = PCA(n_components=n_components, **pca_kwargs)
        self.default_loss = None

    def fit(self, X):
        self.pca_model.fit(X)
    
    def transform(self, X):
        return self.pca_model.transform(X)
    
    def forward(self, x):
        raise NotImplementedError("PCAHead does not support forward(). Use fit() and transform() instead.")
