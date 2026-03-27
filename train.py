"""
Integration 2 — PyTorch: Housing Price Prediction
Module 2 — Programming for AI & Data Science

Complete each section below. Remove the TODO: comments and pass statements
as you implement each section. Do not change the overall structure.

Before running this script, install PyTorch:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn


# ─── Model Definition ─────────────────────────────────────────────────────────

class HousingModel(nn.Module):
    """Neural network for predicting housing prices from property features.

    Architecture: Linear(5, 32) -> ReLU -> Linear(32, 1)
    """

    def __init__(self):
        """Define the model layers."""
        super().__init__()
        # TODO: Define three layers as attributes:
        #   self.layer1 = nn.Linear(5, 32)   — 5 input features → 32 hidden units
        #   self.relu   = nn.ReLU()           — activation function
        #   self.layer2 = nn.Linear(32, 1)    — 32 hidden → 1 output (price prediction)
        self.layer1 = nn.Linear(5, 32)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(32, 1)

    def forward(self, x):
        """Define the forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 5).

        Returns:
            torch.Tensor: Predictions of shape (N, 1).
        """
        # TODO: Pass x through layer1, then relu, then layer2
        # TODO: Return the output
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# ─── Main Training Script ─────────────────────────────────────────────────────

def main():
    """Load data, train HousingModel, and save predictions."""

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    # TODO: Load data/housing.csv using pd.read_csv
    # TODO: Print the shape of the DataFrame
    df = pd.read_csv('data/housing.csv')
    print(f"DataFrame shape: {df.shape}")
    # ── 2. Separate Features and Target ──────────────────────────────────────
    feature_cols = ['area_sqm', 'bedrooms', 'floor', 'age_years', 'distance_to_center_km']
    # TODO: X = df[feature_cols]
    # TODO: y = df[['price_jod']]   — use double brackets to keep shape (N, 1)
    X = df[feature_cols]
    y = df[['price_jod']]
    # ── 3. Standardize Features ───────────────────────────────────────────────
    # TODO: X_mean = X.mean()
    # TODO: X_std  = X.std()
    # TODO: X_scaled = (X - X_mean) / X_std
    # Why: features have very different scales; standardization ensures
    #      gradient updates are balanced across all input dimensions.

    X_mean = X.mean()
    X_std = X.std()
    X_scaled = (X - X_mean) / X_std


    # ── 4. Convert to Tensors ─────────────────────────────────────────────────
    # TODO: X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    # TODO: y_tensor = torch.tensor(y.values,        dtype=torch.float32)
    # TODO: Print X_tensor.shape and y_tensor.shape
    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    print(f"X_tensor shape: {X_tensor.shape}")
    print(f"y_tensor shape: {y_tensor.shape}")

    # ── 5. Instantiate Model, Loss, and Optimizer ─────────────────────────────
    # TODO: model     = HousingModel()
    # TODO: criterion = nn.MSELoss()
    # TODO: optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model = HousingModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # ── 6. Training Loop ──────────────────────────────────────────────────────
    num_epochs = 100
    # TODO: for epoch in range(num_epochs):
    #     Forward pass:  predictions = model(X_tensor)
    #     Compute loss:  loss = criterion(predictions, y_tensor)
    #     Zero grads:    optimizer.zero_grad()
    #     Backward:      loss.backward()
    #     Update:        optimizer.step()
    #     Print every 10 epochs: f"Epoch {epoch:3d}: Loss = {loss.item():.4f}"
    for epoch in range(num_epochs):
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}: Loss = {loss.item():.4f}")
    # ── 7. Save Predictions ───────────────────────────────────────────────────
    # TODO: Generate predictions (wrap in torch.no_grad() for good practice)
    # TODO: Convert predictions and actuals to numpy arrays
    # TODO: Build a DataFrame with columns 'actual' and 'predicted'
    # TODO: Save to predictions.csv with index=False
    # TODO: Print "Saved predictions.csv"
    with torch.no_grad():
        predictions = model(X_tensor)
    predictions_np = predictions.numpy()
    y_np = y_tensor.numpy()
    df_predictions = pd.DataFrame({'actual': y_np.flatten(), 'predicted': predictions_np.flatten()})
    df_predictions.to_csv('predictions.csv', index=False)
    print("Saved predictions.csv")


if __name__ == "__main__":
    main()
