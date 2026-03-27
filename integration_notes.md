## Model training summary

- Target variable: `price_jod`
- Predicted value: housing price in Jordanian dinars

### Input features
1. `area_sqm`
2. `bedrooms`
3. `floor`
4. `age_years`
5. `distance_to_center_km`

### Training configuration
- Epochs: `100`
- Learning rate: `0.01`
- Optimizer: `Adam`
- Loss function: `MSELoss`

### Training outcome
- Loss decreased over training.
- Final reported loss: `1,943,051,648.0000`

### Observation
- Loss decreased steadily but slowly, from about `1.9505e9` at epoch 10 to `1.9430e9` at epoch 100.
