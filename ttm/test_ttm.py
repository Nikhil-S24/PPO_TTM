import numpy as np
from ttm.zero_shot_ttm import ZeroShotTTM

# Load model
ttm = ZeroShotTTM(
    context_length=512,
    prediction_length=96,
)

# Create artificial degradation pattern
# Simulate battery slowly degrading from 1.0 to 0.85
history = np.linspace(1.0, 0.85, 512)

print("Input History (last 5 values):", history[-5:])

# Run prediction
forecast = ttm.predict(history)

print("Forecast shape:", forecast.shape)
print("First 10 forecast values:", forecast[:10])
print("Min forecast value:", np.min(forecast))
print("Max forecast value:", np.max(forecast))