import numpy as np
from ttm.zero_shot_ttm import ZeroShotTTM

# Initialize TTM
ttm = ZeroShotTTM(
    context_length=512,
    prediction_length=96,
)

# Create synthetic exponential degradation
x = np.linspace(0, 5, 512)
history = 1.0 - 0.15 * (1 - np.exp(-x))

print("Last 5 history values:", history[-5:])
print("Last actual SoH:", history[-1])

# Run prediction
forecast = ttm.predict(history)

print("Forecast shape:", forecast.shape)
print("First 10 forecast values:", forecast[:10])
print("Last forecast value:", forecast[-1])
print("Mean forecast:", np.mean(forecast))
print("Min forecast:", np.min(forecast))
print("Max forecast:", np.max(forecast))