import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity


# -------------------------------
# STEP 1: Load and preprocess data
# -------------------------------
def load_and_prepare_data(file_path):
    # Load CSV
    df = pd.read_csv(file_path)

    # Convert pickup_time to datetime
    df['pickup_time'] = pd.to_datetime(df['pickup_time'])

    # Convert time to numeric (hour of day)
    df['time'] = df['pickup_time'].dt.hour + df['pickup_time'].dt.minute / 60.0

    # Select features
    data = df[['time', 'pickup_location', 'dropoff_location']].values

    return data


# -------------------------------
# STEP 2: Train KDE model
# -------------------------------
def train_kde(data):
    kde = KernelDensity(kernel='gaussian', bandwidth=1.0)
    kde.fit(data)
    return kde


# -------------------------------
# STEP 3: Generate new ride
# -------------------------------
def generate_ride(kde):
    sample = kde.sample(1)[0]

    # Extract values
    time = sample[0]%24
    pickup = int(round(sample[1]))
    drop = int(round(sample[2]))

    return {
        "time": time,
        "pickup_location": pickup,
        "dropoff_location": drop
    }


# -------------------------------
# MAIN TEST
# -------------------------------
if __name__ == "__main__":
    # Load data
    data = load_and_prepare_data("data/nyc-sample-demand.csv")

    print("Sample processed data:")
    print(data[:5])

    # Train KDE
    kde = train_kde(data)
    print("\nKDE model trained successfully!")

    # Generate sample rides
    print("\nGenerated rides:")
    for _ in range(5):
        ride = generate_ride(kde)
        print(ride)