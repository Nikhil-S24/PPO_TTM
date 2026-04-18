import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity


# -------------------------------
# STEP 1: Load and preprocess data
# -------------------------------
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)

    df['pickup_time'] = pd.to_datetime(df['pickup_time'])

    # Convert time to continuous hour
    df['time'] = df['pickup_time'].dt.hour + df['pickup_time'].dt.minute / 60.0

    # Use only required features
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
# STEP 3: Demand intensity (🔥 FIXED)
# -------------------------------
def get_demand_count(hour):
    """
    LOW demand → creates competition → PPO learns better
    """

    hour = int(hour)

    if 7 <= hour <= 10:        # Morning rush
        return np.random.randint(3, 6)

    elif 17 <= hour <= 20:     # Evening rush
        return np.random.randint(3, 6)

    elif 0 <= hour <= 5:       # Night
        return np.random.randint(1, 3)

    else:                      # Normal hours
        return np.random.randint(2, 4)


# -------------------------------
# STEP 4: Generate rides
# -------------------------------
def generate_rides(kde, current_time):
    rides = []

    current_hour = current_time.hour
    demand_count = get_demand_count(current_hour)

    for _ in range(demand_count):
        sample = kde.sample(1)[0]

        # Cycle time (important for long simulation)
        time = sample[0] % 24

        # Extract locations
        pickup = int(round(sample[1]))
        drop = int(round(sample[2]))

        # 🔥 IMPORTANT: match your region (10 zones)
        pickup = max(0, min(9, pickup))
        drop = max(0, min(9, drop))

        # Avoid same pickup/drop
        if pickup == drop:
            drop = (drop + 1) % 10

        rides.append({
            "time": time,
            "pickup_location": pickup,
            "dropoff_location": drop
        })

    return rides


# -------------------------------
# MAIN TEST
# -------------------------------
if __name__ == "__main__":
    data = load_and_prepare_data("data/nyc-sample-demand.csv")

    print("Sample processed data:")
    print(data[:5])

    kde = train_kde(data)
    print("\nKDE model trained successfully!")

    import datetime
    current_time = datetime.datetime.now()

    print("\nGenerated rides:")
    rides = generate_rides(kde, current_time)

    for ride in rides:
        print(ride)