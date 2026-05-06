import yaml
from simulator.region import CyclicZoneGraph
from simulator.job import Job
from kde_model import load_and_prepare_data, train_kde, generate_rides
import datetime
import random

with open("configs/nyc.yaml", "r") as f:
    config = yaml.safe_load(f)

region = CyclicZoneGraph(config["city"])
print("Nodes in region:", len(region.map))

data = load_and_prepare_data(config["demand"])
kde = train_kde(data)

t = datetime.datetime.strptime(config["start t"], "%Y/%m/%d %H:%M:%S")
arrived = 0
skipped = 0

for _ in range(100):
    for ride in generate_rides(kde, t):
        data = {
            "pickup_location": ride["pickup_location"],
            "dropoff_location": ride["dropoff_location"],
            "pickup_time": t.strftime("%Y-%m-%d %H:%M:%S"),
            "dropoff_time": t.strftime("%Y-%m-%d %H:%M:%S"),
            "distance": random.uniform(1, 10),
            "fare": random.uniform(5, 30),
        }
        job = Job(data, random.randint(0, 1000000), region)
        d_conn, t_conn = job.pickup_location.to(job.dropoff_location)
        if d_conn == float("inf") or t_conn == float("inf"):
            skipped += 1
        else:
            arrived += 1
    t += datetime.timedelta(hours=1)

print(f"Arrived: {arrived}, Skipped: {skipped}")
