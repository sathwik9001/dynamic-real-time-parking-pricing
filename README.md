# dynamic-real-time-parking-pricing
!pip install pathway bokeh ipywidgets
import pandas as pd
import numpy as np
from google.colab import files
uploaded = files.upload()

df = pd.read_csv("dataset.csv")
df
df['timestamp'] = pd.to_datetime(
    df['LastUpdatedDate'].astype(str) + ' ' + df['LastUpdatedTime'].astype(str),
    errors='coerce',
    utc=True
)

df = df.dropna(subset=['timestamp'])  # Drop invalid timestamps

# ✅ Assign location_id column directly (instead of renaming)
df['location_id'] = df['SystemCodeNumber'].astype(str)

# ✅ Rename remaining columns
df.rename(columns={
    'Occupancy': 'occupancy',
    'Capacity': 'capacity',
    'QueueLength': 'queue_length',
    'TrafficConditionNearby': 'traffic_level',
    'IsSpecialDay': 'is_special_day',
    'VehicleType': 'vehicle_type'
}, inplace=True)

# ✅ Map traffic conditions
traffic_map = {'low': 1, 'medium': 2, 'high': 3}
df['traffic_level'] = df['traffic_level'].map(traffic_map).fillna(1)

# ✅ Sort by time
df = df.sort_values(by='timestamp')
# Base demand-based price (Model 2)
def calculate_dynamic_price(
    base_price,
    occupancy,
    capacity,
    queue_length,
    traffic_level,
    is_special_day,
    vehicle_type
):
    weights = {
        'occupancy': 1.0,
        'queue': 0.8,
        'traffic': 0.6,
        'special': 0.5,
        'vehicle': {'car': 1.0, 'bike': 0.5, 'truck': 1.5}
    }

    vehicle_weight = weights['vehicle'].get(str(vehicle_type).lower(), 1.0)

    demand = (
        weights['occupancy'] * (occupancy / max(capacity, 1)) +
        weights['queue'] * queue_length -
        weights['traffic'] * traffic_level +
        weights['special'] * is_special_day +
        vehicle_weight
    )

    demand = max(min(demand, 10), 0)
    normalized_demand = demand / 10
    price = base_price * (1 + normalized_demand)

    return round(min(max(price, base_price * 0.5), base_price * 2.0), 2)
# Haversine formula to compute distance between lat/lon
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

# Distance matrix between parking lots
location_coords = df.groupby('location_id')[['Latitude', 'Longitude']].first()
location_ids = location_coords.index.tolist()

distances = {
    loc1: {
        loc2: haversine(lat1, lon1, lat2, lon2)
        for loc2, (lat2, lon2) in location_coords.iterrows()
    }
    for loc1, (lat1, lon1) in location_coords.iterrows()
}
def calculate_competitive_price(
    current_loc,
    timestamp,
    base_price,
    occupancy,
    capacity,
    queue_length,
    traffic_level,
    is_special_day,
    vehicle_type,
    current_prices,
    distances,
    radius=500
):
    price = calculate_dynamic_price(
        base_price,
        occupancy,
        capacity,
        queue_length,
        traffic_level,
        is_special_day,
        vehicle_type
    )

    nearby = [loc for loc, dist in distances[current_loc].items() if loc != current_loc and dist <= radius]
    competitor_prices = [current_prices.get(loc, price) for loc in nearby]

    if not competitor_prices:
        return price

    avg_competitor_price = np.mean(competitor_prices)

    # Apply adjustments
    if occupancy >= capacity * 0.95 and price > avg_competitor_price:
        price *= 0.95  # reduce to compete
    elif price < avg_competitor_price:
        price *= 1.05  # raise since others are expensive

    return round(min(max(price, base_price * 0.5), base_price * 2.0), 2)
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.io import push_notebook
import pandas as pd

output_notebook()

top_locations = df['location_id'].value_counts().head(3).index.tolist()
sources = {}
colors = ['red', 'green', 'blue']

p = figure(title="Real-Time Dynamic Parking Prices",
           x_axis_type="datetime",
           x_axis_label="Time", y_axis_label="Price ($)",
           width=800, height=400)

for i, loc in enumerate(top_locations):
    sources[loc] = ColumnDataSource(data=dict(x=[], y=[]))
    p.line(x='x', y='y', source=sources[loc], line_width=2,
           legend_label=str(loc), color=colors[i])

p.legend.location = "top_left"
p.y_range.start = 5
p.y_range.end = 25
p.xaxis.major_label_orientation = 0.5

hover = HoverTool(
    tooltips=[("Time", "@x{%F %T}"), ("Price", "@y{$0.00}")],
    formatters={'@x': 'datetime'},
    mode='vline'
)
p.add_tools(hover)

handle = show(p, notebook_handle=True)
current_prices = {}



import asyncio

async def simulate_streaming():
    count = 0
    for _, row in df.iterrows():
        loc = str(row['location_id'])
        if loc not in sources:
            continue

        price = calculate_competitive_price(
            current_loc=loc,
            timestamp=row['timestamp'],
            base_price=10,
            occupancy=row['occupancy'],
            capacity=row['capacity'],
            queue_length=row['queue_length'],
            traffic_level=row['traffic_level'],
            is_special_day=row['is_special_day'],
            vehicle_type=row['vehicle_type'],
            current_prices=current_prices,
            distances=distances
        )

        print(f"[{row['timestamp']}] Location: {loc} → Price: ${price}")

        current_prices[loc] = price
        sources[loc].stream({'x': [row['timestamp']], 'y': [price]}, rollover=300)
        push_notebook(handle=handle)

        count += 1
        if count >= 100:
            break

        await asyncio.sleep(0.2)

await simulate_streaming()


current_prices = {}

def stream_prices():
    count = 0
    for _, row in df.iterrows():
        while not play_toggle.value:
            time.sleep(0.1)

        loc = str(row['location_id'])
        if loc not in sources:
            continue

        try:
            price = calculate_competitive_price(
                current_loc=loc,
                timestamp=row['timestamp'],
                base_price=10,
                occupancy=row['occupancy'],
                capacity=row['capacity'],
                queue_length=row['queue_length'],
                traffic_level=row['traffic_level'],
                is_special_day=row['is_special_day'],
                vehicle_type=row['vehicle_type'],
                current_prices=current_prices,
                distances=distances
            )
            # Corrected indentation for the following lines
            print(f"[{row['timestamp']}] Location: {loc} → Price: ${price}")

            current_prices[loc] = price
            sources[loc].stream({'x': [row['timestamp']], 'y': [price]}, rollover=300)
            push_notebook(handle=handle)
            count += 1

            if count % 50 == 0:
                print(f"{count} rows streamed...")

            time.sleep(speed_slider.value)

        except Exception as e:
            print(f"Error: {e}")

# 🔁 Start in background
threading.Thread(target=stream_prices).start()
