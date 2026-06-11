import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "uploads" / "generated_sales_dataset.xlsx"
OUT.parent.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

n_rows = 1200

# SKU / product catalog
skus = [f"SKU{str(i).zfill(4)}" for i in range(1, 101)]
products = [f"Product {i}" for i in range(1, 101)]
categories = ["Electronics", "Home", "Office", "Clothing", "Grocery"]
regions = ["North", "South", "East", "West"]
stores = [f"STORE{str(i).zfill(3)}" for i in range(1, 51)]

start_date = datetime(2023, 1, 1)
end_date = datetime(2025, 12, 31)
delta_days = (end_date - start_date).days

rows = []
for i in range(n_rows):
    sku_idx = np.random.randint(0, len(skus))
    sku = skus[sku_idx]
    product = products[sku_idx]
    category = np.random.choice(categories, p=[0.25, 0.2, 0.2, 0.2, 0.15])
    region = np.random.choice(regions)
    store = np.random.choice(stores)
    # random date
    date = start_date + pd.to_timedelta(np.random.randint(0, delta_days + 1), unit="d")
    quantity = int(np.random.poisson(lam=8))
    price = round(float(np.random.uniform(5.0, 500.0)), 2)
    revenue = round(quantity * price, 2)
    promotion = np.random.choice([True, False], p=[0.08, 0.92])
    order_id = f"ORD{100000 + i}"

    rows.append({
        "order_date": date.strftime("%Y-%m-%d"),
        "order_id": order_id,
        "sku": sku,
        "product_name": product,
        "category": category,
        "region": region,
        "store_id": store,
        "quantity_sold": quantity,
        "unit_price": price,
        "revenue": revenue,
        "promotion": promotion,
    })

# Create DataFrame and save as Excel
df = pd.DataFrame(rows)
# Ensure at least 1000 rows
if len(df) < 1000:
    raise SystemExit("Dataset too small")

df.to_excel(OUT, index=False)
print(f"Wrote {len(df)} rows to {OUT}")
