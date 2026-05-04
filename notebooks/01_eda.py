# %%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

PROJECT_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
assert (PROJECT_ROOT / "data").exists(), f"data/ not found from {Path.cwd()}"
DATA_DIR = PROJECT_ROOT / "data"

df_master = pd.read_parquet(DATA_DIR / "master.parquet")
df_reviews = pd.read_parquet(DATA_DIR / "reviews.parquet")
df_geo = pd.read_parquet(DATA_DIR / "geo.parquet")

# Merge reviews scores onto master
df = df_master.merge(
    df_reviews[["order_id", "review_score"]],
    on="order_id",
    how="left"
)

# %%
# Filter to delivered orders for time-based calculations
delivered = df[df["order_status"] == "delivered"].copy()

delivered["delivered_days"] = (
    delivered["order_delivered_customer_date"] - delivered["order_purchase_timestamp"]
).dt.days

delivered["delay_days"] = (
    delivered["order_delivered_customer_date"] - delivered["order_estimated_delivery_date"]
).dt.days

delivered["is_late"] = delivered["delay_days"] > 0

# Time feature (on full df)
df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M")

# %%
# Missing value % per column
null_pct = (df.isna().mean() * 100).sort_values(ascending=False)
null_pct = null_pct[null_pct > 0]

fig = px.bar(
    x=null_pct.values, y=null_pct.index,
    orientation="h",
    labels={"x": "Missing %", "y": "Column"},
    title="Missing Values by Column"
)
fig.show()

# %%
status_counts = df["order_status"].value_counts()
fig_pie = px.pie(
    values=status_counts.values, names=status_counts.index,
    title="Order Status Distribution")
fig_pie.show()

# %%
monthly = df.groupby("order_month")["order_id"].nunique().reset_index()
monthly.columns = ["month", "orders"]
monthly["month"] = monthly["month"].astype(str) # Plotly cannot JSON-serialize Period in Jupyter (Marimo handled it implicitly)

fig_line = px.line(
    monthly, x="month", y="orders",
    title="Monthly Order Volume"
)
fig_line.update_xaxes(tickangle=45)
fig_line.show()
