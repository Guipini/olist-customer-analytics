# %%
import duckdb
import pandas as pd
import plotly.express as px

# %%
from pathlib import Path

con = duckdb.connect()

PROJECT_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
assert (PROJECT_ROOT / "data").exists(), f"data/ not found from {Path.cwd()}"
DATA_DIR = PROJECT_ROOT / "data"

# Register each CSV as a view so SQL queries can reference them by name
tables = {
    "customers": DATA_DIR / "olist_customers_dataset.csv",
    "orders": DATA_DIR / "olist_orders_dataset.csv",
    "order_items": DATA_DIR / "olist_order_items_dataset.csv",
    "payments": DATA_DIR / "olist_order_payments_dataset.csv",
    "reviews": DATA_DIR / "olist_order_reviews_dataset.csv",
    "products": DATA_DIR / "olist_products_dataset.csv",
    "sellers": DATA_DIR / "olist_sellers_dataset.csv",
    "geolocation": DATA_DIR / "olist_geolocation_dataset.csv",
    "category_translation": DATA_DIR / "product_category_name_translation.csv",
}

for name, path in tables.items():
    con.sql(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_csv_auto('{path}')")

# %%
master_query = """
WITH payments_agg AS (
    SELECT
        order_id,
        SUM(payment_value) AS total_payment,
        COUNT(*) AS payment_count,
        MODE(payment_type) AS primary_payment_type,
        MAX(payment_installments) AS max_installments
    FROM payments
    GROUP BY order_id
)
SELECT
    o.order_id,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    c.customer_unique_id,
    c.customer_city,
    c.customer_state,
    oi.order_item_id,
    oi.product_id,
    oi.seller_id,
    oi.price,
    oi.freight_value,
    p.product_category_name,
    t.product_category_name_english AS category_english,
    p.product_weight_g,
    p.product_length_cm,
    p.product_height_cm,
    p.product_width_cm,
    s.seller_city,
    s.seller_state,
    pa.total_payment,
    pa.payment_count,
    pa.primary_payment_type,
    pa.max_installments
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN products p ON oi.product_id = p.product_id
LEFT JOIN category_translation t ON p.product_category_name = t.product_category_name
LEFT JOIN sellers s ON oi.seller_id = s.seller_id
LEFT JOIN payments_agg pa ON o.order_id = pa.order_id
"""

# %%
df_master = con.sql(master_query).df()

# Parse datetime columns
datetime_cols = [
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for col in datetime_cols:
    df_master[col] = pd.to_datetime(df_master[col])

# %%
df_reviews = con.sql("""
    SELECT review_id, order_id, review_score, 
           review_comment_title, review_comment_message, 
           review_creation_date, review_answer_timestamp
    FROM reviews
""").df()

df_reviews["review_creation_date"] = pd.to_datetime(df_reviews["review_creation_date"])
df_reviews["review_answer_timestamp"] = pd.to_datetime(df_reviews["review_answer_timestamp"])

# %%
df_geo = con.sql("""
    SELECT geolocation_zip_code_prefix, 
           AVG(geolocation_lat) AS lat,
           AVG(geolocation_lng) AS lng,
           MODE(geolocation_city) AS city,
           MODE(geolocation_state) AS state
    FROM geolocation
    GROUP BY geolocation_zip_code_prefix
""").df()

# %%
# Row counts
print(f"Master: {len(df_master):,} rows")
print(f"Reviews: {len(df_reviews):,} rows")
print(f"Geolocation: {len(df_geo):,} rows")

# Null checks on critical columns
print(df_master[["order_id", "customer_unique_id", "price"]].isnull().sum())

# Order status distribution
print(df_master["order_status"].value_counts())

# Unique customers
n_unique = df_master["customer_unique_id"].nunique()
n_orders= df_master["order_id"].nunique()
print(f"Unique customers: {n_unique:,}, Unique orders: {n_orders:,}")

# %%
df_master.head()

# %%
df_master.to_parquet(DATA_DIR / "master.parquet", index=False)
df_reviews.to_parquet(DATA_DIR / "reviews.parquet", index=False)
df_geo.to_parquet(DATA_DIR / "geo.parquet", index=False)

print(f"Exported: master.parquet ({len(df_master):,} rows), "
      f"reviews.parquet ({len(df_reviews):,} rows), "
      f"geo.parquet ({len(df_geo):,} rows)")

# %%
con.close()
