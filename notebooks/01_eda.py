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

# %% [markdown]
# ### Repeat-purchase rate
#
# **Insight**: The Olist marketplace is dominated by one-time transactions. 96.9% of customers (92,507 of 95,420) place exactly one order across the 2-year window; only 3.1% return for a second purchase or more. This is the single most important constraint on Phase 5's segmentation strategy.
#
# **Method**: Counted distinct `order_id`s per `customer_unique_id` (not `customer_id`, which is regenerated per order), then binned the long tail into {1, 2, 3, 4+} so the chart reads at a glance. Reported as % of total customers with absolute counts shown as text labels for scale-anchoring.
#
# **Trade-off**: Reporting on customer count flattens the visual difference between 2-order and 4+-order buyers (both ~1%). The Phase 5 segmentation will need a complementary view focused on revenue concentration (Pareto on revenue per tier) where the repeater minority's outsized value will surface.
#
# **Gotcha**: 96.9% of customers being one-timers does not mean 96.9% of revenue comes from one-timers. Repeat buyers spend more per order on average, so their revenue share will exceed 3.1% (likely meaningfully so). Don't conflate customer-share with revenue-share in the README copy. The revenue-share question is Phase 5's job.

# %%
# Repeat-purchase distribution. Use customer_unique_id (not customer_id);
# nunique collapses multi-item orders into 1 per order
orders_per_customer = df.groupby("customer_unique_id")["order_id"].nunique()
n_customers = len(orders_per_customer)

freq_binned = (
    orders_per_customer.apply(lambda n: str(n) if n < 4 else "4+")
    .value_counts()
    .reindex(["1", "2", "3", "4+"], fill_value=0)
)
freq_pct = (freq_binned / n_customers * 100).round(1)

print(f"One-time buyers: {freq_pct['1']}% ({freq_binned['1']:,} of {n_customers:,})")

fig_repeat_purchase = px.bar(
    x=freq_binned.index,
    y=freq_pct.values,
    text=[f"{c:,}" for c in freq_binned.values],
    labels={"x": "Orders per Customer", "y": "Percentage of Customers"},
    title="Repeat-purchase rate"
)
fig_repeat_purchase.update_traces(textposition="outside")
fig_repeat_purchase.show()

# %% [markdown]
# ### Review score distribution
#
# **Insight**: The review distribution is bimodal. 5-star is the primary mode (57.8%), but 1-star is the secondary mode at 11.5%, higher than both 2-star (3.2%) and 3-star (8.2%). Customers polarize into delighted and disappointed camps with a thin middle. The 99,224 reviews break down as 5-star: 57,328 / 4-star: 19,142 / 3-star: 8,179 / 2-star: 3,151 / 1-star: 11,424.
#
# **Method**: Bar chart of all five ordinal score levels with percentages on the y-axis and absolute counts as text labels. Used `df_reviews` directly (one row per review) rather than the order_items-level `df` to avoid inflating the count for multi-item orders. Reported mode (5-star) and IQR (Q1=4, Q3=5) instead of mean and standard deviation, since review scores are ordinal and the bimodal shape makes any average misleading.
#
# **Trade-off**: Mode + IQR captures the positive skew (75% of reviews are 4 or 5 stars) but undersells the weight on the negative tail. The 1-star secondary peak (11.5%, more than 11,000 reviews) is the actionable signal, not the 5-star pile-up. Always report the bimodal shape alongside the central tendency, not instead of it.
#
# **Gotcha**: A naive reading of "57.8% gave 5 stars" reads as uniformly high satisfaction. The actual story is more polarized: roughly 1 in 9 customers gives the lowest possible score. The README copy should frame the experience as polarized rather than uniformly positive. Phase 2's delivery analysis and Phase 8's NLP topic modeling will explain why the 1-star peak exists.

# %%
# Review score distribution. Scores are ordinal (1-5), so plot as bar
# (not histogram). Use df_reviews directly: one row per review.
score_counts = (
    df_reviews["review_score"].dropna().astype(int)
    .value_counts()
    .reindex([1, 2, 3, 4, 5], fill_value=0)
)
n_reviews = score_counts.sum()
score_pct = (score_counts / n_reviews * 100).round(1)

print(f"Mode: {score_counts.idxmax()}-star ({score_pct[score_counts.idxmax()]}%)")
q1= score_counts.cumsum().searchsorted(n_reviews * 0.25)+1
q3 = score_counts.cumsum().searchsorted(n_reviews * 0.75)+1
print(f"IQR: Q1={q1}, Q3={q3}")

fig_review_dist = px.bar(
    x=score_counts.index.astype(str),
    y=score_pct.values,
    text=[f"{c:,}" for c in score_counts.values],
    labels={"x": "Review Score", "y": "Percentage of Reviews"},
    title="Review Score Distribution"
)
fig_review_dist.update_traces(textposition="outside")
fig_review_dist.show()

# %% [markdown]
# ### Delivery time distribution
#
# **Insight**: Olist over-promises delivery time aggressively as a customer-experience strategy. The median order arrives 12 days before its estimated delivery date, and only 6.8% of delivered orders are actually late (6,534 of 96,470). The marketplace runs on under-promise, over-deliver: estimates carry roughly two weeks of safety buffer, which sets up a positive surprise for most customers.
#
# **Method**: Histogram of `delay_days = order_delivered_customer_date - order_estimated_delivery_date`, computed on the delivered-orders subset and de-duplicated to one row per order (the master DataFrame is at the order_items level, so multi-item orders would otherwise be over-counted). Negative values mean early, positive mean late. A vertical line at zero separates the early/on-time region from the late region. Reported median (not mean) and percentage late, since the long right tail of extreme delays would skew a mean.
#
# **Trade-off**: Aggregating to a single nationwide distribution loses the per-state and per-seller structure. Northern Brazilian states (Amazonas, Pará, Roraima) likely have substantially longer transit times than the southeastern hubs (São Paulo, Rio de Janeiro), and individual seller reliability varies widely. Phase 2's deeper delivery analysis will decompose those dimensions; Phase 1 gives only the marketplace-wide overview.
#
# **Gotcha**: A 6.8% late rate sounds excellent in isolation, but the bimodal review distribution from the previous chart shows 11.5% of customers leave 1-star reviews. Late delivery alone cannot account for the full negative tail. The remaining dissatisfaction must come from other sources: product mismatch, packaging damage, seller communication, or post-purchase support. Phase 8's NLP analysis on the Portuguese review text will eventually decompose what those 1-star reviews actually complain about.

# %%
# Delivery delay distribution. delay_days = actual - estimated.
# Negative means early, positive means late. Vertical line at 0
# separates on-time/early from late. Use the `delivered` DataFrame
# (already filtered to delivered orders in cell 1).
delay = delivered.drop_duplicates("order_id")["delay_days"].dropna()

n_total = len(delay)
n_late = int((delay > 0).sum())
median_delay = float(delay.median())
pct_late = round(n_late / n_total * 100, 1)

print(f"Total delivered orders: {n_total:,}")
print(f"Median delay: {median_delay:.1f} days (negative = early)")
print(f"Late orders: {n_late:,} ({pct_late}%)")

fig_delivery_delay = px.histogram(
    delay,
    nbins=80,
    labels={"value": "Delivery Delay (days)", "count": "Number of Orders"},
    title="Delivery delay (days late vs Olist's estimated delivery)"
) 
fig_delivery_delay.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="On-time/late boundary")
fig_delivery_delay.show()
