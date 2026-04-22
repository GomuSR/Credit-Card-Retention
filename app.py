import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Retention Intelligence", layout="wide")

# -----------------------------
# LOAD SAVED MODEL + SCALER
# -----------------------------
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

# ----------------------------
# CONSTANTS
# -----------------------------
feature_cols = [
    "LIMIT_BAL",
    "MAX_DELAY",
    "DELAY_WT_AVG",
    "BILL_WT_AVG",
    "PAY_BILL_RATIO"
]

delay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
pay_cols = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

required_cols = [
    "LIMIT_BAL",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

# -----------------------------
# HELPERS
# -----------------------------
def load_data(file):
    if isinstance(file, str):
        if file.endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)

    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def feature_engineering(df):
    df_fe = df.copy()

    # Delay cleanup: anything <= 0 treated as no delay
    delay_clean = df_fe[delay_cols].clip(lower=0)

    # Max delay
    df_fe["MAX_DELAY"] = delay_clean.max(axis=1)

    # Weighted delay average
    df_fe["DELAY_WT_AVG"] = (
        4 * delay_clean["PAY_0"] +
        3 * delay_clean["PAY_2"] +
        2 * delay_clean["PAY_3"] +
        1 * delay_clean["PAY_4"] +
        1 * delay_clean["PAY_5"] +
        1 * delay_clean["PAY_6"]
    ) / 12

    # Clean individual bill values first
    df_fe[bill_cols] = df_fe[bill_cols].clip(lower=0)

    # Weighted bill average
    df_fe["BILL_WT_AVG"] = (
        4 * df_fe["BILL_AMT1"] +
        3 * df_fe["BILL_AMT2"] +
        2 * df_fe["BILL_AMT3"] +
        1 * df_fe["BILL_AMT4"] +
        1 * df_fe["BILL_AMT5"] +
        1 * df_fe["BILL_AMT6"]
    ) / 12

    # Weighted payment average
    df_fe["PAY_WT_AVG"] = (
        4 * df_fe["PAY_AMT1"] +
        3 * df_fe["PAY_AMT2"] +
        2 * df_fe["PAY_AMT3"] +
        1 * df_fe["PAY_AMT4"] +
        1 * df_fe["PAY_AMT5"] +
        1 * df_fe["PAY_AMT6"]
    ) / 12

    # Payment to bill ratio
    df_fe["PAY_BILL_RATIO"] = np.where(
        df_fe["BILL_WT_AVG"] == 0,
        0,
        df_fe["PAY_WT_AVG"] / df_fe["BILL_WT_AVG"]
    )

    # Cap extreme ratios so clustering does not get hijacked
    df_fe["PAY_BILL_RATIO"] = df_fe["PAY_BILL_RATIO"].clip(lower=0, upper=5)

    return df_fe


def assign_segments(cluster_summary):
    """
    Assign business-friendly names based on cluster behavior.
    This is rule-based on cluster averages, so the labels remain
    meaningful even if cluster numbers shift.
    """
    summary = cluster_summary.copy()

    labels = {}
    used_names = set()

    for idx, row in summary.iterrows():
        bill = row["BILL_WT_AVG"]
        ratio = row["PAY_BILL_RATIO"]
        delay = row["DELAY_WT_AVG"]
        limit_bal = row["LIMIT_BAL"]

        if delay >= 0.8:
            label = "High Risk Customers"
        elif bill > 100000 and ratio < 0.15:
            label = "High Value Revolvers"
        elif bill < 15000 and ratio >= 0.8:
            label = "Low Utilization Customers"
        else:
            label = "Stable Revenue Customers"

        # prevent duplicate labels from colliding awkwardly
        if label in used_names:
            label = f"{label} ({idx})"

        used_names.add(label)
        labels[idx] = label

    return labels


# -----------------------------
# UI
# -----------------------------
st.title("Customer Retention Intelligence for Credit Cards")

st.markdown("""
Segment credit card customers to identify where retention spend actually makes sense.

This app uses:
- spending behavior
- repayment pattern
- payment delay severity

to classify customers into actionable retention segments.
""")

st.markdown("""
### What this helps answer
- Which customers should be **aggressively retained**
- Which customers are **stable and worth keeping**
- Which customers are **too risky or too low-value** for retention spending
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Ignore", "Low value / high risk")

with col2:
    st.metric("Retain", "Stable customers")

with col3:
    st.metric("Aggressively Retain", "High value / revenue potential")

st.info("""
**Accepted file types:** CSV, XLSX

**Required columns (one row = one customer):**

- LIMIT_BAL → Credit limit assigned  
- PAY_0 → Recent month repayment status  
- PAY_2 to PAY_6 → Past repayment status (months back)  
- BILL_AMT1 to BILL_AMT6 → Monthly bill amounts  
- PAY_AMT1 to PAY_AMT6 → Monthly payment amounts 

Each row should represent one customer.
""")

st.subheader("Try the app")

use_sample = st.button("Try with Sample Data")
uploaded_file = st.file_uploader("Or upload your own customer file", type=["csv", "xlsx"])

input_df = None

if use_sample:
    input_df = load_data("Credit_Card_Data.xlsx")
    st.success("Sample dataset loaded successfully.")

elif uploaded_file is not None:
    input_df = load_data(uploaded_file)

if input_df is not None:
    df = input_df.copy()

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        df_fe = feature_engineering(df)

        X = df_fe[feature_cols].copy()
        X_scaled = scaler.transform(X)
        df_fe["Cluster"] = kmeans.predict(X_scaled)

        cluster_summary = df_fe.groupby("Cluster")[feature_cols].mean().round(2)
        cluster_summary["Customer_Count"] = df_fe.groupby("Cluster").size()

        # dynamic business segment naming
        cluster_name_map = assign_segments(cluster_summary)
        df_fe["Segment"] = df_fe["Cluster"].map(cluster_name_map)
        cluster_summary["Segment"] = cluster_summary.index.map(cluster_name_map)

        # top metrics
        st.subheader("Portfolio Snapshot")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Customers", f"{len(df_fe):,}")
        m2.metric("Average Bill", f"{int(df_fe['BILL_WT_AVG'].mean()):,}")
        m3.metric("Average Pay/Bill Ratio", round(df_fe["PAY_BILL_RATIO"].mean(), 2))

        st.subheader("Segment Summary")
        st.dataframe(
            cluster_summary[
                [
                    "Segment",
                    "LIMIT_BAL",
                    "MAX_DELAY",
                    "DELAY_WT_AVG",
                    "BILL_WT_AVG",
                    "PAY_BILL_RATIO",
                    "Customer_Count"
                ]
            ]
        )

        st.subheader("Customer Count by Segment")
        bar_data = cluster_summary.set_index("Segment")["Customer_Count"]
        st.bar_chart(bar_data)

        st.subheader("Spending vs Repayment Behavior")
        fig, ax = plt.subplots(figsize=(9, 5))
        scatter = ax.scatter(
            df_fe["BILL_WT_AVG"],
            df_fe["PAY_BILL_RATIO"],
            c=df_fe["Cluster"]
        )
        ax.set_xlabel("Weighted Bill Amount")
        ax.set_ylabel("Pay/Bill Ratio")
        ax.set_title("Customer Segments by Spending and Repayment Pattern")
        st.pyplot(fig)

        st.subheader("Business Reading Guide")
        st.markdown("""
- **High Value Revolvers**: high bill, low repayment ratio, low delay → strongest retention candidates  
- **Stable Revenue Customers**: moderate bill, manageable repayment behavior, low delay → retain and protect  
- **Low Utilization Customers**: low usage despite available capacity → low retention priority  
- **High Risk Customers**: higher delays and weak repayment → retention spending should be selective
""")

        st.subheader("Sample Customer-Level Output")
        st.dataframe(
            df_fe[
                [
                    "LIMIT_BAL",
                    "MAX_DELAY",
                    "DELAY_WT_AVG",
                    "BILL_WT_AVG",
                    "PAY_WT_AVG",
                    "PAY_BILL_RATIO",
                    "Segment"
                ]
            ].head(20)
        )

else:
    st.warning("Use the sample button or upload a file to generate customer segments.")