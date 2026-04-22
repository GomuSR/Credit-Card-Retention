import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Card Retention Segmentation", layout="wide")

st.title("Credit Card Customer Retention Segmentation")
st.write(
    "Upload a credit card customer file to segment customers based on spending, "
    "repayment behavior, and delay severity."
)

# Load saved scaler and model
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

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


def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def feature_engineering(df):
    df_fe = df.copy()

    # Delay cleanup
    delay_clean = df_fe[delay_cols].clip(lower=0)

    df_fe["MAX_DELAY"] = delay_clean.max(axis=1)

    df_fe["DELAY_WT_AVG"] = (
        4 * delay_clean["PAY_0"] +
        3 * delay_clean["PAY_2"] +
        2 * delay_clean["PAY_3"] +
        1 * delay_clean["PAY_4"] +
        1 * delay_clean["PAY_5"] +
        1 * delay_clean["PAY_6"]
    ) / 12

    # Bill cleanup at source
    df_fe[bill_cols] = df_fe[bill_cols].clip(lower=0)

    df_fe["BILL_WT_AVG"] = (
        4 * df_fe["BILL_AMT1"] +
        3 * df_fe["BILL_AMT2"] +
        2 * df_fe["BILL_AMT3"] +
        1 * df_fe["BILL_AMT4"] +
        1 * df_fe["BILL_AMT5"] +
        1 * df_fe["BILL_AMT6"]
    ) / 12

    df_fe["PAY_WT_AVG"] = (
        4 * df_fe["PAY_AMT1"] +
        3 * df_fe["PAY_AMT2"] +
        2 * df_fe["PAY_AMT3"] +
        1 * df_fe["PAY_AMT4"] +
        1 * df_fe["PAY_AMT5"] +
        1 * df_fe["PAY_AMT6"]
    ) / 12

    df_fe["PAY_BILL_RATIO"] = np.where(
        df_fe["BILL_WT_AVG"] == 0,
        0,
        df_fe["PAY_WT_AVG"] / df_fe["BILL_WT_AVG"]
    )

    df_fe["PAY_BILL_RATIO"] = df_fe["PAY_BILL_RATIO"].clip(lower=0, upper=5)

    return df_fe


uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    required_cols = [
        "LIMIT_BAL", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        df_fe = feature_engineering(df)

        X = df_fe[feature_cols].copy()
        X_scaled = scaler.transform(X)
        df_fe["Cluster"] = kmeans.predict(X_scaled)

        st.subheader("Engineered Features with Cluster")
        st.dataframe(df_fe[feature_cols + ["Cluster"]].head())

        cluster_summary = df_fe.groupby("Cluster")[feature_cols].mean().round(2)
        cluster_summary["Customer_Count"] = df_fe.groupby("Cluster").size()

        st.subheader("Cluster Summary")
        st.dataframe(cluster_summary)

        st.subheader("Customer Count by Cluster")
        st.bar_chart(cluster_summary["Customer_Count"])

        st.subheader("Cluster Visualization")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            df_fe["BILL_WT_AVG"],
            df_fe["PAY_BILL_RATIO"],
            c=df_fe["Cluster"]
        )
        ax.set_xlabel("Weighted Bill Average")
        ax.set_ylabel("Pay/Bill Ratio")
        ax.set_title("Customer Segments by Spending and Repayment Behavior")
        st.pyplot(fig)

        st.subheader("Business Reading Guide")
        st.markdown("""
- **High bill + low pay/bill ratio + low delay**: high-value revolving customers, likely strong retention targets  
- **Low bill + high pay/bill ratio + low delay**: low-risk but low-revenue customers  
- **High delay + low ratio**: risky customers, retention spending should be selective  
- **Moderate bill + moderate ratio + low delay**: stable core segment worth protecting
""")

        st.subheader("Clustered Records")
        st.dataframe(df_fe.head(20))
else:
    st.info("Upload a file to generate customer segments.")