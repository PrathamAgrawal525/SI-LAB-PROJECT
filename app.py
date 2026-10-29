"""
AutoVision AI - Streamlit Analytics & Valuation Dashboard
Interactive market intelligence, feature importance, and real-time used car valuation.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model_utils import (
    get_feature_importances,
    load_car_data,
    load_model,
    predict_price,
)

st.set_page_config(
    page_title="AutoVision AI · Car Market Intelligence",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return load_car_data()


@st.cache_resource
def get_model():
    return load_model()


def apply_custom_styling():
    st.markdown(
        """
        <style>
            /* Base Theme */
            .stApp {
                background: linear-gradient(135deg, #070a18 0%, #0d152a 50%, #081926 100%);
                color: #e4ecf7;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }

            /* Metric Cards */
            div[data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.035);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                padding: 18px 20px;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
                backdrop-filter: blur(8px);
                transition: transform 0.2s ease, border-color 0.2s ease;
            }
            div[data-testid="stMetric"]:hover {
                transform: translateY(-2px);
                border-color: rgba(99, 102, 241, 0.4);
            }

            /* Hero Banner */
            .hero-card {
                background: linear-gradient(120deg, rgba(99, 102, 241, 0.25) 0%, rgba(6, 182, 212, 0.2) 100%);
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 20px;
                padding: 28px 32px;
                margin-bottom: 24px;
                box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4);
            }
            .hero-tag {
                color: #38bdf8;
                font-size: 0.8rem;
                font-weight: 800;
                letter-spacing: 2px;
                text-transform: uppercase;
                margin-bottom: 6px;
            }
            .hero-title {
                color: #ffffff;
                font-size: 2.2rem;
                font-weight: 800;
                margin: 0 0 8px 0;
            }
            .hero-desc {
                color: #cbd5e1;
                font-size: 1rem;
                max-width: 800px;
                line-height: 1.5;
                margin: 0;
            }

            /* Sidebar styling */
            section[data-testid="stSidebar"] {
                background-color: #0b1122;
                border-right: 1px solid rgba(255, 255, 255, 0.08);
            }

            /* Tabs and Tables */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 10px;
                padding: 8px 16px;
                background-color: rgba(255, 255, 255, 0.04);
                color: #94a3b8;
            }
            .stTabs [aria-selected="true"] {
                background-color: #4f46e5 !important;
                color: #ffffff !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    apply_custom_styling()
    df = get_dataset()
    bundle = get_model()

    # Hero Banner
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-tag">✦ Automotive Intelligence & Market Analytics</div>
            <h1 class="hero-title">🚘 AutoVision AI Dashboard</h1>
            <p class="hero-desc">
                High-precision machine learning valuation engine and used car market exploratory analytics.
                Adjust parameters in the sidebar to simulate live depreciation and resale values.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Vehicle Configurator")
        st.caption("Configure specifications to compute fair resale value:")

        present_price = st.number_input(
            "Showroom Price (₹ Lakhs)",
            min_value=0.5,
            max_value=100.0,
            value=8.5,
            step=0.25,
            help="Original showroom or ex-showroom invoice price.",
        )

        kms_driven = st.number_input(
            "Odometer Reading (Km)",
            min_value=0,
            max_value=1000000,
            value=35000,
            step=1000,
            help="Total distance traveled by the car.",
        )

        age = st.slider(
            "Vehicle Age (Years)",
            min_value=0,
            max_value=25,
            value=5,
            help="Years since manufacturing/first registration.",
        )

        col_f, col_s = st.columns(2)
        with col_f:
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
        with col_s:
            seller_type = st.selectbox("Seller", ["Dealer", "Individual"])

        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

        st.markdown("---")
        st.subheader("💡 Model Info")
        st.markdown(
            """
            - **Algorithm**: Multiple Linear Regression
            - **Features**: 6 primary factors (numeric + categorical)
            - **Dataset**: Verified used car transactions
            """
        )

    # Valuation Computation
    inputs = {
        "Present_Price": float(present_price),
        "Kms_Driven": float(kms_driven),
        "Age": int(age),
        "Fuel_Type": fuel_type,
        "Seller_Type": seller_type,
        "Transmission": transmission,
    }

    predicted_price = predict_price(bundle, inputs)
    fair_min = max(0.0, predicted_price * 0.85)
    fair_max = predicted_price * 1.15
    retention_rate = (predicted_price / present_price) * 100 if present_price > 0 else 0
    depreciation_lakhs = max(0.0, present_price - predicted_price)

    # Top KPI Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        label="Predicted Resale Value",
        value=f"₹ {predicted_price:.2f} L",
        delta=f"{predicted_price - df['Selling_Price'].mean():+.2f} L vs Avg",
    )
    m2.metric(
        label="Fair Market Band",
        value=f"₹ {fair_min:.2f} - {fair_max:.2f} L",
        help="Expected valuation range accounting for condition and negotiation room.",
    )
    m3.metric(
        label="Value Retention",
        value=f"{retention_rate:.1f}%",
        delta=f"-{depreciation_lakhs:.2f} L lost",
        delta_color="inverse",
    )
    m4.metric(
        label="Market Benchmark Avg",
        value=f"₹ {df['Selling_Price'].mean():.2f} L",
        delta=f"{df.shape[0]} cars sampled",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabbed Views: Market Analytics & What-if Simulation
    tab_analytics, tab_simulation, tab_dataset = st.tabs(
        ["📊 Market Analytics & Feature Impact", "🔮 Scenario What-If Simulator", "📁 Dataset Explorer"]
    )

    with tab_analytics:
        col_left, col_right = st.columns([1.3, 1])

        with col_left:
            st.subheader("📈 Present Price vs. Selling Price")
            scatter = px.scatter(
                df,
                x="Present_Price",
                y="Selling_Price",
                color="Fuel_Type",
                size="Kms_Driven",
                hover_name="Car_Name",
                title="Used Car Resale vs Showroom Price by Fuel Category",
                labels={
                    "Present_Price": "Showroom Price (₹ Lakhs)",
                    "Selling_Price": "Resale Price (₹ Lakhs)",
                    "Fuel_Type": "Fuel",
                },
                template="plotly_dark",
            )
            # Add user's car as an annotation star
            scatter.add_trace(
                go.Scatter(
                    x=[present_price],
                    y=[predicted_price],
                    mode="markers+text",
                    marker=dict(symbol="star", size=18, color="#00f2fe", line=dict(width=2, color="#ffffff")),
                    text=["Your Car"],
                    textposition="top center",
                    name="Current Estimate",
                )
            )
            scatter.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(scatter, use_container_width=True)

        with col_right:
            st.subheader("🔥 Model Feature Impact")
            fi_df = get_feature_importances(bundle)
            bar = px.bar(
                fi_df,
                x="feature",
                y="impact",
                color="impact",
                color_continuous_scale="Purp",
                template="plotly_dark",
                title="Regression Coefficients (Influence on Selling Price)",
                labels={"feature": "Predictor", "impact": "Impact (₹ Lakhs)"},
            )
            bar.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_tickangle=-35,
            )
            st.plotly_chart(bar, use_container_width=True)

        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            st.subheader("📉 Resale Price vs. Vehicle Age")
            age_fig = px.box(
                df,
                x="Age",
                y="Selling_Price",
                color="Fuel_Type",
                template="plotly_dark",
                title="Depreciation Trend Across Vehicle Age",
                labels={"Age": "Age (Years)", "Selling_Price": "Selling Price (₹ Lakhs)"},
            )
            age_fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(age_fig, use_container_width=True)

        with col_dist2:
            st.subheader("📊 Market Price Distribution")
            hist = px.histogram(
                df,
                x="Selling_Price",
                nbins=24,
                color_discrete_sequence=["#6366f1"],
                template="plotly_dark",
                title="Histogram of Selling Prices",
                labels={"Selling_Price": "Price (₹ Lakhs)"},
            )
            hist.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(hist, use_container_width=True)

    with tab_simulation:
        st.subheader("🔮 Depreciation & Scenario Projection")
        st.write("Examine how future mileage and vehicle aging impact your estimated resale value:")

        proj_cols = st.columns(3)
        # Scenario 1: Next year
        s1_inputs = dict(inputs)
        s1_inputs["Age"] = inputs["Age"] + 1
        s1_inputs["Kms_Driven"] = inputs["Kms_Driven"] + 12000
        p_s1 = predict_price(bundle, s1_inputs)

        # Scenario 2: In 2 years
        s2_inputs = dict(inputs)
        s2_inputs["Age"] = inputs["Age"] + 2
        s2_inputs["Kms_Driven"] = inputs["Kms_Driven"] + 25000
        p_s2 = predict_price(bundle, s2_inputs)

        # Scenario 3: In 3 years
        s3_inputs = dict(inputs)
        s3_inputs["Age"] = inputs["Age"] + 3
        s3_inputs["Kms_Driven"] = inputs["Kms_Driven"] + 40000
        p_s3 = predict_price(bundle, s3_inputs)

        proj_cols[0].metric(
            label="In 1 Year (+12k km)",
            value=f"₹ {p_s1:.2f} L",
            delta=f"{p_s1 - predicted_price:.2f} L",
            delta_color="inverse",
        )
        proj_cols[1].metric(
            label="In 2 Years (+25k km)",
            value=f"₹ {p_s2:.2f} L",
            delta=f"{p_s2 - predicted_price:.2f} L",
            delta_color="inverse",
        )
        proj_cols[2].metric(
            label="In 3 Years (+40k km)",
            value=f"₹ {p_s3:.2f} L",
            delta=f"{p_s3 - predicted_price:.2f} L",
            delta_color="inverse",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📋 Specification Summary Table")
        summary_df = pd.DataFrame(
            [
                {"Feature": "Original Showroom Price", "Specification": f"₹ {present_price:.2f} Lakhs"},
                {"Feature": "Odometer Distance", "Specification": f"{kms_driven:,} km"},
                {"Feature": "Car Age", "Specification": f"{age} years"},
                {"Feature": "Fuel Type", "Specification": fuel_type},
                {"Feature": "Seller Channel", "Specification": seller_type},
                {"Feature": "Transmission Type", "Specification": transmission},
                {"Feature": "Calculated Resale Value", "Specification": f"₹ {predicted_price:.2f} Lakhs"},
                {"Feature": "Fair Negotiation Range", "Specification": f"₹ {fair_min:.2f} to ₹ {fair_max:.2f} Lakhs"},
            ]
        )
        st.table(summary_df)

    with tab_dataset:
        st.subheader("📁 Historical Used Car Dataset")
        st.dataframe(df, use_container_width=True)
        st.caption(f"Showing all {len(df)} records from car_data.csv.")


if __name__ == "__main__":
    main()
