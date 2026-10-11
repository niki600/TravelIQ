import streamlit as st

import pandas as pd

import numpy as np

import joblib

import shap

from ai_explanation import generate_ai_explanation



# ============================================================

# TravelIQ - Hotel Booking Cancellation Risk Prediction

# ============================================================



st.set_page_config(

    page_title="TravelIQ | Cancellation Risk",

    page_icon="✈️",

    layout="wide",

    initial_sidebar_state="expanded"

)


# ============================================================
# Custom CSS - Professional TravelIQ UI
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: #f7f9fc;
    }

    /* Hide default Streamlit menu/footer */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Main content width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Hero section */
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 20px;

        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #1e3a8a 55%,
            #2563eb 100%
        );

        color: white;
        margin-bottom: 1.5rem;

        box-shadow:
            0 10px 30px rgba(15, 23, 42, 0.18);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        font-size: 18px;
        opacity: 0.9;
        margin-bottom: 0;
    }

    .hero-description {
        font-size: 14px;
        opacity: 0.78;
        margin-top: 0.7rem;
    }

    /* Section cards */
    .section-card {
        background: white;
        border-radius: 18px;
        padding: 1.4rem;
        border: 1px solid #e5e7eb;

        box-shadow:
            0 4px 18px rgba(15, 23, 42, 0.06);

        margin-bottom: 1rem;
    }

    /* Result cards */
    .result-card {
        background: white;
        border-radius: 18px;
        padding: 1.4rem;
        border: 1px solid #e5e7eb;
        text-align: center;
        min-height: 140px;

        box-shadow:
            0 5px 18px rgba(15, 23, 42, 0.07);
    }

    .result-label {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .result-value {
        color: #0f172a;
        font-size: 28px;
        font-weight: 800;
    }

    /* Risk badges */

    .risk-low {
        background: #dcfce7;
        color: #166534;
        padding: 0.55rem 1rem;
        border-radius: 999px;
        font-weight: 800;
        display: inline-block;
    }

    .risk-medium {
        background: #fef3c7;
        color: #92400e;
        padding: 0.55rem 1rem;
        border-radius: 999px;
        font-weight: 800;
        display: inline-block;
    }

    .risk-high {
        background: #fee2e2;
        color: #991b1b;
        padding: 0.55rem 1rem;
        border-radius: 999px;
        font-weight: 800;
        display: inline-block;
    }

    /* Business insight cards */

    .insight-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.2rem;
        min-height: 120px;

        box-shadow:
            0 4px 16px rgba(15, 23, 42, 0.05);
    }

    .insight-title {
        color: #64748b;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .insight-value {
        color: #0f172a;
        font-size: 25px;
        font-weight: 800;
    }

    .insight-small {
        color: #64748b;
        font-size: 12px;
        margin-top: 0.3rem;
    }

    /* Info boxes */

    .info-box {
        background: #eff6ff;
        border-left: 5px solid #2563eb;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        color: #1e3a8a;
        margin: 1rem 0;
    }

    /* Footer */

    .app-footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        padding-top: 2rem;
    }

    /* Improve buttons */

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 48px;
    }

    /* Sidebar */

    section[data-testid="stSidebar"] {
        background: #0f172a;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================

# Load saved Random Forest model

# ============================================================



MODEL_PATH = "models/final_random_forest.pkl"
FEATURE_MEDIANS_PATH = "models/feature_medians.pkl"





@st.cache_resource

def load_model():

    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_feature_medians():

    return joblib.load(FEATURE_MEDIANS_PATH)



model = load_model()
feature_medians = load_feature_medians()



@st.cache_data
def load_predictions():
    return pd.read_csv("data/final_predictions.csv")



# ============================================================

# SHAP Explainer

# ============================================================



@st.cache_resource

def load_shap_explainer():

    return shap.TreeExplainer(model)





explainer = load_shap_explainer()



# ============================================================

# Sidebar

# ============================================================



with st.sidebar:



    st.markdown(
    """
    <div style="text-align:center; padding:1rem 0 1.5rem 0;">
        <div style="font-size:42px;">✈️</div>
        <div style="font-size:25px; font-weight:800;">TravelIQ</div>
        <div style="font-size:13px; opacity:0.75;">Booking Intelligence Platform</div>
    </div>
    """,
    unsafe_allow_html=True
)



    st.divider()



    st.markdown("### 🎯 Risk Thresholds")



    st.markdown(

        """

        🟢 **Low Risk**  

        Cancellation probability < 30%



        🟡 **Medium Risk**  

        30% – 59%



        🔴 **High Risk**  

        ≥ 60%

        """

    )



    st.divider()



    st.markdown("### 🤖 Model")



    st.markdown(

        """

        **Random Forest Classifier**



        • 81 features  

        • Cancellation probability  

        • Business risk classification  

        • Proactive recommendations

        """

    )



    st.divider()



    st.caption("TravelIQ • Hotel Booking Analytics")


# ============================================================
# Hero Header
# ============================================================

st.markdown(
    """
<div class="hero">
    <div class="hero-title">✈️ TravelIQ</div>
    <div class="hero-subtitle">Hotel Booking Cancellation Risk Intelligence</div>
    <div class="hero-description">
        Predict cancellation probability, classify booking risk,
        and support proactive business decisions.
    </div>
</div>
    """,
    unsafe_allow_html=True
)

# ============================================================

# Tabs

# ============================================================



prediction_tab, dashboard_tab = st.tabs(

    [

        "🔮 Booking Risk Prediction",

        "📊 Business Intelligence"

    ]

)





# ============================================================

# TAB 1 - Prediction

# ============================================================



with prediction_tab:


    st.markdown(
    """
    <div class="section-card">
        <h3>📋 Booking Information</h3>
        <p style="color:#64748b;">
            Enter booking details to estimate cancellation risk.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

    # --------------------------------------------------------

    # Booking Inputs

    # --------------------------------------------------------



    col1, col2, col3 = st.columns(3)



    with col1:



        lead_time = st.number_input(

            "Lead Time (days)",

            min_value=0,

            max_value=1000,

            value=100

        )



        adr = st.number_input(

            "ADR",

            min_value=0.0,

            value=100.0,

            step=10.0

        )



        total_stay_nights = st.number_input(

            "Total Stay Nights",

            min_value=0,

            max_value=100,

            value=3

        )



    with col2:



        parking_spaces = st.number_input(

            "Required Car Parking Spaces",

            min_value=0,

            max_value=10,

            value=0

        )



        previous_cancellations = st.number_input(

            "Previous Cancellations",

            min_value=0,

            max_value=20,

            value=0

        )



        booking_changes = st.number_input(

            "Booking Changes",

            min_value=0,

            max_value=20,

            value=0

        )



    with col3:



        special_requests = st.number_input(

            "Total Special Requests",

            min_value=0,

            max_value=10,

            value=0

        )



        estimated_revenue = st.number_input(

            "Estimated Revenue",

            min_value=0.0,

            value=300.0,

            step=50.0

        )



        market_segment = st.selectbox(

            "Market Segment",

            [

                "Online TA",

                "Offline TA/TO",

                "Groups",

                "Direct",

                "Corporate",

                "Complementary",

                "Undefined"

            ]

        )



    # --------------------------------------------------------

    # Customer / Deposit Information

    # --------------------------------------------------------



    st.markdown("### 👤 Customer & Payment Information")



    col4, col5 = st.columns(2)



    with col4:



        customer_type = st.selectbox(

            "Customer Type",

            [

                "Transient",

                "Transient-Party",

                "Contract",

                "Group"

            ]

        )



    with col5:



        deposit_type = st.selectbox(

            "Deposit Type",

            [

                "No Deposit",

                "Non Refund",

                "Refundable"

            ]

        )



    st.write("")



    # --------------------------------------------------------

    # Prediction Button

    # --------------------------------------------------------



    predict_clicked = st.button(

        "🔮 Predict Cancellation Risk",

        type="primary",

        use_container_width=True

    )



    # ========================================================


    # ========================================================
    # Prediction Logic
    # ========================================================

    if predict_clicked:

        # Get model feature names
        feature_names = list(model.feature_names_in_)

        # Initialize every feature using medians calculated from X_train.
        # Then overwrite the features explicitly supplied by the user below.
        aligned_medians = feature_medians.reindex(feature_names)

        if aligned_medians.isna().any():
            missing_features = aligned_medians[aligned_medians.isna()].index.tolist()
            st.error(
                "Training medians are missing for these model features: "
                + ", ".join(missing_features)
            )
            st.stop()

        input_data = pd.DataFrame(
            [aligned_medians.to_numpy()],
            columns=feature_names
        )

        # ----------------------------------------------------
        # Numeric Features
        # ----------------------------------------------------

        numeric_values = {
            "LEAD_TIME": lead_time,
            "ADR": adr,
            "TOTAL_STAY_NIGHTS": total_stay_nights,
            "REQUIRED_CAR_PARKING_SPACES": parking_spaces,
            "PREVIOUS_CANCELLATIONS": previous_cancellations,
            "BOOKING_CHANGES": booking_changes,
            "TOTAL_SPECIAL_REQUESTS": special_requests,
            "ESTIMATED_REVENUE": estimated_revenue
        }

        for feature, value in numeric_values.items():
            if feature in input_data.columns:
                input_data.loc[0, feature] = value

        # ----------------------------------------------------
        # Derived Binary Features
        # ----------------------------------------------------

        binary_features = {
            "HAS_SPECIAL_REQUESTS": int(special_requests > 0),
            "HAS_PREVIOUS_CANCELLATIONS": int(previous_cancellations > 0),
            "HAS_BOOKING_CHANGES": int(booking_changes > 0)
        }

        for feature, value in binary_features.items():
            if feature in input_data.columns:
                input_data.loc[0, feature] = value

        # ----------------------------------------------------
        # Market Segment
        # ----------------------------------------------------

        market_column = f"MARKET_SEGMENT_{market_segment}"

        if market_column in input_data.columns:
            input_data.loc[0, market_column] = 1

        # ----------------------------------------------------
        # Customer Type
        # ----------------------------------------------------

        customer_column = f"CUSTOMER_TYPE_{customer_type}"

        if customer_column in input_data.columns:
            input_data.loc[0, customer_column] = 1

        # ----------------------------------------------------
        # Deposit Type
        # ----------------------------------------------------

        deposit_column = f"DEPOSIT_TYPE_{deposit_type}"

        if deposit_column in input_data.columns:
            input_data.loc[0, deposit_column] = 1

        # ----------------------------------------------------
        # Model Prediction
        # ----------------------------------------------------

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # ========================================================
        # SHAP Explanation
        # ========================================================

        shap_values = explainer.shap_values(input_data)

        # For binary classification, use cancellation class (class 1)
        if isinstance(shap_values, list):
            shap_array = np.asarray(shap_values[1])[0]
        else:
            shap_array = np.asarray(shap_values)[0]

        # Get top 5 features by absolute SHAP impact
        top_indices = np.argsort(np.abs(shap_array))[::-1][:5]

        shap_explanations = []

        for index in top_indices:
            feature_name = feature_names[index]
            shap_value = float(shap_array[index])

            if shap_value > 0:
                direction = "increases cancellation risk"
            else:
                direction = "decreases cancellation risk"

            shap_explanations.append(
                (
                    feature_name,
                    f"SHAP value = {shap_value:+.4f}; {direction}"
                )
            )

        # ----------------------------------------------------
        # Risk Category
        # ----------------------------------------------------

        if probability < 0.30:
            risk_category = "Low Risk"
        elif probability < 0.60:
            risk_category = "Medium Risk"
        else:
            risk_category = "High Risk"

        # ========================================================
        # AI Business Explanation - Phi-3
        # ========================================================

        booking_details = {
            "Lead Time": lead_time,
            "ADR": adr,
            "Total Stay Nights": total_stay_nights,
            "Required Car Parking Spaces": parking_spaces,
            "Previous Cancellations": previous_cancellations,
            "Booking Changes": booking_changes,
            "Total Special Requests": special_requests,
            "Estimated Revenue": estimated_revenue,
            "Market Segment": market_segment,
            "Customer Type": customer_type,
            "Deposit Type": deposit_type
        }

        try:
            ai_explanation = generate_ai_explanation(
                probability,
                risk_category,
                shap_explanations,
                booking_details
            )
        except Exception as e:
            ai_explanation = (
                "AI explanation could not be generated. "
                f"Error: {str(e)}"
            )

        # ----------------------------------------------------
        # Prediction Result Header
        # ----------------------------------------------------

        st.markdown("---")

        st.markdown(
            """
            <div class="section-card">
                <h2>📊 Prediction Result</h2>
                <p style="color:#64748b;">
                    Model-generated cancellation risk for this booking.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # Result Cards
        # ----------------------------------------------------

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Cancellation Probability
                    </div>
                    <div class="result-value">
                        {probability * 100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with result_col2:
            prediction_text = "Yes" if prediction == 1 else "No"
            prediction_icon = "🔴" if prediction == 1 else "🟢"

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Predicted Cancellation
                    </div>
                    <div class="result-value">
                        {prediction_icon} {prediction_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with result_col3:
            if risk_category == "Low Risk":
                badge_class = "risk-low"
            elif risk_category == "Medium Risk":
                badge_class = "risk-medium"
            else:
                badge_class = "risk-high"

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Risk Category
                    </div>
                    <div style="margin-top:1rem;">
                        <span class="{badge_class}">
                            {risk_category}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # Probability Progress Bar
        # ----------------------------------------------------

        st.write("")
        st.markdown(
            f"**Cancellation Risk Score — {probability * 100:.2f}%**"
        )
        st.progress(float(probability))

        # ----------------------------------------------------
        # Business Recommendation
        # ----------------------------------------------------

        st.markdown("### 💡 Business Recommendation")

        if risk_category == "High Risk":
            st.error(
                "🔴 **High-risk booking:** Consider proactive "
                "customer confirmation, payment verification "
                "and inventory planning."
            )
        elif risk_category == "Medium Risk":
            st.warning(
                "🟡 **Medium-risk booking:** Consider monitoring "
                "the booking and sending a confirmation reminder."
            )
        else:
            st.success(
                "🟢 **Low-risk booking:** No immediate intervention "
                "is required."
            )

        # --------------------------------------------------------
        # AI-Powered Explanation
        # --------------------------------------------------------

        st.markdown("---")
        st.markdown("### 🤖 AI-Powered Business Explanation")

        st.info(
            "TravelIQ uses SHAP explanations and Phi-3 to convert "
            "the model prediction into simple business language."
        )

        st.markdown(ai_explanation)

        # ----------------------------------------------------
        # Model Information
        # ----------------------------------------------------

        with st.expander("🤖 Model Information"):

            info_col1, info_col2, info_col3 = st.columns(3)

            with info_col1:
                st.metric("Model", "Random Forest")

            with info_col2:
                st.metric("Features", len(feature_names))

            with info_col3:
                st.metric("Threshold", "0.50")

            st.caption(
                "The saved Random Forest classifier generates the cancellation "
                "probability. Features not entered in the form are initialized "
                "using medians calculated from the training features."
            )

# TAB 2 - Business Intelligence Dashboard

with dashboard_tab:

    st.markdown(
        """
        <div class="section-card">
            <h2>📊 TravelIQ Business Intelligence</h2>
            <p style="color:#64748b;">High-risk booking analysis and business-level insights.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load predictions generated by 04_modeling_mlflow.ipynb
    try:
        df_results = load_predictions()
    except FileNotFoundError:
        st.error(
            "Prediction file not found: data/final_predictions.csv. "
            "Generate it from the modeling notebook first."
        )
        st.stop()

    required_columns = [
        "RISK_CATEGORY",
        "ADR",
        "STAYS_IN_WEEKEND_NIGHTS",
        "STAYS_IN_WEEK_NIGHTS",
        "LEAD_TIME",
    ]
    missing_columns = [col for col in required_columns if col not in df_results.columns]

    if missing_columns:
        st.error(
            "The predictions CSV is missing required columns: "
            + ", ".join(missing_columns)
        )
        st.stop()

    # Calculate dashboard metrics from the predictions CSV
    high_risk = df_results[df_results["RISK_CATEGORY"] == "High Risk"].copy()
    high_risk_count = len(high_risk)

    high_risk["REVENUE_AT_RISK"] = (
        pd.to_numeric(high_risk["ADR"], errors="coerce").fillna(0)
        * (
            pd.to_numeric(high_risk["STAYS_IN_WEEKEND_NIGHTS"], errors="coerce").fillna(0)
            + pd.to_numeric(high_risk["STAYS_IN_WEEK_NIGHTS"], errors="coerce").fillna(0)
        )
    )
    revenue_at_risk = high_risk["REVENUE_AT_RISK"].sum()
    avg_lead_time = pd.to_numeric(high_risk["LEAD_TIME"], errors="coerce").mean()
    avg_adr = pd.to_numeric(df_results["ADR"], errors="coerce").mean()
    avg_stay = (
        pd.to_numeric(df_results["STAYS_IN_WEEKEND_NIGHTS"], errors="coerce").fillna(0)
        + pd.to_numeric(df_results["STAYS_IN_WEEK_NIGHTS"], errors="coerce").fillna(0)
    ).mean()

    # --------------------------------------------------------
    # KPI Cards
    # --------------------------------------------------------
    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">🔴 HIGH-RISK BOOKINGS</div>
                <div class="insight-value">{high_risk_count:,}</div>
                <div class="insight-small">Bookings requiring attention in prediction data</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi2:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">💰 REVENUE AT RISK</div>
                <div class="insight-value">{revenue_at_risk:,.2f}</div>
                <div class="insight-small">ADR × stay nights for high-risk bookings</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi3:
        lead_time_display = f"{avg_lead_time:.2f}" if pd.notna(avg_lead_time) else "N/A"
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">📅 AVG. LEAD TIME</div>
                <div class="insight-value">{lead_time_display}</div>
                <div class="insight-small">Days before arrival, high-risk bookings</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")
    kpi4, kpi5, kpi6 = st.columns(3)

    with kpi4:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">💵 AVG. ADR</div>
                <div class="insight-value">{avg_adr:,.2f}</div>
                <div class="insight-small">Average daily rate across predictions</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi5:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">🏨 AVG. STAY</div>
                <div class="insight-value">{avg_stay:.2f}</div>
                <div class="insight-small">Nights per booking</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Derive market-segment counts from one-hot encoded columns
    market_columns = [
        col for col in df_results.columns
        if col.startswith("MARKET_SEGMENT_")
    ]
    market_counts = pd.Series(dtype="int64")

    if market_columns and not high_risk.empty:
        market_counts = high_risk[market_columns].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0).sum().astype(int)
        market_counts.index = [
            col.replace("MARKET_SEGMENT_", "") for col in market_counts.index
        ]
        market_counts = market_counts[market_counts > 0].sort_values(ascending=False)

    with kpi6:
        top_market = market_counts.index[0] if not market_counts.empty else "Unavailable"
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">🌐 TOP MARKET SEGMENT</div>
                <div class="insight-value">{top_market}</div>
                <div class="insight-small">Most represented encoded segment among high-risk bookings</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ========================================================
    # Charts Row 1
    # ========================================================
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown(
            """
            <div class="section-card">
                <h3>📈 High-Risk Bookings by Lead Time</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        if not high_risk.empty:
            lead_bins = [-1, 30, 60, 90, 180, float("inf")]
            lead_labels = ["0–30 days", "31–60 days", "61–90 days", "91–180 days", "180+ days"]
            lead_values = pd.to_numeric(high_risk["LEAD_TIME"], errors="coerce")
            lead_categories = pd.cut(
                lead_values,
                bins=lead_bins,
                labels=lead_labels,
                right=True
            )
            lead_time_data = (
                lead_categories.value_counts(sort=False)
                .reindex(lead_labels, fill_value=0)
                .rename_axis("Lead Time")
                .reset_index(name="High-Risk Bookings")
            )
            st.bar_chart(
                lead_time_data.set_index("Lead Time"),
                use_container_width=True
            )
        else:
            st.info("No high-risk bookings are present in the predictions file.")

    with chart_col2:
        st.markdown(
            """
            <div class="section-card">
                <h3>🌐 High-Risk Market Segment</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        if not market_counts.empty:
            market_segment_data = (
                market_counts.rename_axis("Market Segment")
                .reset_index(name="High-Risk Bookings")
            )
            st.bar_chart(
                market_segment_data.set_index("Market Segment"),
                use_container_width=True
            )
        else:
            st.info(
                "Market-segment one-hot columns were not found, "
                "so this chart cannot be calculated from the current CSV."
            )

    st.markdown("---")

    # ========================================================
    # Business Interpretation
    # ========================================================
    st.markdown(
        """
        <div class="section-card">
            <h2>💡 Key Business Insights</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    insight1, insight2 = st.columns(2)

    with insight1:
        if not market_counts.empty and high_risk_count > 0:
            top_market_count = int(market_counts.iloc[0])
            top_market_pct = top_market_count / high_risk_count * 100
            st.info(
                f"🌐 **{market_counts.index[0]}** is the most represented "
                f"encoded market segment among high-risk bookings "
                f"({top_market_count:,}, approximately {top_market_pct:.2f}%)."
            )
        else:
            st.info("Market-segment insight is unavailable from the current prediction file.")

        if not high_risk.empty:
            lead_counts = pd.cut(
                pd.to_numeric(high_risk["LEAD_TIME"], errors="coerce"),
                bins=[-1, 30, 60, 90, 180, float("inf")],
                labels=["0–30 days", "31–60 days", "61–90 days", "91–180 days", "180+ days"]
            ).value_counts()
            if not lead_counts.empty:
                busiest_lead_group = lead_counts.idxmax()
                st.warning(
                    f"📅 The **{busiest_lead_group}** lead-time group has the "
                    f"largest number of high-risk bookings ({int(lead_counts.max()):,})."
                )
        st.info(f"📊 The prediction file contains **{len(df_results):,} bookings**.")

    with insight2:
        st.error(f"💰 Estimated **revenue at risk is {revenue_at_risk:,.2f}**.")
        if pd.notna(avg_lead_time):
            st.warning(
                f"📅 Average lead time among high-risk bookings is "
                f"**{avg_lead_time:.2f} days**."
            )
        st.success(
            "🎯 Prioritize high-risk bookings for proactive confirmation, "
            "payment verification and inventory planning."
        )

    # ========================================================
    # Risk Strategy
    # ========================================================
    st.markdown("---")
    st.markdown(
        """
        <div class="section-card">
            <h2>🎯 Recommended Risk Strategy</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    strategy1, strategy2, strategy3 = st.columns(3)

    with strategy1:
        st.success(
            """
            **🟢 Low Risk**

            No immediate intervention.

            Continue normal booking operations.
            """
        )

    with strategy2:
        st.warning(
            """
            **🟡 Medium Risk**

            Monitor booking activity.

            Send confirmation reminders.
            """
        )

    with strategy3:
        st.error(
            """
            **🔴 High Risk**

            Prioritize proactive confirmation.

            Verify payment and plan inventory.
            """
        )



# ============================================================
# Footer
# ============================================================

st.markdown(
    """
    <div class="app-footer">
        ✈️ TravelIQ — Hotel Booking Cancellation Risk Intelligence
        <br>
        Machine Learning • Business Analytics • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
