# ✈️ TravelIQ --- Hotel Booking Cancellation Risk Intelligence

TravelIQ is an end-to-end data analytics and machine learning project
that predicts hotel booking cancellation risk, explains predictions
using SHAP, generates AI-powered business explanations using Phi-3 Mini,
and presents actionable insights through an interactive Streamlit
dashboard.

The project combines data engineering, cloud data warehousing, machine
learning, explainable AI, and business intelligence to help hotel teams
investigate cancellation patterns and prioritize proactive booking
follow-up.

## 📌 Project Overview

Hotel booking cancellations can affect revenue, room inventory, and
operational planning. TravelIQ analyzes booking characteristics,
estimates cancellation probability, and groups predictions into risk
categories to support data-informed decisions.

### Project Objectives

-   Analyze hotel booking data and identify cancellation patterns.
-   Clean and prepare data for analytics and machine learning.
-   Build and compare multiple classification models.
-   Predict cancellation probability for individual bookings.
-   Explain model predictions using SHAP.
-   Generate natural-language business explanations using Phi-3 Mini.
-   Estimate revenue exposure associated with high-risk bookings.
-   Present key metrics and recommended actions through an interactive
    dashboard.

## 🏗️ Project Architecture

``` text
Hotel Booking Demand Dataset
          |
          v
Data Cleaning & Preparation
     Databricks + PySpark
          |
          v
Data Analysis & Feature Engineering
          |
          v
Snowflake SQL Analytics
          |
          v
Model Training & Experiment Tracking
 Logistic Regression | Random Forest
        XGBoost | MLflow
          |
          v
Selected Random Forest Model
          |
          v
Streamlit Application
       /             \
      v               v
Risk Prediction   Business Intelligence
      |               |
      v               v
SHAP Explanations  KPIs & Interactive Charts
      |
      v
Phi-3 Mini Business Explanation
```

## 🛠️ Technology Stack

  Category                  Technologies
  ------------------------- --------------------------------------
  Programming               Python
  Data manipulation         Pandas, NumPy
  Big data processing       Databricks, PySpark
  Cloud data warehouse      Snowflake, SQL
  Machine learning          Scikit-learn, XGBoost
  Experiment tracking       MLflow
  Explainable AI            SHAP
  AI-powered explanations   Phi-3 Mini, Ollama
  Web application           Streamlit
  Model persistence         Joblib
  Business intelligence     Power BI
  Development tools         Visual Studio Code, Jupyter Notebook
  Version control           Git, GitHub

## 📊 Dataset

**Dataset:** [Hotel Booking Demand ---
Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

The dataset contains hotel booking information used to explore
cancellation behavior and booking characteristics.

Important features include:

-   Lead time
-   Average Daily Rate (ADR)
-   Length of stay
-   Previous cancellations
-   Booking changes
-   Special requests
-   Required car parking spaces
-   Market segment
-   Customer type
-   Deposit type

The project uses engineered numerical and categorical features for model
training and prediction.

## ⚙️ Project Workflow

### 1. Data Engineering and Preparation

Databricks and PySpark were used for data processing and preparation,
including:

-   Data quality and missing-value analysis.
-   Cleaning and preparing datasets.
-   Feature engineering for analytics.
-   Preparing data for machine learning and business reporting.

### 2. SQL Analytics with Snowflake

Snowflake was used for SQL-based analysis and business metric
exploration. The workflow supports structured querying, booking-pattern
analysis, and preparation of insights for business decision-making.

### 3. Machine Learning and MLflow

The following classification models were trained and compared:

-   Logistic Regression
-   Logistic Regression (Scaled)
-   Random Forest
-   XGBoost

MLflow was used to track experiments and evaluation metrics.

### 4. Model Performance

The following metrics were recorded in MLflow and checked against the
project's experiment results.

  Model                            Precision   Recall   F1 Score   ROC-AUC
  ------------------------------ ----------- -------- ---------- ---------
  Logistic Regression                  0.499    0.784      0.609     0.774
  Logistic Regression (Scaled)         0.457    0.902      0.607     0.792
  Random Forest (Selected)             0.523    0.766      0.621     0.787
  XGBoost                              0.659    0.514      0.578     0.802

**Why was Random Forest selected?**

Random Forest achieved the highest F1 score among the evaluated models
at **0.621**, providing a balance between precision and recall. It was
selected for the Streamlit prediction application based on this
evaluation priority.

XGBoost achieved the highest ROC-AUC (approximately **0.802**) and the
highest precision (approximately **0.659**). Scaled Logistic Regression
achieved the highest recall (approximately **0.902**). Random Forest was
not the best model on every metric; the selection reflects the project's
chosen balance of precision and recall.

### 5. Cancellation Risk Prediction

The Streamlit application accepts booking information and generates a
predicted cancellation probability using the saved Random Forest
classifier.

Risk categories use these probability thresholds:

  Risk Category    Cancellation Probability
  ---------------- --------------------------
  🟢 Low Risk      Below 30%
  🟡 Medium Risk   30% to below 60%
  🔴 High Risk     60% or above

The application displays:

-   Cancellation probability.
-   Predicted cancellation outcome.
-   Risk category.
-   Business recommendations.
-   Model information.

Features not entered directly through the form are initialized using
saved training-feature medians in `models/feature_medians.pkl`.

### 6. Explainable AI with SHAP

SHAP is used to identify features that contribute to an individual
prediction. The application displays the top five features by absolute
SHAP impact and indicates whether each feature increases or decreases
the model's predicted cancellation risk.

This helps users interpret model outputs and understand factors
influencing a prediction.

### 7. AI-Powered Business Explanations

TravelIQ integrates Phi-3 Mini through Ollama to turn prediction results
and SHAP explanations into natural-language business interpretations.

The explanation uses:

-   Predicted cancellation probability.
-   Risk category.
-   Important SHAP features.
-   Booking details.

AI-generated explanations are decision-support outputs and should be
interpreted alongside the underlying model results.

## 📈 Business Intelligence Dashboard

The dashboard summarizes high-risk bookings and their potential business
implications.

### Key Performance Indicators

The values below describe the saved prediction dataset. Average ADR and
average length of stay are **overall averages across all prediction
records**, not averages for high-risk records only.

  Metric                                                  Value
  ---------------------------------------------- --------------
  Prediction Records                                     25,556
  High-Risk Bookings                                      6,442
  Estimated Revenue at Risk                        3,937,088.65
  Average Lead Time of High-Risk Bookings           139.01 days
  Overall Average ADR                                    128.74
  Overall Average Length of Stay                    3.81 nights
  Average ADR of High-Risk Bookings                      137.50
  Average Length of Stay of High-Risk Bookings      4.39 nights

**Revenue note:** Revenue figures are in the dataset's monetary units;
the currency has not been independently confirmed. Revenue at risk is an
estimate, not a guaranteed loss.

### Dashboard Visualizations

The Business Intelligence tab includes:

-   High-risk bookings by lead-time group.
-   High-risk bookings by market segment.
-   KPI cards for booking risk and revenue exposure.
-   Key business insights.
-   Recommended risk strategies.

### Key Business Insights

**1. Online TA is the dominant represented segment among high-risk
bookings.**

Online TA accounts for **6,135 of 6,442 high-risk bookings (95.23%)** in
the saved prediction dataset. This suggests that Online TA bookings
deserve further investigation when prioritizing channel-level
cancellation monitoring. This is the share of high-risk bookings
represented by Online TA, **not** the cancellation rate among all Online
TA bookings.

**2. Many high-risk bookings have long lead times.**

The dashboard reports **2,218 high-risk bookings** in the 91--180 day
lead-time group and **1,931** with a lead time exceeding 180 days. These
patterns can inform when to schedule booking confirmations and
monitoring activities.

**3. High-risk bookings represent potential revenue exposure.**

The reported estimated revenue at risk is **3,937,088.65**, calculated
by multiplying ADR by total stay nights for each high-risk booking and
summing those estimates. This is estimated exposure, not a guaranteed
loss or a forecast of realized financial impact.

**4. High-risk bookings have a long average lead time.**

The average lead time among high-risk bookings is **139.01 days**. This
may help the business consider earlier monitoring and proactive
confirmation strategies.

### Revenue at Risk Calculation

``` text
Revenue at Risk = ADR × Total Stay Nights
```

For each high-risk booking, ADR is multiplied by the sum of weekend and
weekday stay nights. The estimates are then summed across high-risk
bookings. This metric is intended for prioritization and planning; it
does not account for every factor affecting realized revenue.

## 🎯 Recommended Business Actions

  -----------------------------------------------------------------------
  Risk Category                       Suggested Action
  ----------------------------------- -----------------------------------
  Low Risk                            Continue normal booking operations.

  Medium Risk                         Monitor booking activity and send
                                      confirmation reminders when
                                      appropriate.

  High Risk                           Prioritize proactive confirmation,
                                      review payment status, and plan
                                      room inventory carefully.
  -----------------------------------------------------------------------

These are suggested actions based on predicted risk, not guarantees that
a booking will or will not be cancelled. Predictions should be evaluated
against actual outcomes before they are used for consequential
operational decisions.

## 🚀 Running the Streamlit Application

### Prerequisites

-   Python installed.
-   Visual Studio Code or another Python development environment.
-   The saved model and required supporting files.
-   Ollama and the configured Phi-3 model for AI-powered explanations.

### 1. Clone the Repository

After replacing the placeholder with your repository URL:

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd TravelIQ
```

### 2. Create and Activate a Virtual Environment

``` bash
python -m venv .venv
```

On Windows:

``` bash
.venv\\Scripts\\activate
```

### 3. Install Dependencies

If the repository contains a `requirements.txt` file:

``` bash
pip install -r requirements.txt
```

Make sure `requirements.txt` includes the packages required by your
actual application and compatible versions.

### 4. Check the Required Files

The core application structure may look like this:

``` text
TravelIQ/
├── app.py
├── ai_explanation.py
├── requirements.txt
├── models/
│   ├── final_random_forest.pkl
│   └── feature_medians.pkl
└── data/
    └── final_predictions.csv
```

This is an example structure; keep filenames and paths consistent with
your actual repository.

The prediction application requires the saved Random Forest model. The
Business Intelligence dashboard requires `data/final_predictions.csv` if
the app loads its data from that location. AI explanations depend on
`ai_explanation.py` and the configured Ollama/Phi-3 setup.

### 5. Configure Phi-3 with Ollama

Install Ollama and make sure the model configured in `ai_explanation.py`
is available locally. For example, if your code uses the model name
`phi3`:

``` bash
ollama pull phi3
ollama run phi3
```

Use the exact model name configured in your application. Keep Ollama
running when using the AI explanation feature.

### 6. Launch TravelIQ

From the project root directory:

``` bash
streamlit run app.py
```

Open the local URL shown in the terminal, typically
`http://localhost:8501`.

## 📁 Suggested Project Structure

``` text
TravelIQ/
├── app.py
├── ai_explanation.py
├── requirements.txt
├── README.md
├── models/
│   ├── final_random_forest.pkl
│   └── feature_medians.pkl
├── data/
│   └── final_predictions.csv
├── notebooks/
│   └── 04_modeling_mlflow.ipynb
├── sql/
│   └── snowflake_analytics.sql
└── screenshots/
    ├── prediction_page.png
    └── business_intelligence.png
```

This is an organizational example, not a claim that every listed file
already exists. Keep your actual filenames and paths consistent with the
application code.

## 💼 Business Value

TravelIQ demonstrates how data engineering, machine learning,
explainable AI, and business intelligence can work together to support
hotel operations.

Potential applications include:

-   Identifying bookings that warrant additional monitoring.
-   Prioritizing customer confirmation efforts.
-   Understanding patterns in predicted cancellation risk.
-   Estimating potential revenue exposure.
-   Communicating model predictions to non-technical stakeholders.
-   Supporting more informed booking and inventory planning.

TravelIQ is an analytical decision-support project. Model outputs should
be evaluated against actual cancellation outcomes before being used to
make consequential operational decisions.

## 🔮 Future Improvements

-   Deploy the Streamlit application to a cloud hosting platform.
-   Connect the dashboard directly to Snowflake for updated analytics.
-   Add monitoring for model performance and data drift.
-   Evaluate probability calibration and alternative decision
    thresholds.
-   Compare model performance across customer and market segments.
-   Add automated reporting and booking-risk alerts.
-   Evaluate predicted risks against subsequently observed cancellation
    outcomes.
-   Improve revenue-exposure estimates using business-specific revenue
    and cancellation assumptions.

## 👩‍💻 Author

**Nikita Swain**\
MCA Graduate \| Data Analytics \| Data Science \| Machine Learning

-   GitHub: \[Add your GitHub profile link\]
-   LinkedIn: \[Add your LinkedIn profile link\]

------------------------------------------------------------------------

⭐ If you find this project interesting, feel free to explore the
repository and share your feedback.
