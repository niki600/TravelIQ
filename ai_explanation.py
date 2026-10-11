# ============================================================
# TravelIQ - AI Explanation Layer
# Ollama + Phi-3
# ============================================================

import ollama


# ------------------------------------------------------------
# Generate Plain-English Explanation
# ------------------------------------------------------------

def generate_ai_explanation(
    probability,
    risk_category,
    shap_explanations,
    booking_details
):
    """
    Converts model prediction + SHAP explanation
    into a simple business-friendly explanation using Phi-3.
    """

    # Convert probability to percentage
    probability_percent = probability * 100

    # Format SHAP factors
    shap_text = "\n".join(
        [f"- {feature}: {value}" for feature, value in shap_explanations]
    )

    # Format booking information
    booking_text = "\n".join(
        [f"- {key}: {value}" for key, value in booking_details.items()]
    )

    # --------------------------------------------------------
    # Prompt for Phi-3
    # --------------------------------------------------------

    prompt = f"""
You are TravelIQ, an AI assistant for hotel booking analytics.

Explain the cancellation risk of a hotel booking in simple
business language.

Do not invent facts.
Use only the information provided below.

BOOKING DETAILS:
{booking_text}

MODEL PREDICTION:
Cancellation Probability: {probability_percent:.2f}%
Risk Category: {risk_category}

SHAP FACTORS:
{shap_text}

Give the response in this format:

Risk Summary:
Explain the cancellation risk in 1-2 sentences.

Key Reasons:
Give the most important reasons based on the SHAP factors.

Business Recommendation:
Give 2-3 practical actions the hotel/business team can take.

Keep the explanation concise and easy for a non-technical
business user to understand.
"""

    # --------------------------------------------------------
    # Call Phi-3 through Ollama
    # --------------------------------------------------------

    response = ollama.chat(
        model="phi3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # --------------------------------------------------------
    # Return AI response
    # --------------------------------------------------------

    return response["message"]["content"]