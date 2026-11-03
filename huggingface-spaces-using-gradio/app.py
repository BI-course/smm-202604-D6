import os

import gradio as gr
import joblib
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# ZeroGPU keep-alive. Hugging Face's free tier provisions new Gradio Spaces
# on ZeroGPU, and some fail to start unless at least one function is
# decorated with @spaces.GPU. None of the four models needs a GPU.
# The `spaces` package exists only on Hugging Face, so import it safely
# and the same file still runs locally.
# -----------------------------------------------------------------------
try:
    import spaces

    @spaces.GPU
    def _warm_zerogpu():
        return None

    _warm_zerogpu()
except ImportError:
    pass

# -----------------------------------------------------------------------
# Load all four persisted artifacts ONCE, at import time.
# Looks next to app.py (the Space layout), then ./model, then ../model
# (the repo layout), so the same file works in both places.
# -----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = next(
    p for p in (HERE, os.path.join(HERE, "model"), os.path.join(HERE, "..", "model"))
    if os.path.exists(os.path.join(p, "lasso_regressor_for_sme_revenue.joblib"))
)

regression_pipeline = joblib.load(os.path.join(MODEL_DIR, "lasso_regressor_for_sme_revenue.joblib"))
classification_pipeline = joblib.load(os.path.join(MODEL_DIR, "svc_classifier_for_sme_credit_risk.joblib"))

clustering_artifact = joblib.load(os.path.join(MODEL_DIR, "kmeans_mall_customer_segmentation.joblib"))
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]

arm_artifact = joblib.load(os.path.join(MODEL_DIR, "apriori_recommendation_rules.joblib"))
recommendation_lookup = arm_artifact["recommendation_lookup"]
sku_lookup = arm_artifact["sku_lookup"]

# Categories exactly as the pipelines' encoders were fitted
SECTORS = ["Agriculture", "Hospitality", "Manufacturing", "Retail", "Services", "Technology"]
COUNTIES = ["Eldoret", "Kisumu", "Mombasa", "Nairobi", "Nakuru", "Other"]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
EDUCATION = ["Primary", "Secondary", "Certificate", "Diploma", "Bachelors", "Masters", "Doctorate"]
LOAN_PURPOSES = ["Business Expansion", "Equipment Purchase", "Inventory Financing", "Working Capital"]


def require_filled(**fields):
    """Raise a friendly error if any numeric field was left blank."""
    missing = [name for name, value in fields.items() if value is None]
    if missing:
        raise gr.Error(f"Please fill in: {', '.join(missing)}")


# -----------------------------------------------------------------------
# 1. Regression: predict monthly revenue (KES) for an SME
# -----------------------------------------------------------------------
def predict_revenue(sector, county, quarter, owner_education_level,
                    business_age_years, num_employees,
                    monthly_marketing_spend_kes, avg_daily_foot_traffic,
                    online_presence_score, interest_rate_pct,
                    competitor_density, customer_satisfaction_score):
    require_filled(
        business_age_years=business_age_years, num_employees=num_employees,
        monthly_marketing_spend_kes=monthly_marketing_spend_kes,
        online_presence_score=online_presence_score,
        interest_rate_pct=interest_rate_pct,
        competitor_density=competitor_density,
        customer_satisfaction_score=customer_satisfaction_score,
    )
    try:
        input_row = pd.DataFrame([{
            "business_age_years": business_age_years,
            "monthly_marketing_spend_kes": monthly_marketing_spend_kes,
            "customer_satisfaction_score": customer_satisfaction_score,
            "num_employees": num_employees,
            "online_presence_score": online_presence_score,
            "competitor_density": competitor_density,
            # 0 / blank means "not applicable" (e.g. online-only business)
            "avg_daily_foot_traffic": avg_daily_foot_traffic if avg_daily_foot_traffic else np.nan,
            "interest_rate_pct": interest_rate_pct,
            "sector": sector,
            "county": county,
            "quarter": quarter,
            "owner_education_level": owner_education_level,
        }])
        predicted_log_revenue = regression_pipeline.predict(input_row)[0]
        # Model was trained on log1p(revenue); invert to report KES
        predicted_revenue_kes = float(np.expm1(predicted_log_revenue))
        return f"KES {predicted_revenue_kes:,.2f} / month"
    except Exception as e:
        raise gr.Error(f"Inference error: {e}")


# -----------------------------------------------------------------------
# 2. Classification: predict SME credit risk category
# -----------------------------------------------------------------------
def predict_credit_risk(sector, county, quarter, loan_purpose, owner_education_level,
                        business_age_years, num_employees, monthly_revenue_kes,
                        loan_amount_kes, loan_term_months, interest_rate_pct,
                        collateral_value_kes, debt_to_income_ratio, credit_score,
                        num_previous_loans, missed_payments_count,
                        online_presence_score, customer_satisfaction_score,
                        competitor_density):
    require_filled(
        business_age_years=business_age_years, num_employees=num_employees,
        monthly_revenue_kes=monthly_revenue_kes, loan_amount_kes=loan_amount_kes,
        loan_term_months=loan_term_months, interest_rate_pct=interest_rate_pct,
        debt_to_income_ratio=debt_to_income_ratio, credit_score=credit_score,
        num_previous_loans=num_previous_loans,
        missed_payments_count=missed_payments_count,
        online_presence_score=online_presence_score,
        customer_satisfaction_score=customer_satisfaction_score,
        competitor_density=competitor_density,
    )
    try:
        input_row = pd.DataFrame([{
            "sector": sector,
            "county": county,
            "quarter": quarter,
            "loan_purpose": loan_purpose,
            "owner_education_level": owner_education_level,
            "business_age_years": business_age_years,
            "num_employees": num_employees,
            "monthly_revenue_kes": monthly_revenue_kes,
            "loan_amount_kes": loan_amount_kes,
            "loan_term_months": loan_term_months,
            "interest_rate_pct": interest_rate_pct,
            # 0 / blank means unsecured loan (no collateral)
            "collateral_value_kes": collateral_value_kes if collateral_value_kes else np.nan,
            "debt_to_income_ratio": debt_to_income_ratio,
            "credit_score": credit_score,
            "num_previous_loans": num_previous_loans,
            "missed_payments_count": missed_payments_count,
            "online_presence_score": online_presence_score,
            "customer_satisfaction_score": customer_satisfaction_score,
            "competitor_density": competitor_density,
        }])
        predicted_class = classification_pipeline.predict(input_row)[0]
        predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
        class_labels = classification_pipeline.named_steps["model"].classes_
        probability_lines = "\n".join(
            f"  {label}: {prob:.1%}"
            for label, prob in zip(class_labels, predicted_probabilities)
        )
        return f"Predicted category: {predicted_class}\n\n{probability_lines}"
    except Exception as e:
        raise gr.Error(f"Inference error: {e}")


# -----------------------------------------------------------------------
# 3. Clustering: assign a new mall customer to an existing segment
# -----------------------------------------------------------------------
def predict_segment(age, annual_income_k, spending_score):
    require_filled(age=age, annual_income_k=annual_income_k, spending_score=spending_score)
    try:
        input_row = pd.DataFrame([{
            "Age": age,
            "Annual Income (k$)": annual_income_k,
            "Spending Score (1-100)": spending_score,
        }])
        scaled_input = clustering_scaler.transform(input_row)
        cluster = int(clustering_kmeans.predict(scaled_input)[0])
        return f"Assigned segment: Cluster {cluster}"
    except Exception as e:
        raise gr.Error(f"Inference error: {e}")


# -----------------------------------------------------------------------
# 4. Association Rule Mining: recommend products for a shopping cart
# -----------------------------------------------------------------------
def recommend_products(cart_text, top_n):
    cart = {sku.strip() for sku in (cart_text or "").split(",") if sku.strip()}
    if not cart:
        raise gr.Error("Enter at least one SKU, comma-separated (e.g. E02, O01).")
    top_n = int(top_n) if top_n else 3

    candidates = {}
    for antecedent, consequent_list in recommendation_lookup.items():
        if antecedent.issubset(cart):
            for sku, confidence, lift in consequent_list:
                if sku not in cart:
                    if sku not in candidates or confidence > candidates[sku][0]:
                        candidates[sku] = (confidence, lift)

    ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:top_n]
    if not ranked:
        return ("No recommendations for this cart. Try a SKU combination seen in "
                f"the training orders. Valid SKUs: {', '.join(sorted(map(str, sku_lookup.index)))}")

    return "\n".join(
        f"{sku_lookup.loc[sku, 'product_name']} ({sku})  --  "
        f"confidence {confidence:.1%}, lift {lift:.2f}"
        for sku, (confidence, lift) in ranked
    )


# -----------------------------------------------------------------------
# One gr.Blocks() app, four gr.Tab()s: one shared Space URL
# -----------------------------------------------------------------------
with gr.Blocks(title="BBT 4206 Model Validation Demos") as demo:
    gr.Markdown("# BBT 4206 -- Model Validation Demos\nSwitch tabs to try a different model.")

    with gr.Tab("SME Revenue Regressor"):
        gr.Interface(
            fn=predict_revenue,
            inputs=[
                gr.Dropdown(label="Sector", choices=SECTORS, value="Technology"),
                gr.Dropdown(label="County", choices=COUNTIES, value="Nairobi"),
                gr.Dropdown(label="Quarter", choices=QUARTERS, value="Q3"),
                gr.Dropdown(label="Owner Education Level", choices=EDUCATION, value="Diploma"),
                gr.Number(label="Business Age (years)", value=4.5),
                gr.Number(label="Number of Employees", value=12),
                gr.Number(label="Monthly Marketing Spend (KES)", value=45000),
                gr.Number(label="Avg Daily Foot Traffic (0 if not applicable)", value=0),
                gr.Number(label="Online Presence Score", value=78.0),
                gr.Number(label="Interest Rate (%)", value=14.5),
                gr.Number(label="Competitor Density", value=5),
                gr.Number(label="Customer Satisfaction Score", value=4.2),
            ],
            outputs=gr.Textbox(label="Predicted Monthly Revenue"),
            flagging_mode="never",
        )

    with gr.Tab("SME Credit Risk Classifier"):
        gr.Interface(
            fn=predict_credit_risk,
            inputs=[
                gr.Dropdown(label="Sector", choices=SECTORS, value="Retail"),
                gr.Dropdown(label="County", choices=COUNTIES, value="Nairobi"),
                gr.Dropdown(label="Quarter", choices=QUARTERS, value="Q2"),
                gr.Dropdown(label="Loan Purpose", choices=LOAN_PURPOSES, value="Working Capital"),
                gr.Dropdown(label="Owner Education Level", choices=EDUCATION, value="Secondary"),
                gr.Number(label="Business Age (years)", value=2.1),
                gr.Number(label="Number of Employees", value=4),
                gr.Number(label="Monthly Revenue (KES)", value=850000),
                gr.Number(label="Loan Amount (KES)", value=150000),
                gr.Number(label="Loan Term (months)", value=12),
                gr.Number(label="Interest Rate (%)", value=16.5),
                gr.Number(label="Collateral Value (0 if unsecured)", value=0),
                gr.Number(label="Debt-to-Income Ratio", value=0.62),
                gr.Number(label="Credit Score", value=590),
                gr.Number(label="Number of Previous Loans", value=1),
                gr.Number(label="Missed Payments Count", value=3),
                gr.Number(label="Online Presence Score", value=40.0),
                gr.Number(label="Customer Satisfaction Score", value=3.4),
                gr.Number(label="Competitor Density", value=6),
            ],
            outputs=gr.Textbox(label="Predicted Credit Risk Category"),
            flagging_mode="never",
        )

    with gr.Tab("Mall Customer Segmenter"):
        gr.Interface(
            fn=predict_segment,
            inputs=[
                gr.Number(label="Age", value=27),
                gr.Number(label="Annual Income (k$)", value=45),
                gr.Number(label="Spending Score (1-100)", value=72),
            ],
            outputs=gr.Textbox(label="Prediction"),
            flagging_mode="never",
        )

    with gr.Tab("E-commerce Recommender"):
        gr.Interface(
            fn=recommend_products,
            inputs=[
                gr.Textbox(label="Cart (comma-separated SKUs)", value="E02, O01"),
                gr.Number(label="Top N Recommendations", value=3, precision=0),
            ],
            outputs=gr.Textbox(label="Recommendations"),
            flagging_mode="never",
        )

if __name__ == "__main__":
    demo.launch()