import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="BBT 4206 Model Validation Demos", layout="wide")

# -----------------------------------------------------------------------
# Locate the model folder: ./model, ../model (repo layout), or next to app.py
# -----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REG_FILE = "lasso_regressor_for_sme_revenue.joblib"
MODEL_DIR = next(
    p for p in (os.path.join(HERE, "model"), os.path.join(HERE, "..", "model"), HERE)
    if os.path.exists(os.path.join(p, REG_FILE))
)


@st.cache_resource
def load_models():
    """Load all four artifacts once and reuse them across reruns and sessions."""
    load = lambda name: joblib.load(os.path.join(MODEL_DIR, name))
    clustering = load("kmeans_mall_customer_segmentation.joblib")
    arm = load("apriori_recommendation_rules.joblib")
    return {
        "reg": load(REG_FILE),
        "clf": load("svc_classifier_for_sme_credit_risk.joblib"),
        "scaler": clustering["scaler"],
        "kmeans": clustering["kmeans"],
        "reco": arm["recommendation_lookup"],
        "sku": arm["sku_lookup"],
    }


M = load_models()

# Categories exactly as the pipelines' encoders were fitted
SECTORS = ["Agriculture", "Hospitality", "Manufacturing", "Retail", "Services", "Technology"]
COUNTIES = ["Eldoret", "Kisumu", "Mombasa", "Nairobi", "Nakuru", "Other"]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
EDUCATION = ["Primary", "Secondary", "Certificate", "Diploma", "Bachelors", "Masters", "Doctorate"]
LOAN_PURPOSES = ["Business Expansion", "Equipment Purchase", "Inventory Financing", "Working Capital"]


def select(label, options, default, key):
    return st.selectbox(label, options, index=options.index(default), key=key)


def num(label, value, key, step=None):
    return st.number_input(label, value=value, step=step, key=key)


st.title("BBT 4206 -- Model Validation Demos")
st.caption("Switch tabs to try a different model.")
tab_reg, tab_clf, tab_clu, tab_rec = st.tabs([
    "SME Revenue Regressor",
    "SME Credit Risk Classifier",
    "Mall Customer Segmenter",
    "E-commerce Recommender",
])

# -----------------------------------------------------------------------
# 1. Regression: monthly revenue (KES)
# -----------------------------------------------------------------------
with tab_reg:
    c1, c2 = st.columns(2)
    with c1:
        sector = select("Sector", SECTORS, "Technology", "reg_sector")
        county = select("County", COUNTIES, "Nairobi", "reg_county")
        quarter = select("Quarter", QUARTERS, "Q3", "reg_quarter")
        education = select("Owner Education Level", EDUCATION, "Diploma", "reg_edu")
        age = num("Business Age (years)", 4.5, "reg_age")
        employees = num("Number of Employees", 12, "reg_emp", step=1)
    with c2:
        marketing = num("Monthly Marketing Spend (KES)", 45000.0, "reg_mkt")
        foot = num("Avg Daily Foot Traffic (0 if not applicable)", 0.0, "reg_foot")
        online = num("Online Presence Score", 78.0, "reg_online")
        interest = num("Interest Rate (%)", 14.5, "reg_int")
        competitors = num("Competitor Density", 5, "reg_comp", step=1)
        satisfaction = num("Customer Satisfaction Score", 4.2, "reg_sat")

    if st.button("Predict revenue", key="reg_btn", type="primary"):
        try:
            row = pd.DataFrame([{
                "business_age_years": age,
                "monthly_marketing_spend_kes": marketing,
                "customer_satisfaction_score": satisfaction,
                "num_employees": employees,
                "online_presence_score": online,
                "competitor_density": competitors,
                # 0 means "not applicable" (e.g. online-only business)
                "avg_daily_foot_traffic": foot if foot else np.nan,
                "interest_rate_pct": interest,
                "sector": sector,
                "county": county,
                "quarter": quarter,
                "owner_education_level": education,
            }])
            # Model was trained on log1p(revenue); invert to report KES
            revenue = float(np.expm1(M["reg"].predict(row)[0]))
            st.success(f"KES {revenue:,.2f} / month")
        except Exception as e:
            st.error(f"Inference error: {e}")

# -----------------------------------------------------------------------
# 2. Classification: credit risk category
# -----------------------------------------------------------------------
with tab_clf:
    c1, c2 = st.columns(2)
    with c1:
        sector = select("Sector", SECTORS, "Retail", "clf_sector")
        county = select("County", COUNTIES, "Nairobi", "clf_county")
        quarter = select("Quarter", QUARTERS, "Q2", "clf_quarter")
        purpose = select("Loan Purpose", LOAN_PURPOSES, "Working Capital", "clf_purpose")
        education = select("Owner Education Level", EDUCATION, "Secondary", "clf_edu")
        age = num("Business Age (years)", 2.1, "clf_age")
        employees = num("Number of Employees", 4, "clf_emp", step=1)
        revenue = num("Monthly Revenue (KES)", 850000.0, "clf_rev")
        loan_amt = num("Loan Amount (KES)", 150000.0, "clf_amt")
        term = num("Loan Term (months)", 12, "clf_term", step=1)
    with c2:
        interest = num("Interest Rate (%)", 16.5, "clf_int")
        collateral = num("Collateral Value (0 if unsecured)", 0.0, "clf_coll")
        dti = num("Debt-to-Income Ratio", 0.62, "clf_dti")
        credit = num("Credit Score", 590, "clf_credit", step=1)
        prev = num("Number of Previous Loans", 1, "clf_prev", step=1)
        missed = num("Missed Payments Count", 3, "clf_missed", step=1)
        online = num("Online Presence Score", 40.0, "clf_online")
        satisfaction = num("Customer Satisfaction Score", 3.4, "clf_sat")
        competitors = num("Competitor Density", 6, "clf_comp", step=1)

    if st.button("Predict credit risk", key="clf_btn", type="primary"):
        try:
            row = pd.DataFrame([{
                "sector": sector,
                "county": county,
                "quarter": quarter,
                "loan_purpose": purpose,
                "owner_education_level": education,
                "business_age_years": age,
                "num_employees": employees,
                "monthly_revenue_kes": revenue,
                "loan_amount_kes": loan_amt,
                "loan_term_months": term,
                "interest_rate_pct": interest,
                # 0 means unsecured loan (no collateral)
                "collateral_value_kes": collateral if collateral else np.nan,
                "debt_to_income_ratio": dti,
                "credit_score": credit,
                "num_previous_loans": prev,
                "missed_payments_count": missed,
                "online_presence_score": online,
                "customer_satisfaction_score": satisfaction,
                "competitor_density": competitors,
            }])
            label = M["clf"].predict(row)[0]
            probs = M["clf"].predict_proba(row)[0]
            classes = M["clf"].named_steps["model"].classes_
            st.success(f"Predicted category: {label}")
            prob_df = pd.DataFrame({
                "Category": [str(c) for c in classes],
                "Probability": [f"{p:.1%}" for p in probs],
            })
            st.dataframe(prob_df, hide_index=True)
            st.bar_chart(pd.Series(probs, index=[str(c) for c in classes]))
        except Exception as e:
            st.error(f"Inference error: {e}")

# -----------------------------------------------------------------------
# 3. Clustering: mall customer segment
# -----------------------------------------------------------------------
with tab_clu:
    age = num("Age", 27, "clu_age", step=1)
    income = num("Annual Income (k$)", 45, "clu_income", step=1)
    score = num("Spending Score (1-100)", 72, "clu_score", step=1)

    if st.button("Find segment", key="clu_btn", type="primary"):
        try:
            row = pd.DataFrame([{
                "Age": age,
                "Annual Income (k$)": income,
                "Spending Score (1-100)": score,
            }])
            cluster = int(M["kmeans"].predict(M["scaler"].transform(row))[0])
            st.success(f"Assigned segment: Cluster {cluster}")
        except Exception as e:
            st.error(f"Inference error: {e}")

# -----------------------------------------------------------------------
# 4. Association rules: product recommender
# -----------------------------------------------------------------------
with tab_rec:
    sku_lookup = M["sku"]
    cart = st.multiselect(
        "Cart (select SKUs)",
        options=list(sku_lookup.index),
        default=[s for s in ("E02", "O01") if s in sku_lookup.index],
        format_func=lambda s: f"{sku_lookup.loc[s, 'product_name']} ({s})",
        key="rec_cart",
    )
    top_n = st.slider("Top N Recommendations", 1, 10, 3, key="rec_topn")

    if st.button("Recommend", key="rec_btn", type="primary"):
        cart_set = set(cart)
        if not cart_set:
            st.error("Select at least one SKU.")
        else:
            try:
                candidates = {}
                for antecedent, consequents in M["reco"].items():
                    if antecedent.issubset(cart_set):
                        for sku, confidence, lift in consequents:
                            if sku not in cart_set:
                                if sku not in candidates or confidence > candidates[sku][0]:
                                    candidates[sku] = (confidence, lift)
                ranked = sorted(candidates.items(), key=lambda kv: kv[1][0], reverse=True)[:top_n]
                if not ranked:
                    st.info("No recommendations for this cart. Try a different SKU combination.")
                else:
                    st.dataframe(
                        pd.DataFrame([{
                            "sku": str(sku),
                            "product_name": sku_lookup.loc[sku, "product_name"],
                            "confidence": round(float(conf), 3),
                            "lift": round(float(lift), 2),
                        } for sku, (conf, lift) in ranked]),
                        hide_index=True,
                    )
            except Exception as e:
                st.error(f"Inference error: {e}")