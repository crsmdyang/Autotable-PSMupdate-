import streamlit as st

def render_tab5():
    st.header("📝 Methods Section Draft")
    st.info("논문 작성 시 참고할 수 있는 영문 초안입니다. 포함할 분석을 선택한 뒤 텍스트를 복사해서 사용하세요.")
    
    alpha = float(st.session_state.get("alpha", 0.05))
    raw_policy = st.session_state.get("missing_policy", "Complete-case (drop rows)")
    
    # Missing data description
    if raw_policy.startswith("Complete-case"):
        mp_desc = "a complete-case approach, excluding observations with missing values in the variables of interest"
    elif raw_policy.startswith("Categorical"):
        mp_desc = (
            "by treating missing values in categorical variables as a separate 'Missing' category, "
            "while leaving continuous variables unchanged"
        )
    elif raw_policy.startswith("Simple imputation"):
        mp_desc = (
            "by simple imputation, using the median for continuous variables and the mode for categorical variables"
        )
    else:  # Variable-wise or others
        mp_desc = "by excluding observations with missing values only from the specific analyses in which those variables were used"
    
    st.markdown("### 📌 Include Sections")
    use_baseline = st.checkbox("Baseline comparison (Table 1)", True)
    use_survival = st.checkbox("Survival analysis (KM & Cox regression)", True)
    use_logistic = st.checkbox("Binary logistic regression", True)
    use_psm = st.checkbox("Propensity score matching (PSM)", True)
    
    parts = []
    
    if use_baseline:
        parts.append(
f"""**Statistical Analysis**

Continuous variables were compared using the Student's t-test or the Mann-Whitney U test, as appropriate, and categorical variables were compared using the Chi-square test or Fisher's exact test. Normality of the data distribution was assessed using the Shapiro-Wilk test. Data are presented as mean ± standard deviation for normally distributed continuous variables, median [interquartile range] for non-normally distributed variables, and number (percentage) for categorical variables. Missing data were handled using {mp_desc}. A two-sided p-value < {alpha:.3f} was considered statistically significant."""
        )
    
    if use_survival:
        parts.append(
"""Survival analysis was performed using the Kaplan-Meier method, and differences between groups were assessed using the log-rank test. Hazard ratios (HRs) and 95% confidence intervals (CIs) were estimated using univariate and multivariate Cox proportional hazards models. Variables with a p-value below a predefined threshold in the univariate analysis or those considered clinically important were included in the multivariate model."""
        )
    
    if use_logistic:
        parts.append(
"""To evaluate factors associated with a binary outcome, univariate and multivariate binary logistic regression analyses were performed. Odds ratios (ORs) and 95% confidence intervals (CIs) were calculated. Variables with a p-value below a predefined threshold in the univariate analysis or those considered clinically relevant were entered into the multivariate model. Model performance was assessed using the receiver operating characteristic (ROC) curve and the area under the curve (AUC)."""
        )
    
    # PSM caliper: try to read from session_state if available
    caliper = st.session_state.get("psm_caliper", 0.2)
    if use_psm:
        parts.append(
f"""To reduce selection bias, propensity score matching (PSM) was performed. Propensity scores were estimated using a logistic regression model based on baseline covariates. A 1:1 nearest neighbor matching algorithm with a caliper width of {caliper:.2f} standard deviations of the logit of the propensity score was used. The balance of covariates between groups was assessed using the standardized mean difference (SMD), with an SMD < 0.1 indicating negligible imbalance."""
        )
    
    # Software paragraph (always included)
    parts.append(
"""All statistical analyses were performed using Python (version 3.x) with the pandas, scipy, statsmodels, scikit-learn, and lifelines libraries."""
    )
    
    full_text = "\n\n".join(parts)
    st.markdown("### ✂️ Copy & Paste")
    st.text_area("Methods Draft", full_text, height=400)
