import streamlit as st

def render_tab5():
    st.header("📝 Methods Section Draft")
    st.info("논문 작성 시 참고할 수 있는 영문 초안입니다.")
    
    text = """
**Statistical Analysis**

Continuous variables were compared using the Student's t-test or the Mann-Whitney U test, as appropriate, and categorical variables were compared using the Chi-square test or Fisher's exact test. Normality of the data distribution was assessed using the Shapiro-Wilk test. Data are presented as mean ± standard deviation for normally distributed continuous variables, median [interquartile range] for non-normally distributed variables, and number (percentage) for categorical variables.

Survival analysis was performed using the Kaplan-Meier method, and differences between groups were assessed using the log-rank test. Hazard ratios (HRs) and 95% confidence intervals (CIs) were estimated using univariate and multivariate Cox proportional hazards models. Variables with a p-value < 0.05 in the univariate analysis or those considered clinically significant were included in the multivariate analysis.

To reduce selection bias, we performed Propensity Score Matching (PSM). Propensity scores were estimated using a logistic regression model based on baseline covariates. A 1:1 nearest neighbor matching algorithm with a caliper width of 0.2 standard deviations of the logit of the propensity score was used. The balance of covariates between groups was assessed using the Standardized Mean Difference (SMD), with an SMD < 0.1 indicating negligible imbalance.

All statistical analyses were performed using Python (version 3.x) with pandas, scipy, statsmodels, and lifelines libraries. A p-value < 0.05 was considered statistically significant.
    """
    st.text_area("Copy & Paste", text, height=400)