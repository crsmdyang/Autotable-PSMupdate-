import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter

# 현재 프로젝트 루트를 import 경로에 추가
sys.path.append(os.getcwd())

from modules.utils import (
    analyze_table1_robust,
    run_psm,
)

"""
Simple verification script for the medical statistics app.

- Creates a small fake cohort
- Runs:
  * Table 1 (analyze_table1_robust)
  * PSM (run_psm)
  * Logistic regression (statsmodels)
  * Cox regression (lifelines)
"""

# -----------------------------
# 1. Fake cohort data
# -----------------------------
np.random.seed(42)

N = 200
df = pd.DataFrame({
    "group": np.random.choice(["A", "B"], N),          # 그룹 변수 (Table 1 용)
    "age": np.random.normal(60, 10, N),                # 연속형
    "sex": np.random.choice(["M", "F"], N),            # 범주형
    "treatment": np.random.choice([0, 1], N),          # 0/1 처치 변수 (PSM, Cox, Logistic)
    "time": np.random.exponential(10, N),              # 생존 시간
    "event": np.random.binomial(1, 0.4, N),            # 사건 여부 (0/1)
})

print("[OK] Dummy cohort created: shape =", df.shape)

# -----------------------------
# 2. Table 1 test
# -----------------------------
print("\n--- Testing Table 1 ---")
try:
    val_map = {"A": "Group A", "B": "Group B"}
    target_vars = ["age", "sex"]
    user_cont = ["age"]
    user_cat = ["sex"]

    t1, err = analyze_table1_robust(
        df,
        group_col="group",
        value_map=val_map,
        target_cols=target_vars,
        user_cont_vars=user_cont,
        user_cat_vars=user_cat,
    )

    if err:
        print(f"[FAIL] Table 1 Error: {err}")
    else:
        print("[OK] Table 1 analysis successful")
        print(t1.head())
except Exception as e:
    print(f"[FAIL] Table 1 Exception: {e}")

# -----------------------------
# 3. PSM test
# -----------------------------
print("\n--- Testing PSM ---")
try:
    matched, org = run_psm(df, treatment_col="treatment", covariates=["age", "time"], caliper=0.2)
    if matched is None:
        print("[WARN] PSM returned None (no matches) – this can happen with random data.")
    else:
        print(f"[OK] PSM successful, matched N={len(matched)}")
except Exception as e:
    print(f"[FAIL] PSM Exception: {e}")

# -----------------------------
# 4. Logistic regression test
# -----------------------------
print("\n--- Testing Logistic Regression (statsmodels) ---")
try:
    log_df = df.copy()
    # 이진 결과 변수: event 사용
    log_df = log_df[["event", "age", "treatment"]].dropna()
    X = sm.add_constant(log_df[["age", "treatment"]])
    y = log_df["event"]

    if y.sum() < 5:
        print("[WARN] Too few events for logistic regression in dummy data.")
    else:
        model = sm.Logit(y, X).fit(disp=0)
        print("[OK] Logistic regression fit successful")
        print(model.summary2().tables[1].head())
except Exception as e:
    print(f"[FAIL] Logistic Exception: {e}")

# -----------------------------
# 5. Cox regression test
# -----------------------------
print("\n--- Testing Cox Regression (lifelines) ---")
try:
    cox_df = df.copy()
    cox_df = cox_df[["time", "event", "age", "treatment"]].dropna()

    if cox_df["event"].sum() < 5:
        print("[WARN] Too few events for Cox regression in dummy data.")
    else:
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col="time", event_col="event")
        print("[OK] Cox regression fit successful")
        print(cox_df.head())
        print(cph.summary.head())
except Exception as e:
    print(f"[FAIL] Cox Exception: {e}")

print("\n[OK] Verification script finished")
