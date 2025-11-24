import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# ----- 기본 포맷팅 -----
def format_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return "NA"
    if p < 0.001: return "<0.001"
    if p > 0.99: return ">0.99"
    return f"{p:.3f}"

def clean_time(s):
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s

def ensure_binary_event(col, events, censored):
    def _map(x):
        if x in events: return 1
        if x in censored: return 0
        return np.nan
    return col.apply(_map).astype(float)

# ----- 변수 처리 -----
def ordered_levels(series):
    vals = pd.Series(series.dropna().unique()).tolist()
    numeric, non = [], []
    for v in vals:
        try: numeric.append((float(str(v)), v))
        except: non.append(str(v))
    if len(numeric) == len(vals) and len(vals) > 0:
        numeric.sort(key=lambda x: x[0])
        return [v for _, v in numeric]
    return sorted([str(v) for v in vals], key=lambda x: str(x))

def make_dummies(df_in, var, levels):
    cat = pd.Categorical(df_in[var].astype(str), categories=[str(x) for x in levels], ordered=True)
    dmy = pd.get_dummies(cat, prefix=var, prefix_sep="=", drop_first=True, dtype=float)
    dmy.index = df_in.index
    return dmy

def dummy_colname(var, level):
    return f"{var}={str(level)}"

def suggest_variable_type_single(df, var, threshold=20):
    is_numeric = pd.api.types.is_numeric_dtype(df[var])
    many_unique = df[var].nunique() > threshold
    return "Continuous" if (is_numeric and many_unique) else "Categorical"

# ----- Forest Plot (오류 수정됨) -----
def plot_forest(df_res, title="Forest Plot", effect_col="HR"):
    """Forest Plot 그리기 (안전 장치 추가)"""
    df_plot = df_res.iloc[::-1].copy()
    
    fig, ax = plt.subplots(figsize=(6, len(df_plot) * 0.5 + 2))
    
    y_pos = np.arange(len(df_plot))
    
    # 1. Effect Size (HR or OR) - 보통 첫 번째 컬럼
    mid = df_plot[effect_col] if effect_col in df_plot.columns else df_plot.iloc[:, 0]
    
    # 2. CI Lower/Upper 찾기
    try:
        # 이름으로 찾기 시도
        lo_candidates = [c for c in df_plot.columns if any(x in c.lower() for x in ['lower', 'lo', 'min', '0'])]
        hi_candidates = [c for c in df_plot.columns if any(x in c.lower() for x in ['upper', 'hi', 'max', '1'])]
        
        # 이름으로 못 찾으면 위치(Index)로 강제 할당 (보통 2번째, 3번째가 CI임)
        if lo_candidates and hi_candidates:
            lo = df_plot[lo_candidates[0]]
            hi = df_plot[hi_candidates[0]]
        else:
            # Fallback: 컬럼 위치 기반 (0:OR, 1:Lower, 2:Upper, 3:P-val 라고 가정)
            if df_plot.shape[1] >= 3:
                lo = df_plot.iloc[:, 1]
                hi = df_plot.iloc[:, 2]
            else:
                raise ValueError("Columns not found")

        # 에러바 그리기
        xerr = [mid - lo, hi - mid]
        ax.errorbar(mid, y_pos, xerr=xerr, fmt='o', color='black', ecolor='gray', capsize=5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_plot.index)
        ax.axvline(1, color='red', linestyle='--')
        ax.set_xlabel(f"{effect_col} (95% CI)")
        ax.set_title(title)
        
    except Exception as e:
        ax.text(0.5, 0.5, f"Plot Error: {str(e)}", ha='center')
    
    return fig

# ----- 통계 분석 (Table 1) -----
def analyze_table1_robust(df, group_col, value_map, target_cols, user_cont_vars, user_cat_vars):
    result_rows = []
    group_values = list(value_map.keys())
    group_names = list(value_map.values())
    group_n = {g: (df[group_col] == g).sum() for g in group_values}
    
    final_col_order = ['Characteristic']
    for g, g_name in zip(group_values, group_names):
        final_col_order.append(f"{g_name} (n={group_n[g]})")
    final_col_order.extend(['p-value', 'Test Method'])

    for var in target_cols:
        if var == group_col: continue
        valid = df[df[group_col].isin(group_values)].dropna(subset=[var])
        if valid.empty: continue

        # 타입 결정
        if var in user_cont_vars: is_continuous = True
        elif var in user_cat_vars: is_continuous = False
        else: is_continuous = pd.api.types.is_numeric_dtype(valid[var]) and (valid[var].nunique() > 20)

        # 연속형
        if is_continuous:
            try: valid_numeric = pd.to_numeric(valid[var], errors='coerce')
            except: valid_numeric = valid[var]
            groups_data = [valid_numeric[valid[group_col] == g].dropna() for g in group_values]
            if any(len(g) == 0 for g in groups_data): continue 

            is_normal = True
            for g_dat in groups_data:
                if len(g_dat) < 3: is_normal = False; break
                if len(g_dat) < 5000:
                    try:
                        if stats.shapiro(g_dat)[1] < 0.05: is_normal = False
                    except: is_normal = False

            row = {'Characteristic': var}
            for g, g_name in zip(group_values, group_names):
                sub = valid_numeric[valid[group_col] == g].dropna()
                if len(sub) == 0: row[f"{g_name} (n={group_n[g]})"] = "NA"
                elif is_normal: row[f"{g_name} (n={group_n[g]})"] = f"{sub.mean():.1f} ± {sub.std():.1f}"
                else: row[f"{g_name} (n={group_n[g]})"] = f"{sub.median():.1f} [{sub.quantile(0.25):.1f}-{sub.quantile(0.75):.1f}]"

            p = np.nan; method = ""
            try:
                valid_groups = [g for g in groups_data if len(g) > 0]
                if len(valid_groups) == 2:
                    if is_normal:
                        eq_var = stats.levene(*valid_groups)[1] > 0.05
                        p = stats.ttest_ind(*valid_groups, equal_var=eq_var)[1]
                        method = "T-test" if eq_var else "Welch's T-test"
                    else:
                        p = stats.mannwhitneyu(*valid_groups)[1]
                        method = "Mann-Whitney"
                elif len(valid_groups) > 2:
                    if is_normal: p = stats.f_oneway(*valid_groups)[1]; method = "ANOVA"
                    else: p = stats.kruskal(*valid_groups)[1]; method = "Kruskal-Wallis"
            except: pass
            row['p-value'] = format_p(p); row['Test Method'] = method
            result_rows.append(row)
        
        # 범주형
        else:
            try:
                ct = pd.crosstab(valid[group_col], valid[var].astype(str))
                method = "Chi-square"; p = np.nan
                if ct.shape == (2, 2):
                    if ct.min().min() < 5: p = stats.fisher_exact(ct)[1]; method = "Fisher's Exact"
                    else: p = stats.chi2_contingency(ct, correction=True)[1]
                else: p = stats.chi2_contingency(ct)[1]

                row_head = {'Characteristic': var}
                for g, g_name in zip(group_values, group_names): row_head[f"{g_name} (n={group_n[g]})"] = ""
                row_head['p-value'] = format_p(p); row_head['Test Method'] = method
                result_rows.append(row_head)

                unique_levels = sorted(valid[var].astype(str).unique())
                for val in unique_levels:
                    row_sub = {'Characteristic': f"  {val}"}
                    for g, g_name in zip(group_values, group_names):
                        cnt = valid[(valid[group_col] == g) & (valid[var].astype(str) == val)].shape[0]
                        total = group_n[g]
                        pct = (cnt / total * 100) if total > 0 else 0
                        row_sub[f"{g_name} (n={group_n[g]})"] = f"{cnt} ({pct:.1f}%)"
                    row_sub['p-value'] = ""; row_sub['Test Method'] = ""
                    result_rows.append(row_sub)
            except: continue

    df_res = pd.DataFrame(result_rows)
    if not df_res.empty:
        cols_to_use = [c for c in final_col_order if c in df_res.columns]
        df_res = df_res[cols_to_use]
    return df_res, None

# ----- Cox Helper -----
def drop_constant_predictors(X, time_col, event_col):
    pred_cols = [c for c in X.columns if c not in [time_col, event_col]]
    keep = [c for c in pred_cols if X[c].nunique(dropna=True) > 1]
    return X[[time_col, event_col] + keep]

def drop_constant_cols(X):
    keep = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    return X[keep]

def select_penalizer_by_cv(X_all, time_col, event_col, grid=(0.0, 0.01, 0.05, 0.1, 0.2, 0.5), k=5, seed=42):
    if X_all.shape[0] < k + 2 or X_all[event_col].sum() < k: return None, {}
    idx = X_all.index.to_numpy(); rng = np.random.default_rng(seed); rng.shuffle(idx)
    folds = np.array_split(idx, k)
    scores = {}
    for pen in grid:
        cv_scores = []
        for i in range(k):
            test_idx = folds[i]; train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            train = X_all.loc[train_idx].copy(); test = X_all.loc[test_idx].copy()
            train = drop_constant_predictors(train, time_col, event_col); test = test[train.columns]
            if train[event_col].sum() < 2 or test[event_col].sum() < 1: continue
            if train.shape[1] <= 2 or train.shape[0] < 5: continue
            try:
                cph = CoxPHFitter(penalizer=pen)
                cph.fit(train, duration_col=time_col, event_col=event_col)
                s = float(cph.score(test, scoring_method="concordance_index"))
                if np.isfinite(s): cv_scores.append(s)
            except: continue
        if cv_scores: scores[pen] = float(np.mean(cv_scores))
    if not scores: return None, {}
    best_pen = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return best_pen, scores

# ----- PSM -----
def calculate_smd(df, treatment_col, covariate_cols):
    smd_data = []
    treated = df[df[treatment_col] == 1]; control = df[df[treatment_col] == 0]
    for col in covariate_cols:
        if df[col].nunique() > 2:
            m1, m2 = treated[col].mean(), control[col].mean()
            s1, s2 = treated[col].std(), control[col].std()
            pooled_sd = np.sqrt((s1**2 + s2**2) / 2)
            smd = (m1 - m2) / pooled_sd if pooled_sd != 0 else 0
        else:
            p1 = treated[col].mean(); p2 = control[col].mean()
            pooled_sd = np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2)
            smd = (p1 - p2) / pooled_sd if pooled_sd != 0 else 0
        smd_data.append({'Variable': col, 'SMD': abs(smd)})
    return pd.DataFrame(smd_data)

def run_psm(df, treatment_col, covariates, caliper=0.2):
    data = df[[treatment_col] + covariates].dropna()
    X = pd.get_dummies(data[covariates], drop_first=True, dtype=float)
    y = data[treatment_col]
    ps_model = LogisticRegression(solver='liblinear', random_state=42)
    ps_model.fit(X, y)
    data['propensity_score'] = ps_model.predict_proba(X)[:, 1]
    
    ps_score_clipped = np.clip(data['propensity_score'], 1e-6, 1-1e-6)
    data['logit_ps'] = np.log(ps_score_clipped / (1 - ps_score_clipped))
    
    treated = data[data[treatment_col] == 1]
    control = data[data[treatment_col] == 0]
    if treated.empty or control.empty: return None, None
    
    caliper_val = caliper * data['logit_ps'].std()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='euclidean')
    nbrs.fit(control[['logit_ps']])
    distances, indices = nbrs.kneighbors(treated[['logit_ps']])
    
    matched_indices = []
    used_control_indices = set()
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        control_idx = control.index[idx[0]]
        if dist[0] <= caliper_val and control_idx not in used_control_indices:
            matched_indices.append((treated.index[i], control_idx))
            used_control_indices.add(control_idx)
    
    if not matched_indices: return None, None
    matched_df = pd.concat([data.loc[[x[0] for x in matched_indices]], data.loc[[x[1] for x in matched_indices]]])
    matched_df_full = df.loc[matched_df.index].copy()
    matched_df_full['propensity_score'] = matched_df['propensity_score']
    return matched_df_full, data
