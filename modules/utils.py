import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test


# ==========================================
# Helper Functions
# ==========================================

def format_p(p: float) -> str:
    """Formats a p-value as a string."""
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    if p > 0.99:
        return ">0.99"
    return f"{p:.3f}"


def clean_time(s: pd.Series) -> pd.Series:
    """Cleans a time series by converting to numeric and handling infinities."""
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s


def ensure_binary_event(col: pd.Series, events: set, censored: set) -> pd.Series:
    """
    Maps a column to binary 1 (event) and 0 (censored/control).

    - 원 데이터가 숫자든 문자열이든 상관없이
      문자열(str)로 변환해서 비교하므로, UI에서 선택한 값과 안전하게 매칭됩니다.
    """
    events_str = {str(v).strip() for v in events}
    censored_str = {str(v).strip() for v in censored}

    def _map(x):
        sx = str(x).strip()
        if sx in events_str:
            return 1
        if sx in censored_str:
            return 0
        return np.nan

    return col.apply(_map).astype(float)


# ==========================================
# Variable Processing
# ==========================================

def ordered_levels(series: pd.Series) -> list:
    """Returns a sorted list of unique values from a series.

    - Numeric only → numeric sort
    - Mixed/strings → alphabetical
    """
    vals = pd.Series(series.dropna().unique()).tolist()
    numeric, non = [], []
    for v in vals:
        try:
            numeric.append((float(str(v)), v))
        except ValueError:
            non.append(str(v))

    if len(numeric) == len(vals) and len(vals) > 0:
        numeric.sort(key=lambda x: x[0])
        return [v for _, v in numeric]

    return sorted([str(v) for v in vals], key=lambda x: str(x))


def make_dummies(df_in: pd.DataFrame, var: str, levels: list) -> pd.DataFrame:
    """Creates dummy variables for a categorical variable."""
    cat = pd.Categorical(
        df_in[var].astype(str),
        categories=[str(x) for x in levels],
        ordered=True,
    )
    dmy = pd.get_dummies(cat, prefix=var, prefix_sep="=", drop_first=True, dtype=float)
    dmy.index = df_in.index
    return dmy


def dummy_colname(var: str, level: str) -> str:
    """Returns the column name for a dummy variable."""
    return f"{var}={str(level)}"


def suggest_variable_type_single(
    df: pd.DataFrame,
    var: str,
    threshold: int = 15,
) -> str:
    """
    간단한 자동 타입 추론 함수.

    - 수치형이고
    - 유니크 값 개수가 threshold(기본 15) 이상이면  → Continuous
    - 그 외                                        → Categorical
    """
    is_numeric = pd.api.types.is_numeric_dtype(df[var])
    many_unique = df[var].nunique(dropna=True) >= threshold
    return "Continuous" if (is_numeric and many_unique) else "Categorical"


# ==========================================
# Statistical Analysis (Table 1)
# ==========================================

def _analyze_continuous(
    valid: pd.DataFrame,
    var: str,
    group_col: str,
    group_values: list,
    group_names: list,
    group_n: dict,
) -> list:
    """Analyzes a continuous variable across different groups."""
    try:
        valid_numeric = pd.to_numeric(valid[var], errors="coerce")
    except Exception:
        valid_numeric = valid[var]

    groups_data = [
        valid_numeric[valid[group_col].astype(str) == g].dropna()
        for g in group_values
    ]

    if any(len(g) == 0 for g in groups_data):
        return []

    is_normal = True
    for g_dat in groups_data:
        if len(g_dat) < 3:
            is_normal = False
            break
        if len(g_dat) < 5000:
            try:
                if stats.shapiro(g_dat)[1] < 0.05:
                    is_normal = False
            except Exception:
                is_normal = False

    row = {"Characteristic": var}
    for g, g_name in zip(group_values, group_names):
        sub = valid_numeric[valid[group_col].astype(str) == g].dropna()
        if len(sub) == 0:
            row[f"{g_name} (n={group_n[g]})"] = "NA"
        elif is_normal:
            row[f"{g_name} (n={group_n[g]})"] = f"{sub.mean():.1f} ± {sub.std():.1f}"
        else:
            row[f"{g_name} (n={group_n[g]})"] = (
                f"{sub.median():.1f} "
                f"[{sub.quantile(0.25):.1f}-{sub.quantile(0.75):.1f}]"
            )

    p = np.nan
    method = ""
    try:
        valid_groups = [g for g in groups_data if len(g) > 0]
        if len(valid_groups) < 2:
            p = np.nan
        elif len(valid_groups) == 2:
            if is_normal:
                eq_var_test = stats.levene(*valid_groups)
                eq_var = eq_var_test.pvalue > 0.05
                p = stats.ttest_ind(*valid_groups, equal_var=eq_var).pvalue
                method = "T-test" if eq_var else "Welch's T-test"
            else:
                p = stats.mannwhitneyu(*valid_groups).pvalue
                method = "Mann-Whitney"
        elif len(valid_groups) > 2:
            if is_normal:
                p = stats.f_oneway(*valid_groups).pvalue
                method = "ANOVA"
            else:
                p = stats.kruskal(*valid_groups).pvalue
                method = "Kruskal-Wallis"
    except Exception:
        pass

    row["p-value"] = format_p(p)
    row["Test Method"] = method
    return [row]


def _analyze_categorical(
    valid: pd.DataFrame,
    var: str,
    group_col: str,
    group_values: list,
    group_names: list,
    group_n: dict,
) -> list:
    """Analyzes a categorical variable across different groups."""
    rows = []
    try:
        ct = pd.crosstab(valid[group_col].astype(str), valid[var].astype(str))
        method = "Chi-square"
        p = np.nan

        if ct.shape == (2, 2):
            if ct.min().min() < 5:
                p = stats.fisher_exact(ct)[1]
                method = "Fisher's Exact"
            else:
                p = stats.chi2_contingency(ct, correction=True)[1]
        else:
            p = stats.chi2_contingency(ct)[1]

        row_head = {"Characteristic": var}
        for g, g_name in zip(group_values, group_names):
            row_head[f"{g_name} (n={group_n[g]})"] = ""
        row_head["p-value"] = format_p(p)
        row_head["Test Method"] = method
        rows.append(row_head)

        unique_levels = sorted(valid[var].astype(str).unique())
        for val in unique_levels:
            row_sub = {"Characteristic": f"  {val}"}
            for g, g_name in zip(group_values, group_names):
                cnt = valid[
                    (valid[group_col].astype(str) == g)
                    & (valid[var].astype(str) == val)
                ].shape[0]
                total = group_n[g]
                pct = (cnt / total * 100) if total > 0 else 0
                row_sub[f"{g_name} (n={group_n[g]})"] = f"{cnt} ({pct:.1f}%)"
            row_sub["p-value"] = ""
            row_sub["Test Method"] = ""
            rows.append(row_sub)

        return rows
    except Exception as e:
        raise e


def analyze_table1_robust(
    df: pd.DataFrame,
    group_col: str,
    value_map: dict,
    target_cols: list,
    user_cont_vars: list,
    user_cat_vars: list,
) -> tuple:
    """
    Robust Table 1 generator.

    - 중복 인덱스 / 중복 컬럼이 있어도 에러 없이 동작하도록 방어 코드 포함
    - 그룹 값은 문자열로 통일해서 비교
    """
    df = df.copy()
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    df = df.reset_index(drop=True)

    if group_col not in df.columns:
        return None, {
            "type": "group_col_missing",
            "var": group_col,
            "msg": f"group_col '{group_col}' not found in DataFrame.",
        }

    value_map_str = {str(k): v for k, v in value_map.items()}
    group_values = list(value_map_str.keys())
    group_names = list(value_map_str.values())

    group_series = df[group_col].astype(str)
    group_n = {g: int((group_series == g).sum()) for g in group_values}

    result_rows = []
    final_col_order = ["Characteristic"]
    for g, g_name in zip(group_values, group_names):
        final_col_order.append(f"{g_name} (n={group_n[g]})")
    final_col_order.extend(["p-value", "Test Method"])

    for var in target_cols:
        if var == group_col:
            continue
        if var not in df.columns:
            continue

        try:
            mask = group_series.isin(group_values)
            valid = df.loc[mask].dropna(subset=[var])
        except Exception as e:
            return None, {
                "type": "filter_error",
                "var": var,
                "msg": f"Filtering error for variable '{var}': {e}",
            }

        if valid.empty:
            continue

        if var in user_cont_vars:
            is_continuous = True
        elif var in user_cat_vars:
            is_continuous = False
        else:
            is_continuous = pd.api.types.is_numeric_dtype(valid[var]) and (
                valid[var].nunique(dropna=True) > 20
            )

        try:
            if is_continuous:
                new_rows = _analyze_continuous(
                    valid, var, group_col, group_values, group_names, group_n
                )
            else:
                new_rows = _analyze_categorical(
                    valid, var, group_col, group_values, group_names, group_n
                )

            if new_rows:
                result_rows.extend(new_rows)
        except Exception as e:
            return None, {"type": "unknown", "var": var, "msg": str(e)}

    df_res = pd.DataFrame(result_rows)
    if not df_res.empty:
        cols_to_use = [c for c in final_col_order if c in df_res.columns]
        df_res = df_res[cols_to_use]

    return df_res, None


# ==========================================
# Cox Regression Helpers
# ==========================================

def drop_constant_predictors(
    X: pd.DataFrame, time_col: str, event_col: str
) -> pd.DataFrame:
    pred_cols = [c for c in X.columns if c not in [time_col, event_col]]
    keep = [c for c in pred_cols if X[c].nunique(dropna=True) > 1]
    return X[[time_col, event_col] + keep]


def drop_constant_cols(X: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    return X[keep]


def select_penalizer_by_cv(
    X_all: pd.DataFrame,
    time_col: str,
    event_col: str,
    grid: tuple = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5),
    k: int = 5,
    seed: int = 42,
) -> tuple:
    if X_all.shape[0] < k + 2 or X_all[event_col].sum() < k:
        return None, {}

    idx = X_all.index.to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    scores = {}

    for pen in grid:
        cv_scores = []
        for i in range(k):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

            train = X_all.loc[train_idx].copy()
            test = X_all.loc[test_idx].copy()

            train = drop_constant_predictors(train, time_col, event_col)
            valid_cols = [c for c in train.columns if c in test.columns]
            test = test[valid_cols]

            if train[event_col].sum() < 2 or test[event_col].sum() < 1:
                continue
            if train.shape[1] <= 2 or train.shape[0] < 5:
                continue

            try:
                cph = CoxPHFitter(penalizer=pen)
                cph.fit(train, duration_col=time_col, event_col=event_col)
                s = float(
                    cph.score(test, scoring_method="concordance_index")
                )
                if np.isfinite(s):
                    cv_scores.append(s)
            except Exception:
                continue

        if cv_scores:
            scores[pen] = float(np.mean(cv_scores))

    if not scores:
        return None, {}

    best_pen = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return best_pen, scores


# ==========================================
# Logistic & Plots
# ==========================================

def check_vif(X: pd.DataFrame) -> pd.DataFrame:
    if "const" not in X.columns:
        X_const = sm.add_constant(X)
    else:
        X_const = X

    X_numeric = X_const.select_dtypes(include=[np.number]).dropna()
    if X_numeric.empty:
        return pd.DataFrame({"Variable": [], "VIF": []})

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_numeric.columns
    try:
        vif_data["VIF"] = [
            variance_inflation_factor(X_numeric.values, i)
            for i in range(X_numeric.shape[1])
        ]
    except Exception:
        vif_data["VIF"] = "Error"

    return vif_data[vif_data["Variable"] != "const"]


def plot_forest(res_df: pd.DataFrame) -> plt.Figure | None:
    if res_df.empty:
        return None

    df_plot = res_df.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(df_plot) * 0.5)))
    y_pos = np.arange(len(df_plot))

    ax.errorbar(
        df_plot["OR"],
        y_pos,
        xerr=[
            df_plot["OR"] - df_plot["Lower"],
            df_plot["Upper"] - df_plot["OR"],
        ],
        fmt="o",
        color="black",
        ecolor="gray",
        capsize=5,
    )

    ax.axvline(x=1, color="red", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["Variable"])
    ax.set_xlabel("Odds Ratio (95% CI)")
    ax.set_title("Forest Plot")
    ax.grid(True, axis="x", alpha=0.3)

    for i, row in df_plot.iterrows():
        label = f"{row['OR']:.2f} ({row['Lower']:.2f}-{row['Upper']:.2f})"
        ax.text(
            max(df_plot["Upper"].max(), 1.5) * 1.05,
            i,
            label,
            va="center",
        )

    return fig


# ==========================================
# PSM (Propensity Score Matching)
# ==========================================

def calculate_smd(
    df: pd.DataFrame, treatment_col: str, covariate_cols: list
) -> pd.DataFrame:
    smd_data = []
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    for col in covariate_cols:
        if df[col].nunique() > 2:
            m1, m2 = treated[col].mean(), control[col].mean()
            s1, s2 = treated[col].std(), control[col].std()
            pooled_sd = np.sqrt((s1**2 + s2**2) / 2)
            smd = (m1 - m2) / pooled_sd if pooled_sd != 0 else 0
        else:
            p1 = treated[col].mean()
            p2 = control[col].mean()
            pooled_sd = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
            smd = (p1 - p2) / pooled_sd if pooled_sd != 0 else 0
        smd_data.append({"Variable": col, "SMD": abs(smd)})

    return pd.DataFrame(smd_data)


def run_psm(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: list,
    caliper: float = 0.2,
    ratio: int = 1,
    replace: bool = False,
) -> tuple:
    """
    Performs Propensity Score Matching using logistic regression to estimate propensity scores.

    caliper <= 0 이면 caliper 없이 매칭합니다.
    ratio : Treated 1명당 Control 몇 명을 매칭할지 (1 = 1:1).
    replace : 같은 control이 여러 treated에 매칭될 수 있는지 여부.
    """
    data = df[[treatment_col] + covariates].dropna()
    if data.empty:
        return None, None

    X = pd.get_dummies(data[covariates], drop_first=True, dtype=float)
    y = data[treatment_col]

    if y.nunique() < 2:
        return None, None

    try:
        ps_model = LogisticRegression(solver="liblinear", random_state=42)
        ps_model.fit(X, y)
    except Exception:
        return None, None

    data = data.copy()
    data["propensity_score"] = ps_model.predict_proba(X)[:, 1]

    ps_clip = np.clip(data["propensity_score"], 1e-6, 1 - 1e-6)
    data["logit_ps"] = np.log(ps_clip / (1 - ps_clip))

    treated = data[data[treatment_col] == 1]
    control = data[data[treatment_col] == 0]

    if treated.empty or control.empty:
        return None, None

    std_logit = data["logit_ps"].std()
    if caliper is None or caliper <= 0 or std_logit == 0 or np.isnan(std_logit):
        caliper_val = None
    else:
        caliper_val = caliper * std_logit

    ratio = max(1, int(ratio))
    n_neighbors = min(ratio, len(control))
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm="ball_tree", metric="euclidean"
    )
    nbrs.fit(control[["logit_ps"]])
    distances, indices = nbrs.kneighbors(treated[["logit_ps"]])

    matched_pairs = []
    used_controls = set()

    for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
        t_idx = treated.index[i]
        num_matched = 0
        for dist, idx in zip(dist_row, idx_row):
            c_idx = control.index[idx]

            if caliper_val is not None and dist > caliper_val:
                continue
            if not replace and c_idx in used_controls:
                continue

            matched_pairs.append((t_idx, c_idx))
            used_controls.add(c_idx)
            num_matched += 1
            if num_matched >= ratio:
                break

    if not matched_pairs:
        return None, None

    matched_idx = [t for t, _ in matched_pairs] + [c for _, c in matched_pairs]
    matched_df = data.loc[matched_idx]

    matched_df_full = df.loc[matched_df.index].copy()
    matched_df_full["propensity_score"] = matched_df["propensity_score"]
    matched_df_full["logit_ps"] = matched_df["logit_ps"]

    return matched_df_full, data


# ==========================================
# Survival Analysis (KM & PH Check)
# ==========================================

def plot_km_survival(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    group_col: str | None = None,
    title: str = "Kaplan-Meier Survival Curve",
) -> plt.Figure:
    """Plots Kaplan-Meier survival curves (y-axis as %)."""
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(10, 6))

    if group_col:
        groups = sorted(df[group_col].dropna().unique())
        for g in groups:
            mask = df[group_col] == g
            kmf.fit(df[time_col][mask], df[event_col][mask], label=str(g))
            kmf.plot_survival_function(ax=ax, ci_show=True)

        if len(groups) == 2:
            g1 = df[df[group_col] == groups[0]]
            g2 = df[df[group_col] == groups[1]]
            res = logrank_test(
                g1[time_col],
                g2[time_col],
                event_observed_A=g1[event_col],
                event_observed_B=g2[event_col],
            )
            ax.text(
                0.05,
                0.05,
                f"Log-rank p = {format_p(res.p_value)}",
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.5),
            )
    else:
        kmf.fit(df[time_col], df[event_col], label="All")
        kmf.plot_survival_function(ax=ax, ci_show=True)

    ax.set_title(title)
    ax.set_xlabel("Time (month)")
    ax.set_ylabel("Survival probability (%)")

    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y*100:.0f}%" for y in yticks])

    ax.grid(True, alpha=0.3)
    return fig


def check_ph_assumption(
    cph_model: CoxPHFitter,
    df_train: pd.DataFrame,
    p_value_threshold: float = 0.05,
):
    try:
        results = proportional_hazard_test(cph_model, df_train, time_transform="km")
        return results
    except Exception as e:
        return str(e)


def plot_roc_curve(
    y_true: pd.Series,
    y_prob: pd.Series,
    title: str = "ROC Curve",
) -> tuple:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC curve (AUC = {roc_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return fig, roc_auc
