import io
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test, logrank_test
from lifelines.exceptions import ConvergenceError
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# ================== 페이지 설정 ==================
st.set_page_config(page_title="Dr.Stats Ultimate: Medical Statistics", layout="wide")

# ================== 1. 공통 유틸리티 함수 ==================

def format_p(p):
    """P-value를 논문 표준 포맷으로 변환"""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "NA"
    if p < 0.001:
        return "<0.001"
    if p > 0.99:
        return ">0.99"
    return f"{p:.3f}"

def check_vif(X):
    """다중공선성(VIF) 계산 함수"""
    if "const" not in X.columns:
        X_const = sm.add_constant(X)
    else:
        X_const = X
    
    X_numeric = X_const.select_dtypes(include=[np.number]).dropna()
    
    if X_numeric.empty:
        return pd.DataFrame({'Variable': [], 'VIF': []})

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_numeric.columns
    try:
        vif_data["VIF"] = [variance_inflation_factor(X_numeric.values, i) 
                           for i in range(X_numeric.shape[1])]
    except:
        vif_data["VIF"] = "Error"
        
    return vif_data[vif_data["Variable"] != "const"]

def ensure_binary_event(col, events, censored):
    """이벤트/센서링 값을 0/1로 변환"""
    def _map(x):
        if x in events: return 1
        if x in censored: return 0
        return np.nan
    return col.apply(_map).astype(float)

def ordered_levels(series):
    """범주형 변수의 레벨 정렬 (숫자 우선)"""
    vals = pd.Series(series.dropna().unique()).tolist()
    numeric, non = [], []
    for v in vals:
        try:
            numeric.append((float(str(v)), v))
        except:
            non.append(str(v))
    if len(numeric) == len(vals) and len(vals) > 0:
        numeric.sort(key=lambda x: x[0])
        return [v for _, v in numeric]
    return sorted([str(v) for v in vals], key=lambda x: str(x))

def make_dummies(df_in, var, levels):
    """더미 변수 생성"""
    cat = pd.Categorical(df_in[var].astype(str), categories=[str(x) for x in levels], ordered=True)
    dmy = pd.get_dummies(cat, prefix=var, prefix_sep="=", drop_first=True, dtype=float)
    dmy.index = df_in.index
    return dmy

def dummy_colname(var, level):
    return f"{var}={str(level)}"

def clean_time(s):
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s

def plot_forest(df_res, title="Forest Plot", effect_col="HR"):
    """Forest Plot 그리기"""
    df_plot = df_res.iloc[::-1].copy()
    
    fig, ax = plt.subplots(figsize=(6, len(df_plot) * 0.5 + 2))
    
    y_pos = np.arange(len(df_plot))
    mid = df_plot[effect_col] if effect_col in df_plot.columns else df_plot.iloc[:, 0]
    
    lo_col = [c for c in df_plot.columns if "lower" in c.lower() or "0" in str(c) or "Lower" in c][0]
    hi_col = [c for c in df_plot.columns if "upper" in c.lower() or "1" in str(c) or "Upper" in c][0]
    
    lo = df_plot[lo_col]
    hi = df_plot[hi_col]
    
    xerr = [mid - lo, hi - mid]
    
    ax.errorbar(mid, y_pos, xerr=xerr, fmt='o', color='black', ecolor='gray', capsize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot.index)
    ax.axvline(1, color='red', linestyle='--')
    ax.set_xlabel(f"{effect_col} (95% CI)")
    ax.set_title(title)
    
    return fig

# --- Cox 분석 유틸리티 (구버전 복원) ---
def drop_constant_cols(X):
    keep = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    return X[keep]

def drop_constant_predictors(X, time_col, event_col):
    pred_cols = [c for c in X.columns if c not in [time_col, event_col]]
    keep = [c for c in pred_cols if X[c].nunique(dropna=True) > 1]
    return X[[time_col, event_col] + keep]

# --- Cox Penalizer Auto-CV ---
def select_penalizer_by_cv(
    X_all, time_col, event_col,
    grid=(0.0, 0.01, 0.05, 0.1, 0.2, 0.5),
    k=5, seed=42
):
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
            test  = X_all.loc[test_idx].copy()

            train = drop_constant_predictors(train, time_col, event_col)
            test  = test[train.columns]

            if train[event_col].sum() < 2 or test[event_col].sum() < 1:
                continue
            if train.shape[1] <= 2 or train.shape[0] < 5:
                continue

            try:
                cph = CoxPHFitter(penalizer=pen)
                cph.fit(train, duration_col=time_col, event_col=event_col)
                s = float(cph.score(test, scoring_method="concordance_index"))
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

# ================== 2. Table 1 로직 ==================

def suggest_variable_type_single(df, var, threshold=20):
    """단일 변수 타입 제안"""
    is_numeric = pd.api.types.is_numeric_dtype(df[var])
    many_unique = df[var].nunique() > threshold
    return "Continuous" if (is_numeric and many_unique) else "Categorical"

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

        # 변수 타입 결정
        if var in user_cont_vars:
            is_continuous = True
        elif var in user_cat_vars:
            is_continuous = False
        else:
            is_continuous = pd.api.types.is_numeric_dtype(valid[var]) and (valid[var].nunique() > 20)

        # 1. 연속형 분석
        if is_continuous:
            try:
                valid_numeric = pd.to_numeric(valid[var], errors='coerce')
            except:
                valid_numeric = valid[var]

            groups_data = [valid_numeric[valid[group_col] == g].dropna() for g in group_values]
            
            if any(len(g) == 0 for g in groups_data):
                continue 

            is_normal = True
            for g_dat in groups_data:
                if len(g_dat) < 3: 
                    is_normal = False 
                    break
                if len(g_dat) < 5000:
                    try:
                        _, p_norm = stats.shapiro(g_dat)
                        if p_norm < 0.05: is_normal = False
                    except:
                        is_normal = False

            row = {'Characteristic': var}
            for g, g_name in zip(group_values, group_names):
                sub = valid_numeric[valid[group_col] == g].dropna()
                if len(sub) == 0:
                    row[f"{g_name} (n={group_n[g]})"] = "NA"
                elif is_normal:
                    row[f"{g_name} (n={group_n[g]})"] = f"{sub.mean():.1f} ± {sub.std():.1f}"
                else:
                    row[f"{g_name} (n={group_n[g]})"] = f"{sub.median():.1f} [{sub.quantile(0.25):.1f}-{sub.quantile(0.75):.1f}]"

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
            except:
                pass
            
            row['p-value'] = format_p(p)
            row['Test Method'] = method
            result_rows.append(row)

        # 2. 범주형 분석
        else:
            try:
                ct = pd.crosstab(valid[group_col], valid[var].astype(str))
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

                row_head = {'Characteristic': var}
                for g, g_name in zip(group_values, group_names):
                    row_head[f"{g_name} (n={group_n[g]})"] = ""
                row_head['p-value'] = format_p(p)
                row_head['Test Method'] = method
                
                result_rows.append(row_head)

                unique_levels = sorted(valid[var].astype(str).unique()) 
                for val in unique_levels:
                    row_sub = {'Characteristic': f"  {val}"}
                    for g, g_name in zip(group_values, group_names):
                        cnt = valid[(valid[group_col] == g) & (valid[var].astype(str) == val)].shape[0]
                        total = group_n[g]
                        pct = (cnt / total * 100) if total > 0 else 0
                        row_sub[f"{g_name} (n={group_n[g]})"] = f"{cnt} ({pct:.1f}%)"
                    row_sub['p-value'] = ""
                    row_sub['Test Method'] = ""
                    result_rows.append(row_sub)

            except Exception as e:
                return None, {"type": "unknown", "var": var, "msg": str(e)}

    df_res = pd.DataFrame(result_rows)
    if not df_res.empty:
        cols_to_use = [c for c in final_col_order if c in df_res.columns]
        df_res = df_res[cols_to_use]

    return df_res, None

# ================== 3. PSM 관련 함수 ==================

def calculate_smd(df, treatment_col, covariate_cols):
    """표준화된 차이(SMD) 계산"""
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
            pooled_sd = np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2)
            smd = (p1 - p2) / pooled_sd if pooled_sd != 0 else 0
        smd_data.append({'Variable': col, 'SMD': abs(smd)})
    return pd.DataFrame(smd_data)

def run_psm(df, treatment_col, covariates, caliper=0.2):
    """PSM 실행"""
    data = df[[treatment_col] + covariates].dropna()
    X = pd.get_dummies(data[covariates], drop_first=True, dtype=float)
    y = data[treatment_col]
    
    ps_model = LogisticRegression(solver='liblinear', random_state=42)
    ps_model.fit(X, y)
    ps_score = ps_model.predict_proba(X)[:, 1]
    data['propensity_score'] = ps_score
    
    ps_score_clipped = np.clip(ps_score, 1e-6, 1-1e-6)
    data['logit_ps'] = np.log(ps_score_clipped / (1 - ps_score_clipped))
    
    treated = data[data[treatment_col] == 1]
    control = data[data[treatment_col] == 0]
    
    if treated.empty or control.empty:
        return None, None
    
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
    
    if not matched_indices:
        return None, None
        
    treated_idx = [x[0] for x in matched_indices]
    control_idx = [x[1] for x in matched_indices]
    matched_df = pd.concat([data.loc[treated_idx], data.loc[control_idx]])
    matched_df_full = df.loc[matched_df.index].copy()
    matched_df_full['propensity_score'] = matched_df['propensity_score']
    
    return matched_df_full, data

# ================== 메인 앱 UI ==================

st.title("Dr.Stats Ultimate: Medical Statistics Tool")

# 파일 업로더 및 시트 선택
uploaded_file = st.file_uploader("📂 데이터 파일 업로드 (Excel/CSV)", type=['xlsx', 'xls', 'csv'])

if uploaded_file:
    # 파일 및 시트 처리
    selected_sheet = None
    
    if uploaded_file.name.endswith(('.xlsx', '.xls')):
        try:
            xl = pd.ExcelFile(uploaded_file)
            sheet_names = xl.sheet_names
            if len(sheet_names) > 1:
                selected_sheet = st.selectbox("📑 시트 선택", sheet_names)
            else:
                selected_sheet = sheet_names[0]
        except Exception as e:
            st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
            st.stop()
    
    file_id = f"{uploaded_file.name}_{selected_sheet if selected_sheet else 'csv'}_{uploaded_file.size}"
    
    if 'current_file_id' not in st.session_state or st.session_state['current_file_id'] != file_id:
        try:
            if selected_sheet:
                df_load = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            else:
                df_load = pd.read_csv(uploaded_file)
            
            df_load.columns = df_load.columns.astype(str).str.strip()
            
            st.session_state['df'] = df_load
            st.session_state['current_file_id'] = file_id
            
            if 'var_config_df' in st.session_state:
                del st.session_state['var_config_df']
            if 'current_target_hash' in st.session_state:
                del st.session_state['current_target_hash']
                
            st.rerun()
        except Exception as e:
            st.error(f"데이터 로드 실패: {e}")
            st.stop()

    df = st.session_state.get('df')

    if df is not None:
        # 데이터 에디터 (수정 기능)
        with st.expander("✏️ 원본 데이터 미리보기 및 수정", expanded=False):
            st.info("데이터 오류(문자/숫자 혼합)가 있으면 여기서 직접 수정하세요. 수정 시 즉시 반영됩니다.")
            edited_df = st.data_editor(st.session_state['df'], num_rows="dynamic", use_container_width=True, key='main_editor')
            if not edited_df.equals(st.session_state['df']):
                st.session_state['df'] = edited_df
                st.rerun()

        st.divider()

        # 탭 구성
        tab1, tab2, tab3, tab4, tab_methods = st.tabs([
            "📊 Table 1 (기초통계)", 
            "⏱️ Cox Regression", 
            "💊 Logistic Regression",
            "⚖️ PSM (매칭)",
            "📝 Methods 작문"
        ])

        # ------------------ TAB 1 ------------------
        with tab1:
            st.subheader("Table 1: Baseline Characteristics")
            group_col = st.selectbox("그룹 변수 선택", df.columns, key='t1_group')
            
            if group_col:
                unique_vals = df[group_col].dropna().unique()
                col1, col2 = st.columns(2)
                with col1:
                    selected_vals = st.multiselect("비교할 그룹 값 (2개 이상)", unique_vals, default=unique_vals[:2] if len(unique_vals)>=2 else unique_vals)
                
                all_cols = [c for c in df.columns if c != group_col]
                
                if 'var_config_df' not in st.session_state:
                    initial_data = []
                    for col in all_cols:
                        initial_data.append({
                            "Include": True,
                            "Variable": col,
                            "Type": suggest_variable_type_single(df, col)
                        })
                    st.session_state['var_config_df'] = pd.DataFrame(initial_data)
                
                st.write("---")
                st.markdown("#### ⚙️ 분석 변수 및 타입 설정")
                st.caption("💡 **Include 체크를 해제**하면 분석에서 제외되며, 화면이 흔들리지 않습니다.")
                
                col_btn1, col_btn2, _ = st.columns([0.15, 0.15, 0.7])
                if col_btn1.button("✅ 전체 선택", key='btn_select_all'):
                    st.session_state['var_config_df']['Include'] = True
                    st.rerun()
                
                if col_btn2.button("⬜ 전체 해제", key='btn_deselect_all'):
                    st.session_state['var_config_df']['Include'] = False
                    st.rerun()

                edited_config = st.data_editor(
                    st.session_state['var_config_df'],
                    column_config={
                        "Include": st.column_config.CheckboxColumn("Include?", width="small", default=True),
                        "Variable": st.column_config.TextColumn("Variable Name", width="medium", disabled=True),
                        "Type": st.column_config.SelectboxColumn("Data Type", width="medium", options=["Continuous", "Categorical"], required=True)
                    },
                    hide_index=True,
                    use_container_width=True,
                    num_rows="fixed", 
                    key='var_manager_editor'
                )
                
                st.session_state['var_config_df'] = edited_config

                selected_rows = edited_config[edited_config['Include'] == True]
                target_vars = selected_rows['Variable'].tolist()
                user_cont_vars = selected_rows[selected_rows['Type'] == 'Continuous']['Variable'].tolist()
                user_cat_vars = selected_rows[selected_rows['Type'] == 'Categorical']['Variable'].tolist()

                value_map = {v: str(v) for v in selected_vals}
                
                if len(selected_vals) >= 2 and target_vars:
                    if st.button("Table 1 생성", key='btn_t1'):
                        with st.spinner("분석 중... (정규성 검정 포함)"):
                            t1_res, error_info = analyze_table1_robust(
                                df, group_col, value_map, target_vars, 
                                user_cont_vars, user_cat_vars
                            )
                            
                            if error_info:
                                st.error(f"🚨 **데이터 오류: '{error_info['var']}'**")
                                st.warning(f"맨 위 '데이터 수정' 탭에서 값을 통일해주세요. 오류: {error_info['msg']}")
                            else:
                                st.dataframe(t1_res, use_container_width=True)
                                output = io.BytesIO()
                                # [수정] xlsxwriter 사용
                                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                    t1_res.to_excel(writer, index=False)
                                st.download_button("📥 엑셀 다운로드", output.getvalue(), "Table1_Robust.xlsx")

        # ------------------ TAB 2: Cox Regression (Old Logic restored) ------------------
        with tab2:
            st.header("논문 Table: Factor / Subgroup / HR(95%CI) / p-value")

            time_col  = st.selectbox("생존기간 변수명(time)", df.columns, key="cox_time_col")
            event_col = st.selectbox("Event 변수명", df.columns, key="cox_event_col")

            temp_df = df.copy()
            if event_col:
                unique_events = list(df[event_col].dropna().unique())
                st.write(f"이 변수의 실제 값: {unique_events}")
                selected_event    = st.multiselect("이벤트(사건) 값", unique_events, key='selected_event_val')
                selected_censored = st.multiselect("생존/관찰종결(censored) 값", unique_events, key='selected_censored_val')
                st.caption("※ 사건값과 검열값은 서로 겹치면 안 됩니다.")
                temp_df["__event_for_cox"] = ensure_binary_event(temp_df[event_col], set(selected_event), set(selected_censored))
            else:
                temp_df["__event_for_cox"] = np.nan

            candidate_vars = [c for c in df.columns if c not in [time_col, event_col]]
            variables = st.multiselect("분석 후보 변수 선택", candidate_vars, key="cox_variables")

            c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
            with c1:
                p_enter = st.number_input("다변량 포함 기준 p-enter (≤)", min_value=0.001, max_value=1.0, value=0.05, step=0.01)
            with c2:
                max_levels = st.number_input("범주형 판정 최대 고유값", min_value=2, max_value=50, value=10, step=1)
            with c3:
                auto_penal = st.checkbox("penalizer 자동 선택 (CV, C-index)", value=False)
            with c4:
                cv_k = st.number_input("CV folds (K)", min_value=3, max_value=10, value=5, step=1, disabled=not auto_penal)

            penalizer = st.number_input("penalizer (수렴 안정화)", min_value=0.0, max_value=5.0, value=0.1, step=0.1, disabled=auto_penal)

            def basic_clean_cox(df_in, time_col):
                out = df_in.copy()
                out[time_col] = clean_time(out[time_col])
                out = out[out[time_col] > 0]
                out = out.replace([np.inf, -np.inf], np.nan)
                return out

            if st.button("분석 실행 (Cox)"):
                if not selected_event or not selected_censored:
                    st.error("사건값과 검열값을 각각 최소 1개 이상 선택하세요."); st.stop()
                if set(selected_event) & set(selected_censored):
                    st.error("사건값과 검열값이 겹칩니다. 다시 선택하세요."); st.stop()

                temp_df2 = basic_clean_cox(temp_df, time_col).dropna(subset=[time_col, "__event_for_cox"])
                n_events = int(temp_df2["__event_for_cox"].sum())
                st.info(f"총 관측치: {temp_df2.shape[0]}, 이벤트 수: {n_events}")
                if n_events < 5:
                    st.warning("이벤트 수가 매우 적습니다.")

                # 1) Univariate
                uni_sum_dict = {}; uni_na_vars = []; cat_info = {}
                for var in variables:
                    try:
                        dat_raw = temp_df2[[time_col, "__event_for_cox", var]].copy()
                        dat_raw = dat_raw.dropna(subset=[var])
                        if dat_raw.empty: uni_na_vars.append(var); continue

                        if (dat_raw[var].dtype == "object") or (dat_raw[var].nunique(dropna=True) <= max_levels):
                            lvls = ordered_levels(dat_raw[var])
                            if len(lvls) < 2: uni_na_vars.append(var); continue
                            cat_info[var] = {"levels": lvls, "ref": lvls[0]}
                            dmy = make_dummies(dat_raw, var, lvls)
                            dat = pd.concat([dat_raw[[time_col, "__event_for_cox"]], dmy], axis=1)
                        else:
                            cat_info[var] = {"levels": None, "ref": None}
                            dat = dat_raw[[time_col, "__event_for_cox", var]].copy()
                            dat[var] = pd.to_numeric(dat[var], errors="coerce")

                        dat = drop_constant_cols(dat.dropna())
                        if (dat.shape[0] < 3) or (dat["__event_for_cox"].sum() < 1): uni_na_vars.append(var); continue

                        cph = CoxPHFitter(penalizer=penalizer)
                        cph.fit(dat, duration_col=time_col, event_col="__event_for_cox")
                        uni_sum_dict[var] = cph.summary.copy()
                    except: uni_na_vars.append(var)

                # Variable Selection
                univariate_pvals = {}
                for var, summ in uni_sum_dict.items():
                    if cat_info[var]["levels"] is None:
                        if var in summ.index: univariate_pvals[var] = float(summ.loc[var, "p"])
                    else:
                        p_min = min([float(r["p"]) for _, r in summ.iterrows()])
                        univariate_pvals[var] = p_min

                selected_vars = [v for v, p in univariate_pvals.items() if p <= p_enter]
                st.write(f"다변량 후보 변수(≤ {p_enter:.3f}): {selected_vars if selected_vars else '없음'}")

                # 2) Multivariate
                multi_sum = None; multi_na_vars = []; chosen_penalizer = penalizer

                if len(selected_vars) >= 1:
                    try:
                        dat_base = temp_df2[[time_col, "__event_for_cox"]].copy()
                        X_list = []
                        for var in selected_vars:
                            if cat_info.get(var, {}).get("levels") is None:
                                xi = pd.to_numeric(temp_df2[var], errors="coerce").to_frame(var)
                            else:
                                lvls = cat_info[var]["levels"]
                                xi = make_dummies(temp_df2[[var]], var, lvls)
                            X_list.append(xi)

                        X_all = pd.concat([dat_base] + X_list, axis=1).dropna()
                        X_all = drop_constant_predictors(X_all, time_col, "__event_for_cox")

                        if auto_penal and X_all["__event_for_cox"].sum() >= int(cv_k):
                            bp, pen_scores = select_penalizer_by_cv(X_all, time_col, "__event_for_cox", k=int(cv_k))
                            if bp is not None:
                                chosen_penalizer = float(bp)
                                st.success(f"Auto-CV 선택 penalizer = {chosen_penalizer}")

                        if (X_all.shape[0] >= 3) and (X_all["__event_for_cox"].sum() >= 1):
                            cph_multi = CoxPHFitter(penalizer=chosen_penalizer)
                            cph_multi.fit(X_all, duration_col=time_col, event_col="__event_for_cox")
                            multi_sum = cph_multi.summary.copy()
                        else:
                            multi_na_vars = selected_vars
                    except: multi_na_vars = selected_vars

                # 3) Output
                rows = []
                for var in variables:
                    rows.append({"Factor": var, "Subgroup": "", "Univariate analysis HR (95% CI)": "", "Multivariate analysis HR (95% CI)": ""})
                    
                    if (var in uni_na_vars) and ((multi_sum is None) or (var in multi_na_vars)):
                        rows.append({"Factor": "", "Subgroup": "Skipped/Error", "Univariate analysis HR (95% CI)": "NA", "Multivariate analysis HR (95% CI)": "NA"})
                        continue

                    if cat_info.get(var, {}).get("levels") is not None:
                        lvls = cat_info[var]["levels"]
                        rows.append({"Factor": "", "Subgroup": f"{lvls[0]} (Ref)", "Univariate analysis HR (95% CI)": "Ref.", "Multivariate analysis HR (95% CI)": "Ref."})
                        for lv in lvls[1:]:
                            cn = dummy_colname(var, lv)
                            u_res = "NA"; m_res = "NA"
                            if var in uni_sum_dict and cn in uni_sum_dict[var].index:
                                r = uni_sum_dict[var].loc[cn]
                                u_res = f"{r['exp(coef)']:.2f} ({r['exp(coef) lower 95%']:.2f}-{r['exp(coef) upper 95%']:.2f}) p={format_p(r['p'])}"
                            if multi_sum is not None and cn in multi_sum.index:
                                r = multi_sum.loc[cn]
                                m_res = f"{r['exp(coef)']:.2f} ({r['exp(coef) lower 95%']:.2f}-{r['exp(coef) upper 95%']:.2f}) p={format_p(r['p'])}"
                            rows.append({"Factor": "", "Subgroup": str(lv), "Univariate analysis HR (95% CI)": u_res, "Multivariate analysis HR (95% CI)": m_res})
                    else:
                        u_res = "NA"; m_res = "NA"
                        if var in uni_sum_dict:
                            r = uni_sum_dict[var].loc[var]
                            u_res = f"{r['exp(coef)']:.2f} ({r['exp(coef) lower 95%']:.2f}-{r['exp(coef) upper 95%']:.2f}) p={format_p(r['p'])}"
                        if multi_sum is not None and var in multi_sum.index:
                            r = multi_sum.loc[var]
                            m_res = f"{r['exp(coef)']:.2f} ({r['exp(coef) lower 95%']:.2f}-{r['exp(coef) upper 95%']:.2f}) p={format_p(r['p'])}"
                        rows.append({"Factor": "", "Subgroup": "", "Univariate analysis HR (95% CI)": u_res, "Multivariate analysis HR (95% CI)": m_res})

                res_df = pd.DataFrame(rows)
                st.write("**논문 제출용 테이블 (Univariate/Multivariate 병렬, Reference, Factor/수준구조)**")
                st.dataframe(res_df, use_container_width=True)

                output = io.BytesIO()
                # [수정] xlsxwriter 사용
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    res_df.to_excel(writer, index=False)
                st.download_button("📥 Cox 결과 저장", output.getvalue(), "Cox_Result.xlsx")

        # ------------------ TAB 3: Logistic Regression ------------------
        with tab3:
            st.subheader("Binary Logistic Regression")
            dep_var = st.selectbox("Y (종속변수)", df.columns, key='log_y')
            if dep_var:
                ev_vals = st.multiselect("Event(1)", df[dep_var].unique(), key='log_ev')
                ct_vals = st.multiselect("Control(0)", df[dep_var].unique(), key='log_ct')
                
                if ev_vals and ct_vals:
                    df_log = df.copy()
                    df_log['Y'] = ensure_binary_event(df_log[dep_var], set(ev_vals), set(ct_vals))
                    df_log = df_log.dropna(subset=['Y'])
                    
                    indep_vars = st.multiselect("X (독립변수)", [c for c in df.columns if c != dep_var], key='log_x')
                    col_l1, col_l2 = st.columns(2)
                    p_enter_log = col_l1.number_input("Stepwise P", 0.05, key='log_p')
                    forced_log = col_l2.multiselect("강제 포함", indep_vars, key='log_forced')
                    
                    if st.button("Logistic 분석 실행", key='btn_log'):
                        sig_vars_log = []
                        for var in indep_vars:
                            try:
                                temp_df = df_log[['Y', var]].dropna()
                                if temp_df.empty: continue
                                if temp_df[var].dtype == 'object' or temp_df[var].nunique() < 10:
                                    lvls = ordered_levels(temp_df[var])
                                    if len(lvls) < 2: continue
                                    X = make_dummies(temp_df, var, lvls)
                                else:
                                    X = pd.to_numeric(temp_df[var], errors='coerce').to_frame()
                                X = sm.add_constant(X)
                                model = sm.Logit(temp_df['Y'], X).fit(disp=0)
                                p_vals = [model.pvalues[c] for c in model.pvalues.index if c != 'const']
                                if p_vals and min(p_vals) < p_enter_log:
                                    sig_vars_log.append(var)
                            except: pass
                        
                        final_log_vars = list(set(sig_vars_log) | set(forced_log))
                        
                        if not final_log_vars:
                            st.warning("조건을 만족하는 변수가 없습니다.")
                        else:
                            X_list = []
                            for var in final_log_vars:
                                if df_log[var].dtype == 'object' or df_log[var].nunique() < 10:
                                    lvls = ordered_levels(df_log[var])
                                    X_list.append(make_dummies(df_log[[var]], var, lvls))
                                else:
                                    X_list.append(pd.to_numeric(df_log[var], errors='coerce'))
                            
                            X_multi = pd.concat(X_list, axis=1)
                            st.caption("VIF Check")
                            st.dataframe(check_vif(X_multi).T)
                            
                            X_multi = sm.add_constant(X_multi)
                            data_model = pd.concat([df_log['Y'], X_multi], axis=1).dropna()
                            try:
                                logit_model = sm.Logit(data_model['Y'], data_model.drop(columns=['Y'])).fit(disp=0)
                                
                                st.subheader("2. Multivariate Result (OR)")
                                params = logit_model.params
                                conf = logit_model.conf_int()
                                conf['OR'] = params.apply(np.exp)
                                conf['Lower'] = conf[0].apply(np.exp)
                                conf['Upper'] = conf[1].apply(np.exp)
                                conf['p'] = logit_model.pvalues
                                
                                res_df = conf[['OR', 'Lower', 'Upper', 'p']]
                                res_df = res_df.drop('const', errors='ignore')
                                st.dataframe(res_df.style.format("{:.3f}"))
                                st.subheader("🌲 Forest Plot (Odds Ratio)")
                                fig_forest = plot_forest(res_df, title="Forest Plot - Logistic Regression", effect_col="OR")
                                st.pyplot(fig_forest)

                                out_l = io.BytesIO()
                                # [수정] xlsxwriter 사용
                                with pd.ExcelWriter(out_l, engine='xlsxwriter') as writer:
                                    res_df.to_excel(writer, sheet_name="Logistic")
                                st.download_button("📥 로지스틱 저장", out_l.getvalue(), "Logistic.xlsx")
                            except Exception as e:
                                st.error(f"Error: {e}")

        # ------------------ TAB 4: PSM (Matched Table 1 Included) ------------------
        with tab4:
            st.header("⚖️ Propensity Score Matching")
            c_psm1, c_psm2 = st.columns(2)
            treat_col = c_psm1.selectbox("치료 변수 (0/1)", df.columns, key='psm_treat')
            
            is_binary = False
            if treat_col:
                vals = df[treat_col].dropna().unique()
                if len(vals) == 2:
                    is_binary = True
                    treat_1 = c_psm2.selectbox(f"치료군(1) 값", vals, key='psm_val1')
                else:
                    st.warning("치료 변수는 2개의 값이어야 합니다.")

            if is_binary:
                covariates = st.multiselect("매칭 공변량", [c for c in df.columns if c != treat_col], key='psm_cov')
                caliper = st.slider("Caliper", 0.0, 1.0, 0.2, 0.05)
                
                if st.button("PSM 실행", key='btn_psm'):
                    if not covariates:
                        st.error("공변량을 선택하세요.")
                    else:
                        with st.spinner("매칭 중..."):
                            df_psm = df.copy()
                            df_psm['__T'] = np.where(df_psm[treat_col] == treat_1, 1, 0)
                            matched_df, original_w_score = run_psm(df_psm, '__T', covariates, caliper)
                            
                            if matched_df is None:
                                st.error("매칭 실패: 조건을 완화하세요.")
                            else:
                                st.success(f"매칭 완료! (N={len(matched_df)})")
                                
                                smd_before = calculate_smd(original_w_score, '__T', covariates)
                                smd_after = calculate_smd(matched_df, '__T', covariates)
                                smd_merge = pd.merge(smd_before, smd_after, on='Variable', suffixes=('_Before', '_After'))
                                st.dataframe(smd_merge.style.format({'SMD_Before': '{:.3f}', 'SMD_After': '{:.3f}'}))
                                
                                fig_love, ax_love = plt.subplots(figsize=(8, len(covariates)*0.5 + 2))
                                sns.scatterplot(data=smd_merge, x='SMD_Before', y='Variable', label='Before', color='red')
                                sns.scatterplot(data=smd_merge, x='SMD_After', y='Variable', label='After', color='blue')
                                plt.axvline(0.1, color='gray', linestyle='--')
                                st.pyplot(fig_love)
                                
                                out_psm = io.BytesIO()
                                # [수정] xlsxwriter 사용
                                with pd.ExcelWriter(out_psm, engine='xlsxwriter') as writer:
                                    matched_df.drop(columns=['__T', 'logit_ps']).to_excel(writer, index=False)
                                st.download_button("📥 매칭 데이터 저장", out_psm.getvalue(), "Matched.xlsx")

                                # Matched Table 1
                                st.markdown("---")
                                st.subheader("📊 Matched Cohort Baseline Table (Table 1)")
                                
                                auto_c, auto_cat = [], []
                                for c in covariates:
                                    if pd.api.types.is_numeric_dtype(matched_df[c]) and matched_df[c].nunique() > 20:
                                        auto_c.append(c)
                                    else:
                                        auto_cat.append(c)
                                
                                mt_vals = matched_df[treat_col].unique()
                                val_map = {v: str(v) for v in mt_vals}
                                
                                mt1, err = analyze_table1_robust(matched_df, treat_col, val_map, covariates, auto_c, auto_cat)
                                
                                if err:
                                    st.error(f"Table 1 생성 오류: {err}")
                                else:
                                    st.dataframe(mt1, use_container_width=True)
                                    out_mt1 = io.BytesIO()
                                    # [수정] xlsxwriter 사용
                                    with pd.ExcelWriter(out_mt1, engine='xlsxwriter') as writer:
                                        mt1.to_excel(writer, index=False)
                                    st.download_button("📥 Matched Table 1 저장", out_mt1.getvalue(), "Table1_Matched.xlsx")

        with tab_methods:
            st.header("📝 Methods")
            st.text_area("Draft", methods_text, height=400)
