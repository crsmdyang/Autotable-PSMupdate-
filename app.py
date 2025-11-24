import io
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# ================== 페이지 설정 ==================
st.set_page_config(page_title="Dr.Stats Pro: Medical Statistics & PSM", layout="wide")

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

# ================== 2. Table 1 로직 (에러 핸들링 강화) ==================

def analyze_table1_robust(df, group_col, value_map, threshold=20):
    result_rows = []
    group_values = list(value_map.keys())
    group_names = list(value_map.values())
    group_n = {g: (df[group_col] == g).sum() for g in group_values}
    
    for var in df.columns:
        if var == group_col: continue
        
        valid = df[df[group_col].isin(group_values)].dropna(subset=[var])
        if valid.empty: continue

        # --- 연속형/범주형 판단 ---
        is_numeric_type = pd.api.types.is_numeric_dtype(valid[var])
        many_unique = valid[var].nunique() > threshold

        # 연속형 변수 처리
        if is_numeric_type and many_unique:
            groups_data = [valid[valid[group_col] == g][var] for g in group_values]
            
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
                sub = valid[valid[group_col] == g][var]
                if is_normal:
                    row[f"{g_name} (n={group_n[g]})"] = f"{sub.mean():.1f} ± {sub.std():.1f}"
                else:
                    row[f"{g_name} (n={group_n[g]})"] = f"{sub.median():.1f} [{sub.quantile(0.25):.1f}-{sub.quantile(0.75):.1f}]"

            p = np.nan
            method = ""
            try:
                if len(groups_data) == 2:
                    if is_normal:
                        _, p_levene = stats.levene(*groups_data)
                        equal_var = (p_levene > 0.05)
                        _, p = stats.ttest_ind(*groups_data, equal_var=equal_var)
                        method = "T-test" if equal_var else "Welch's T-test"
                    else:
                        _, p = stats.mannwhitneyu(*groups_data)
                        method = "Mann-Whitney"
                elif len(groups_data) > 2:
                    if is_normal:
                        _, p = stats.f_oneway(*groups_data)
                        method = "ANOVA"
                    else:
                        _, p = stats.kruskal(*groups_data)
                        method = "Kruskal-Wallis"
            except:
                pass
            
            row['p-value'] = format_p(p)
            row['Test Method'] = method
            result_rows.append(row)

        # --- 범주형 변수 처리 ---
        else:
            try:
                # Crosstab 시도 (여기서 타입 혼합 에러 발생 가능)
                ct = pd.crosstab(valid[group_col], valid[var])
                
                method = "Chi-square"
                p = np.nan
                
                if ct.shape == (2, 2):
                    if ct.min().min() < 5:
                        _, p = stats.fisher_exact(ct)
                        method = "Fisher's Exact"
                    else:
                        _, p, _, _ = stats.chi2_contingency(ct, correction=True)
                else:
                    _, p, _, _ = stats.chi2_contingency(ct)

                row_head = {'Characteristic': var, 'p-value': format_p(p), 'Test Method': method}
                for g, g_name in zip(group_values, group_names):
                    row_head[f"{g_name} (n={group_n[g]})"] = ""
                result_rows.append(row_head)

                # 하위 레벨 정렬 (여기서도 에러 발생 가능)
                unique_levels = sorted(valid[var].unique()) 
                
                for val in unique_levels:
                    row_sub = {'Characteristic': f"  {val}"}
                    for g, g_name in zip(group_values, group_names):
                        cnt = valid[(valid[group_col] == g) & (valid[var] == val)].shape[0]
                        total = group_n[g]
                        pct = (cnt / total * 100) if total > 0 else 0
                        row_sub[f"{g_name} (n={group_n[g]})"] = f"{cnt} ({pct:.1f}%)"
                    row_sub['p-value'] = ""
                    row_sub['Test Method'] = ""
                    result_rows.append(row_sub)

            except TypeError as e:
                # Mixed Type 오류 발생 시, 분석을 중단하고 오류 정보를 리턴
                error_msg = str(e)
                # Pandas/Numpy 정렬 에러 메시지 패턴 확인
                if "not supported between instances" in error_msg or "orderable" in error_msg or "mixed types" in error_msg.lower():
                    types_found = valid[var].apply(type).unique()
                    types_str = [t.__name__ for t in types_found]
                    
                    return None, {
                        "type": "mixed_type",
                        "var": var,
                        "types": types_str,
                        "examples": valid[var].unique()[:5]
                    }
                else:
                    return None, {"type": "unknown", "var": var, "msg": error_msg}
            except Exception as e:
                return None, {"type": "unknown", "var": var, "msg": str(e)}

    return pd.DataFrame(result_rows), None

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

st.title("Dr.Stats Pro: Medical Statistics & PSM")
st.markdown("---")

uploaded_file = st.file_uploader("📂 데이터 파일 업로드 (Excel/CSV)", type=['xlsx', 'xls', 'csv'])

if uploaded_file:
    # 1. 초기 데이터 로드 (세션 상태가 없으면 로드)
    if 'df' not in st.session_state:
        if uploaded_file.name.endswith('.csv'):
            df_load = pd.read_csv(uploaded_file)
        else:
            df_load = pd.read_excel(uploaded_file)
        df_load.columns = df_load.columns.astype(str).str.strip()
        st.session_state['df'] = df_load
    
    # 2. 항상 세션 상태의 df를 사용 (수정 사항 반영을 위해)
    df = st.session_state['df']
    
    st.success(f"데이터 로드 완료: 총 {df.shape[0]} 행, {df.shape[1]} 열")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Table 1 (기초통계)", 
        "⏱️ Cox Regression (생존분석)", 
        "💊 Logistic Regression (위험인자)",
        "⚖️ PSM (성향점수매칭)"
    ])

    # ------------------ TAB 1: Baseline Characteristics ------------------
    with tab1:
        st.subheader("Table 1: 인구통계학적 특성 비교")
        group_col = st.selectbox("그룹 변수 선택 (Group Column)", df.columns, key='t1_group')
        if group_col:
            unique_vals = df[group_col].dropna().unique()
            col1, col2 = st.columns(2)
            with col1:
                selected_vals = st.multiselect("비교할 그룹 값 (2개 이상)", unique_vals, default=unique_vals[:2] if len(unique_vals)>=2 else unique_vals)
            
            value_map = {v: str(v) for v in selected_vals}
            
            if len(selected_vals) >= 2:
                if st.button("Table 1 생성", key='btn_t1'):
                    with st.spinner("분석 중... (정규성 검정 포함)"):
                        t1_res, error_info = analyze_table1_robust(df, group_col, value_map)
                        
                        if error_info:
                            # 🚨 에러 발생 시 수정 UI 표시
                            st.error(f"🚨 **데이터 오류: '{error_info['var']}' 컬럼**")
                            
                            if error_info['type'] == 'mixed_type':
                                st.warning(
                                    f"숫자와 문자가 섞여 있어 분석이 중단되었습니다.\n"
                                    f"아래 표에서 직접 데이터를 수정하고 **[수정 완료 및 재실행]** 버튼을 누르세요."
                                )
                                
                                # 수정할 데이터 준비 (그룹 변수와 문제 변수만 표시)
                                st.markdown(f"### ✏️ 데이터 직접 수정하기 ({error_info['var']})")
                                cols_to_edit = [group_col, error_info['var']]
                                
                                # data_editor 출력
                                edited_subset = st.data_editor(
                                    df[cols_to_edit], 
                                    key=f"editor_{error_info['var']}",
                                    num_rows="dynamic",
                                    use_container_width=True
                                )
                                
                                if st.button("✅ 수정 완료 및 재실행", type="primary"):
                                    # 세션 상태 업데이트
                                    st.session_state['df'][error_info['var']] = edited_subset[error_info['var']]
                                    st.success("수정 사항이 반영되었습니다. 페이지가 새로고침됩니다.")
                                    st.rerun()
                            else:
                                st.error(f"알 수 없는 오류: {error_info['msg']}")
                        else:
                            # ✅ 성공 시 결과 표시
                            st.dataframe(t1_res, use_container_width=True)
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                t1_res.to_excel(writer, index=False)
                            st.download_button("📥 엑셀 다운로드", output.getvalue(), "Table1_Robust.xlsx")

    # ------------------ TAB 2: Cox Regression ------------------
    with tab2:
        st.subheader("Cox Proportional Hazards Model")
        c1, c2 = st.columns(2)
        time_col = c1.selectbox("Time (생존/추적 기간)", df.columns, key='cox_time')
        event_col = c2.selectbox("Event (사건 발생 여부)", df.columns, key='cox_event')
        
        if event_col:
            events = st.multiselect("사건(Event=1) 값", df[event_col].dropna().unique(), key='cox_ev_val')
            censored = st.multiselect("검열(Censored=0) 값", df[event_col].dropna().unique(), key='cox_cen_val')
            
            if events and censored:
                df_cox = df.copy()
                df_cox['T'] = pd.to_numeric(df_cox[time_col], errors='coerce')
                df_cox['E'] = ensure_binary_event(df_cox[event_col], set(events), set(censored))
                df_cox = df_cox.dropna(subset=['T', 'E'])
                df_cox = df_cox[df_cox['T'] > 0] 

                predictors = st.multiselect("분석 변수 (Predictors)", [c for c in df.columns if c not in [time_col, event_col]])
                col_opt1, col_opt2 = st.columns(2)
                p_threshold = col_opt1.number_input("Stepwise P-value Cutoff", 0.01, 1.0, 0.05, 0.01, key='cox_p')
                forced_vars = col_opt2.multiselect("강제 포함 변수 (임상적 중요)", predictors, key='cox_force')
                
                if st.button("Cox 분석 실행", key='btn_cox'):
                    st.info(f"분석 대상 N={len(df_cox)}, Event 발생 수={int(df_cox['E'].sum())}")
                    
                    uni_res = {}
                    significant_vars = []
                    
                    for var in predictors:
                        try:
                            if df_cox[var].nunique() < 2: continue 
                            if df_cox[var].dtype == 'object' or df_cox[var].nunique() < 10:
                                lvls = ordered_levels(df_cox[var])
                                if len(lvls) < 2: continue
                                dmy = make_dummies(df_cox, var, lvls)
                                data = pd.concat([df_cox[['T', 'E']], dmy], axis=1).dropna()
                            else:
                                data = df_cox[['T', 'E', var]].copy()
                                data[var] = pd.to_numeric(data[var], errors='coerce')
                                data = data.dropna()
                            
                            cph = CoxPHFitter()
                            cph.fit(data, duration_col='T', event_col='E')
                            p_vals = cph.summary['p'].values
                            if min(p_vals) < p_threshold:
                                significant_vars.append(var)
                        except: pass
                    
                    final_vars = list(set(significant_vars) | set(forced_vars))
                    
                    if not final_vars:
                        st.warning("다변량 분석에 포함될 변수가 없습니다.")
                    else:
                        st.write("---")
                        st.markdown(f"**다변량 분석 변수:** {', '.join(final_vars)}")
                        X_multi_list = []
                        for var in final_vars:
                            if df_cox[var].dtype == 'object' or df_cox[var].nunique() < 10:
                                lvls = ordered_levels(df_cox[var])
                                X_multi_list.append(make_dummies(df_cox[[var]], var, lvls))
                            else:
                                X_multi_list.append(pd.to_numeric(df_cox[var], errors='coerce'))
                        
                        X_multi = pd.concat(X_multi_list, axis=1)
                        vif_df = check_vif(X_multi)
                        st.subheader("1. 다중공선성 진단 (VIF)")
                        st.dataframe(vif_df.T)
                        if isinstance(vif_df, pd.DataFrame) and 'VIF' in vif_df.columns and pd.to_numeric(vif_df['VIF'], errors='coerce').max() > 10:
                            st.error("⚠️ VIF > 10 인 변수가 있습니다. 다중공선성 문제가 의심되니 변수를 제거하세요.")

                        data_multi = pd.concat([df_cox[['T', 'E']], X_multi], axis=1).dropna()
                        try:
                            cph_multi = CoxPHFitter()
                            cph_multi.fit(data_multi, duration_col='T', event_col='E')
                            st.subheader("2. Multivariate Result")
                            st.dataframe(cph_multi.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']])
                            st.subheader("3. 비례위험 가정 검정 (PH Assumption)")
                            ph_test = proportional_hazard_test(cph_multi, data_multi, time_transform='km')
                            st.dataframe(ph_test.summary)
                            fig, ax = plt.subplots(figsize=(10, 6))
                            cph_multi.plot(ax=ax)
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Multivariate 분석 실패: {e}")

    # ------------------ TAB 3: Logistic Regression ------------------
    with tab3:
        st.subheader("Binary Logistic Regression")
        dep_var = st.selectbox("종속 변수 (Y)", df.columns, key='log_y')
        if dep_var:
            ev_vals = st.multiselect("Event(1) 값", df[dep_var].unique(), key='log_ev')
            ct_vals = st.multiselect("Control(0) 값", df[dep_var].unique(), key='log_ct')
            
            if ev_vals and ct_vals:
                df_log = df.copy()
                df_log['Y'] = ensure_binary_event(df_log[dep_var], set(ev_vals), set(ct_vals))
                df_log = df_log.dropna(subset=['Y'])
                
                indep_vars = st.multiselect("독립 변수 (X)", [c for c in df.columns if c != dep_var], key='log_x')
                col_l1, col_l2 = st.columns(2)
                p_enter_log = col_l1.number_input("Stepwise P-value Cutoff", 0.01, 1.0, 0.05, 0.01, key='log_p')
                forced_log = col_l2.multiselect("강제 포함 변수", indep_vars, key='log_forced')
                
                if st.button("로지스틱 분석 실행", key='btn_log'):
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
                        st.markdown(f"**다변량 모델 변수:** {', '.join(final_log_vars)}")
                        X_list = []
                        for var in final_log_vars:
                            if df_log[var].dtype == 'object' or df_log[var].nunique() < 10:
                                lvls = ordered_levels(df_log[var])
                                X_list.append(make_dummies(df_log[[var]], var, lvls))
                            else:
                                X_list.append(pd.to_numeric(df_log[var], errors='coerce'))
                        
                        X_multi = pd.concat(X_list, axis=1)
                        st.subheader("1. 다중공선성 진단 (VIF)")
                        vif_log = check_vif(X_multi)
                        st.dataframe(vif_log.T)
                        
                        X_multi = sm.add_constant(X_multi)
                        data_model = pd.concat([df_log['Y'], X_multi], axis=1).dropna()
                        try:
                            logit_model = sm.Logit(data_model['Y'], data_model.drop(columns=['Y'])).fit(disp=0)
                            st.subheader("2. Multivariate Result (OR & CI)")
                            params = logit_model.params
                            conf = logit_model.conf_int()
                            conf['OR'] = params.apply(np.exp)
                            conf['Lower'] = conf[0].apply(np.exp)
                            conf['Upper'] = conf[1].apply(np.exp)
                            conf['p'] = logit_model.pvalues
                            res_df = conf[['OR', 'Lower', 'Upper', 'p']]
                            res_df.columns = ['Odds Ratio', '95% CI Lower', '95% CI Upper', 'p-value']
                            st.dataframe(res_df.style.format("{:.3f}"))
                            st.subheader("3. 모델 적합도 (Pseudo R-sq)")
                            st.write(f"Pseudo R-squared: {logit_model.prsquared:.4f}")
                        except Exception as e:
                            st.error(f"로지스틱 분석 실패: {e}")

    # ------------------ TAB 4: PSM (Propensity Score Matching) ------------------
    with tab4:
        st.header("⚖️ 성향점수매칭 (Propensity Score Matching)")
        st.info("💡 교란변수(Confounders)를 통제하여 RCT와 유사한 효과를 내는 기법")

        c_psm1, c_psm2 = st.columns(2)
        treat_col = c_psm1.selectbox("치료/노출 변수 (Treatment, 0/1)", df.columns, key='psm_treat')
        
        is_binary = False
        if treat_col:
            vals = df[treat_col].dropna().unique()
            if len(vals) == 2:
                is_binary = True
                treat_1 = c_psm2.selectbox(f"치료군(1) 값 선택 (나머지는 대조군)", vals, key='psm_val1')
            else:
                st.warning("치료 변수는 반드시 2개의 값(이진 변수)만 가져야 합니다.")

        if is_binary:
            covariates = st.multiselect("매칭 공변량 (Covariates)", [c for c in df.columns if c != treat_col], key='psm_cov')
            caliper = st.slider("Caliper (SD of Logit PS)", 0.0, 1.0, 0.2, 0.05)
            
            if st.button("PSM 실행 (1:1 Matching)", key='btn_psm'):
                if not covariates:
                    st.error("공변량을 선택하세요.")
                else:
                    with st.spinner("매칭 및 밸런스 검증 중..."):
                        df_psm = df.copy()
                        df_psm['__T'] = np.where(df_psm[treat_col] == treat_1, 1, 0)
                        
                        matched_df, original_w_score = run_psm(df_psm, '__T', covariates, caliper)
                        
                        if matched_df is None:
                            st.error("매칭된 쌍이 하나도 없습니다! Caliper를 넓히거나 변수를 조정하세요.")
                        else:
                            n_treat = matched_df['__T'].sum()
                            n_control = len(matched_df) - n_treat
                            st.success(f"매칭 성공! (총 {len(matched_df)}명: 치료군 {n_treat}명 vs 대조군 {n_control}명)")
                            
                            st.subheader("1. 공변량 밸런스 체크 (SMD)")
                            st.caption("SMD (Standardized Mean Difference) < 0.1 이면 밸런스 양호")
                            smd_before = calculate_smd(original_w_score, '__T', covariates)
                            smd_after = calculate_smd(matched_df, '__T', covariates)
                            smd_merge = pd.merge(smd_before, smd_after, on='Variable', suffixes=('_Before', '_After'))
                            smd_merge['Balanced'] = np.where(smd_merge['SMD_After'] < 0.1, "✅ Good", "⚠️ Unbalanced")
                            st.dataframe(smd_merge.style.format({'SMD_Before': '{:.3f}', 'SMD_After': '{:.3f}'}))
                            
                            st.subheader("2. Love Plot (시각화)")
                            fig_love, ax_love = plt.subplots(figsize=(8, len(covariates)*0.5 + 2))
                            sns.scatterplot(data=smd_merge, x='SMD_Before', y='Variable', label='Before Matching', color='red', s=100, ax=ax_love)
                            sns.scatterplot(data=smd_merge, x='SMD_After', y='Variable', label='After Matching', color='blue', s=100, ax=ax_love)
                            plt.axvline(0.1, color='gray', linestyle='--')
                            plt.title("Standardized Mean Differences (SMD)")
                            plt.xlabel("Absolute SMD")
                            st.pyplot(fig_love)
                            
                            st.subheader("3. Propensity Score 분포")
                            fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
                            sns.kdeplot(data=original_w_score[original_w_score['__T']==1], x='propensity_score', fill=True, label='Treated (Before)', color='red', alpha=0.3)
                            sns.kdeplot(data=original_w_score[original_w_score['__T']==0], x='propensity_score', fill=True, label='Control (Before)', color='blue', alpha=0.3)
                            sns.kdeplot(data=matched_df[matched_df['__T']==1], x='propensity_score', color='red', linestyle='--', linewidth=2, label='Treated (Matched)')
                            sns.kdeplot(data=matched_df[matched_df['__T']==0], x='propensity_score', color='blue', linestyle='--', linewidth=2, label='Control (Matched)')
                            plt.legend()
                            st.pyplot(fig_hist)
                            
                            out_psm = io.BytesIO()
                            with pd.ExcelWriter(out_psm, engine='openpyxl') as writer:
                                matched_df.drop(columns=['__T', 'logit_ps']).to_excel(writer, index=False, sheet_name='Matched_Data')
                                smd_merge.to_excel(writer, index=False, sheet_name='Balance_Check')
                            st.download_button("📥 매칭된 데이터셋 다운로드 (Excel)", out_psm.getvalue(), "PSM_Matched_Data.xlsx")

else:
    st.info("👈 좌측 상단 메뉴 혹은 위쪽 버튼을 통해 데이터 파일을 업로드해주세요.")
