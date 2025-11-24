import streamlit as st
import pandas as pd
import io
import xlsxwriter
from lifelines import CoxPHFitter
from .utils import (clean_time, ensure_binary_event, ordered_levels, make_dummies, 
                    drop_constant_cols, drop_constant_predictors, select_penalizer_by_cv, 
                    dummy_colname, format_p)

def render_tab2(df):
    st.subheader("Cox Proportional Hazards Model")
    c1, c2 = st.columns(2)
    time_col = c1.selectbox("Time", df.columns, key='cox_time')
    event_col = c2.selectbox("Event", df.columns, key='cox_event')

    temp_df = df.copy()
    if event_col:
        uniq_ev = list(df[event_col].dropna().unique())
        st.write(f"값: {uniq_ev}")
        sel_ev = st.multiselect("Event(1)", uniq_ev, key='cox_ev_val')
        sel_cen = st.multiselect("Censored(0)", uniq_ev, key='cox_cen_val')
        temp_df["__event_for_cox"] = ensure_binary_event(temp_df[event_col], set(sel_ev), set(sel_cen))
    else: temp_df["__event_for_cox"] = np.nan

    vars_cox = st.multiselect("분석 변수", [c for c in df.columns if c not in [time_col, event_col]], key='cox_vars')
    
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    p_ent = c1.number_input("Stepwise P", 0.05)
    max_lv = c2.number_input("Max Levels", 10)
    auto_p = c3.checkbox("Auto Penalizer", False)
    cv_k = c4.number_input("CV K", 5, disabled=not auto_p)
    pen_val = st.number_input("Penalizer", 0.0, 5.0, 0.1, disabled=auto_p)

    if st.button("분석 실행 (Cox)"):
        if not sel_ev or not sel_cen: st.error("Event/Censored 선택 필수"); st.stop()
        df2 = clean_time(temp_df, time_col).dropna(subset=[time_col, "__event_for_cox"])
        st.info(f"N={len(df2)}, Events={int(df2['__event_for_cox'].sum())}")

        # Univariate
        uni_sum = {}; uni_na = []; cat_info = {}
        for v in vars_cox:
            try:
                dr = df2[[time_col, "__event_for_cox", v]].dropna()
                if dr.empty: uni_na.append(v); continue
                if dr[v].dtype=='object' or dr[v].nunique()<=max_lv:
                    lv = ordered_levels(dr[v])
                    if len(lv)<2: uni_na.append(v); continue
                    cat_info[v] = {"levels": lv, "ref": lv[0]}
                    dmy = make_dummies(dr, v, lv)
                    dt = pd.concat([dr[[time_col, "__event_for_cox"]], dmy], axis=1)
                else:
                    cat_info[v] = {"levels": None, "ref": None}
                    dt = dr.copy(); dt[v] = pd.to_numeric(dt[v], errors='coerce')
                dt = drop_constant_cols(dt.dropna())
                if dt.shape[0]<3 or dt["__event_for_cox"].sum()<1: uni_na.append(v); continue
                cph = CoxPHFitter(penalizer=pen_val); cph.fit(dt, time_col, "__event_for_cox")
                uni_sum[v] = cph.summary.copy()
            except: uni_na.append(v)

        # Selection
        uni_p = {}
        for v, s in uni_sum.items():
            if cat_info[v]["levels"] is None:
                if v in s.index: uni_p[v] = float(s.loc[v, "p"])
            else:
                uni_p[v] = min([float(r["p"]) for _, r in s.iterrows()])
        sel_vars = [v for v, p in uni_p.items() if p <= p_ent]
        st.write(f"후보 변수: {sel_vars}")

        # Multivariate
        multi_sum = None; multi_na = []; chosen_p = pen_val
        if sel_vars:
            try:
                XL = []
                for v in sel_vars:
                    if cat_info[v]["levels"] is None: XL.append(pd.to_numeric(df2[v], errors='coerce').to_frame(v))
                    else: XL.append(make_dummies(df2[[v]], v, cat_info[v]["levels"]))
                XA = pd.concat([df2[[time_col, "__event_for_cox"]]] + XL, axis=1).dropna()
                XA = drop_constant_predictors(XA, time_col, "__event_for_cox")
                
                if auto_p and XA["__event_for_cox"].sum() >= int(cv_k):
                    bp, _ = select_penalizer_by_cv(XA, time_col, "__event_for_cox", k=int(cv_k))
                    if bp is not None: chosen_p = float(bp); st.success(f"Auto-CV Penalizer: {chosen_p}")
                
                cm = CoxPHFitter(penalizer=chosen_p)
                cm.fit(XA, time_col, "__event_for_cox")
                multi_sum = cm.summary.copy()
            except: multi_na = sel_vars

        # Output
        rows = []
        for v in vars_cox:
            rows.append({"Factor": v, "Subgroup": "", "Uni HR (95% CI)": "", "Multi HR (95% CI)": ""})
            if v in uni_na and (multi_sum is None or v in multi_na): continue
            
            is_cat = cat_info.get(v, {}).get("levels") is not None
            if is_cat:
                lvls = cat_info[v]["levels"]
                rows.append({"Factor": "", "Subgroup": f"{lvls[0]} (Ref)", "Uni HR (95% CI)": "Ref", "Multi HR (95% CI)": "Ref"})
                for lv in lvls[1:]:
                    cn = dummy_colname(v, lv)
                    u_res = "NA"; m_res = "NA"
                    if v in uni_sum and cn in uni_sum[v].index:
                        r = uni_sum[v].loc[cn]
                        u_res = f"{r['exp(coef)']:.2f} ({r['exp(coef) lower 95%']:.2f}-{r['exp(coef) upper 95%']:.2f}) p={format_p(r['p'])}"
                    if multi_sum is not None and cn in multi_sum.index:
                        r = multi_sum.loc[cn]
                        m_res = f"{r['exp(coef)']:.2f} ({r['exp(coef) lower 95%']:.2f}-{r['exp(coef) upper 95%']:.2f}) p={format_p(r['p'])}"
                    rows.append({"Factor": "", "Subgroup": str(lv), "Uni HR (95% CI)": u_res, "Multi HR (95% CI)": m_res})
            else:
                u_res = "NA"; m_res = "NA"
                if v in uni_sum:
                    r = uni_sum[v].loc[v]
                    u_res = f"{r['exp(coef)']:.2f} ({r['exp(coef) lower 95%']:.2f}-{r['exp(coef) upper 95%']:.2f}) p={format_p(r['p'])}"
                if multi_sum is not None and v in multi_sum.index:
                    r = multi_sum.loc[v]
                    m_res = f"{r['exp(coef)']:.2f} ({r['exp(coef) lower 95%']:.2f}-{r['exp(coef) upper 95%']:.2f}) p={format_p(r['p'])}"
                rows.append({"Factor": "", "Subgroup": "", "Uni HR (95% CI)": u_res, "Multi HR (95% CI)": m_res})

        res_df = pd.DataFrame(rows)
        st.dataframe(res_df)
        out_c = io.BytesIO()
        with pd.ExcelWriter(out_c, engine='xlsxwriter') as w: res_df.to_excel(w, index=False)
        st.download_button("📥 Cox 저장", out_c.getvalue(), "Cox.xlsx")