import io
import streamlit as st
import pandas as pd
import numpy as np
import xlsxwriter
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter

from .utils import (
    ensure_binary_event,
    clean_time,
    ordered_levels,
    make_dummies,
    drop_constant_cols,
    drop_constant_predictors,
    select_penalizer_by_cv,
    dummy_colname,
    format_p,
    check_ph_assumption,
    plot_km_survival,
)


def render_tab2(df: pd.DataFrame):
    """
    Renders the Cox Proportional Hazards Regression analysis tab.
    """
    st.subheader("⏱️ Cox Proportional Hazards Regression")

    # 1. Time & Event Selection
    c1, c2 = st.columns(2)
    time_col = c1.selectbox(
        "Time Variable",
        df.columns,
        key="cox_time",
        help="추적 기간(예: days, months 등)을 나타내는 변수입니다.",
    )
    event_col = c2.selectbox(
        "Event Variable",
        df.columns,
        key="cox_event",
        help="사건 발생 여부를 나타내는 변수입니다. (예: 사망, 재발 등)",
    )

    if not time_col or not event_col:
        st.info("Please select both Time and Event variables.")
        return

    # 2. Event Value Mapping
    uniq_ev = sorted(df[event_col].dropna().astype(str).unique())
    st.write(f"**Unique values in '{event_col}':** {uniq_ev}")

    c3, c4 = st.columns(2)
    sel_ev = c3.multiselect(
        "Event Value (Event=1)",
        uniq_ev,
        key="cox_ev_val",
        help="Event=1 로 간주할 값을 선택하세요. (예: Dead, Yes 등)",
    )
    sel_cen = c4.multiselect(
        "Censored Value (Censored=0)",
        uniq_ev,
        key="cox_cen_val",
        help="검열(censored, 사건 없음)로 간주할 값을 선택하세요. (예: Alive, No 등)",
    )

    if not sel_ev or not sel_cen:
        st.warning("Please select values for both Event and Censored.")
        return

    # Prepare Data for Cox
    temp_df = df.copy()
    temp_df["__event_for_cox"] = ensure_binary_event(
        temp_df[event_col], set(sel_ev), set(sel_cen)
    )
    temp_df[time_col] = clean_time(temp_df[time_col])

    df_cox = temp_df.dropna(subset=[time_col, "__event_for_cox"])
    n_events = int(df_cox["__event_for_cox"].sum())
    st.info(f"Analysis Data: N={len(df_cox)} (Events={n_events})")

    if len(df_cox) < 10 or n_events < 5:
        st.error("Insufficient data or events for analysis.")
        return

    # --- Kaplan-Meier Analysis ---
    st.markdown("---")
    st.subheader("📉 Kaplan-Meier Survival Curve")

    km_group = st.selectbox(
        "Group Variable (KM Curve)",
        [None] + [c for c in df.columns if c not in [time_col, event_col]],
        key="km_group",
        help="그룹별로 생존 곡선을 나누어 보고 싶을 때 선택합니다.",
    )

    if st.button("Draw KM Curve", key="btn_km"):
        fig = plot_km_survival(df_cox, time_col, "__event_for_cox", km_group)
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        st.download_button(
            "📥 Download KM plot (PNG)",
            buf.getvalue(),
            "KM_curve.png",
            mime="image/png",
        )

    # --- Cox Regression ---
    st.markdown("---")
    st.subheader("📊 Cox Regression Analysis")

    vars_cox = st.multiselect(
        "Select Covariates",
        [c for c in df.columns if c not in [time_col, event_col, "__event_for_cox"]],
        key="cox_vars",
        help="Cox 회귀에 포함할 공변량을 선택하세요.",
    )

    c5, c6, c7 = st.columns(3)
    p_ent = c5.number_input(
        "Univariate p-value threshold for entry to multivariate",
        min_value=0.01,
        max_value=1.0,
        value=0.1,
        step=0.01,
        key="cox_pent",
        help="단변량 분석에서 이 값보다 작은 변수만 다변량 모형에 포함합니다.",
    )
    auto_p = c6.checkbox(
        "Auto-Penalizer (CV)",
        value=False,
        key="cox_auto_p",
        help="체크 시, 교차검증으로 최적 penalizer를 자동 선택합니다.",
    )
    cv_k = c7.number_input(
        "CV Folds (k)",
        min_value=2,
        max_value=10,
        value=5,
        disabled=not auto_p,
        key="cox_cvk",
        help="교차검증 fold 수입니다.",
    )

    pen_val = 0.0
    if not auto_p:
        pen_val = st.number_input(
            "Penalizer (Fixed)",
            min_value=0.0,
            max_value=10.0,
            value=0.1,
            step=0.1,
            key="cox_pen_fixed",
            help="과적합을 줄이기 위한 L2 penalizer 값입니다. (기본 0.1)",
        )

    if st.button("Run Cox Analysis", key="btn_run_cox"):
        if not vars_cox:
            st.error("Please select at least one covariate.")
            return

        max_lv = 20

        uni_sum = {}
        uni_na = []
        cat_info = {}

        with st.spinner("Running Univariate Analysis..."):
            for v in vars_cox:
                try:
                    dr = df_cox[[time_col, "__event_for_cox", v]].dropna()
                    if dr.empty:
                        uni_na.append(v)
                        continue

                    if dr[v].dtype == "object" or dr[v].nunique() <= max_lv:
                        lv = ordered_levels(dr[v])
                        if len(lv) < 2:
                            uni_na.append(v)
                            continue
                        cat_info[v] = {"levels": lv, "ref": lv[0]}
                        dmy = make_dummies(dr, v, lv)
                        dt = pd.concat(
                            [dr[[time_col, "__event_for_cox"]], dmy], axis=1
                        )
                    else:
                        cat_info[v] = {"levels": None, "ref": None}
                        dt = dr.copy()
                        dt[v] = pd.to_numeric(dt[v], errors="coerce")

                    dt = drop_constant_cols(dt.dropna())
                    if dt.shape[0] < 5 or dt["__event_for_cox"].sum() < 2:
                        uni_na.append(v)
                        continue

                    cph = CoxPHFitter(penalizer=0.1)
                    cph.fit(dt, time_col, "__event_for_cox")
                    uni_sum[v] = cph.summary.copy()
                except Exception:
                    uni_na.append(v)

        if not uni_sum:
            st.warning("⚠️ No variables successfully passed univariate analysis.")
            return

        # Selection by p-value
        uni_p = {}
        for v, s in uni_sum.items():
            if cat_info[v]["levels"] is None:
                if v in s.index:
                    uni_p[v] = float(s.loc[v, "p"])
            else:
                uni_p[v] = min(float(r["p"]) for _, r in s.iterrows())

        sel_vars = [v for v, p in uni_p.items() if p <= p_ent]
        st.success(f"Variables entering multivariate (p < {p_ent}): {sel_vars}")

        multi_sum = None
        chosen_p = pen_val
        cm = None
        XA_train = None

        if sel_vars:
            with st.spinner("Running Multivariate Analysis..."):
                try:
                    XL = []
                    for v in sel_vars:
                        if cat_info[v]["levels"] is None:
                            XL.append(
                                pd.to_numeric(df_cox[v], errors="coerce").to_frame(v)
                            )
                        else:
                            XL.append(
                                make_dummies(
                                    df_cox[[v]],
                                    v,
                                    cat_info[v]["levels"],
                                )
                            )

                    XA = pd.concat(
                        [df_cox[[time_col, "__event_for_cox"]]] + XL, axis=1
                    ).dropna()
                    XA = drop_constant_predictors(XA, time_col, "__event_for_cox")

                    if auto_p and XA["__event_for_cox"].sum() >= int(cv_k):
                        bp, _ = select_penalizer_by_cv(
                            XA, time_col, "__event_for_cox", k=int(cv_k)
                        )
                        if bp is not None:
                            chosen_p = float(bp)
                            st.info(f"Auto-CV Penalizer Selected: {chosen_p}")

                    cm = CoxPHFitter(penalizer=chosen_p)
                    cm.fit(XA, time_col, "__event_for_cox")
                    multi_sum = cm.summary.copy()
                    XA_train = XA
                except Exception as e:
                    st.error(f"Multivariate Error: {e}")

        # 결과 테이블 (모든 선택 변수 uni/multi 둘 다 보여줌)
        rows = []
        for v in vars_cox:
            u_res = "NA"
            if v in uni_sum:
                if cat_info[v]["levels"] is None and v in uni_sum[v].index:
                    r = uni_sum[v].loc[v]
                    u_res = (
                        f"{r['exp(coef)']:.2f} "
                        f"({r['exp(coef) lower 95%']:.2f}"
                        f"-{r['exp(coef) upper 95%']:.2f}) "
                        f"p={format_p(r['p'])}"
                    )

            m_res = "NA"
            if multi_sum is not None:
                if cat_info[v]["levels"] is None and v in multi_sum.index:
                    r = multi_sum.loc[v]
                    m_res = (
                        f"{r['exp(coef)']:.2f} "
                        f"({r['exp(coef) lower 95%']:.2f}"
                        f"-{r['exp(coef) upper 95%']:.2f}) "
                        f"p={format_p(r['p'])}"
                    )

            if cat_info.get(v, {}).get("levels") is not None:
                lvls = cat_info[v]["levels"]
                rows.append(
                    {
                        "Factor": v,
                        "Subgroup": f"{lvls[0]} (Ref)",
                        "Uni HR (95% CI)": "Ref",
                        "Multi HR (95% CI)": "Ref",
                    }
                )
                for lv in lvls[1:]:
                    cn = dummy_colname(v, lv)

                    ur = "NA"
                    if v in uni_sum and cn in uni_sum[v].index:
                        r = uni_sum[v].loc[cn]
                        ur = (
                            f"{r['exp(coef)']:.2f} "
                            f"({r['exp(coef) lower 95%']:.2f}"
                            f"-{r['exp(coef) upper 95%']:.2f}) "
                            f"p={format_p(r['p'])}"
                        )

                    mr = "NA"
                    if multi_sum is not None and cn in multi_sum.index:
                        r = multi_sum.loc[cn]
                        mr = (
                            f"{r['exp(coef)']:.2f} "
                            f"({r['exp(coef) lower 95%']:.2f}"
                            f"-{r['exp(coef) upper 95%']:.2f}) "
                            f"p={format_p(r['p'])}"
                        )

                    rows.append(
                        {
                            "Factor": "",
                            "Subgroup": str(lv),
                            "Uni HR (95% CI)": ur,
                            "Multi HR (95% CI)": mr,
                        }
                    )
            else:
                rows.append(
                    {
                        "Factor": v,
                        "Subgroup": "",
                        "Uni HR (95% CI)": u_res,
                        "Multi HR (95% CI)": m_res,
                    }
                )

        res_df = pd.DataFrame(rows)
        st.dataframe(res_df, use_container_width=True)

        # Excel Download (SCIE-like formatting)
        out_c = io.BytesIO()
        with pd.ExcelWriter(out_c, engine="xlsxwriter") as writer:
            sheet_name = "Cox"
            res_df.to_excel(writer, sheet_name=sheet_name, index=False)

            workbook = writer.book
            worksheet = writer.sheets[sheet_name]

            header_fmt = workbook.add_format(
                {
                    "bold": True,
                    "align": "center",
                    "valign": "vcenter",
                    "border": 1,
                }
            )
            body_fmt = workbook.add_format(
                {
                    "border": 1,
                }
            )

            for col_idx, col_name in enumerate(res_df.columns):
                worksheet.write(0, col_idx, col_name, header_fmt)

            for row_idx in range(1, len(res_df) + 1):
                worksheet.set_row(row_idx, None, body_fmt)

            worksheet.set_column(0, 0, 30)
            worksheet.set_column(1, len(res_df.columns) - 1, 22)

        st.download_button(
            "📥 Download Cox Results (SCIE style)",
            out_c.getvalue(),
            "Cox_Results_SCIE.xlsx",
        )

        if cm is not None and XA_train is not None:
            st.markdown("---")
            st.subheader("🔍 Proportional Hazards Assumption Check")
            with st.expander("View Schoenfeld Residuals Test"):
                ph_res = check_ph_assumption(cm, XA_train)
                if isinstance(ph_res, str):
                    st.error(f"PH Check Error: {ph_res}")
                else:
                    st.write(
                        "If p-value < 0.05, the proportional hazards assumption may be violated."
                    )
                    st.dataframe(ph_res.summary)
