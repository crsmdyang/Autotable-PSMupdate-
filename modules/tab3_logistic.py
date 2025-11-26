import io
import streamlit as st
import pandas as pd
import numpy as np
import xlsxwriter
import statsmodels.api as sm

from .missing import apply_missing_policy
from .utils import (
    ensure_binary_event,
    ordered_levels,
    make_dummies,
    check_vif,
    plot_forest,
    plot_roc_curve,
    dummy_colname,
    format_p,
)


def render_tab3(df: pd.DataFrame):
    """
    Binary Logistic Regression
    - 단변량 + 다변량 OR(95% CI), p를 SCIE 스타일 테이블로 생성
    - 선택한 변수는 모두 다변량에 포함
    - 결과를 엑셀로 다운로드 가능
    """
    st.subheader("💊 Binary Logistic Regression")

    alpha_global = float(st.session_state.get("alpha", 0.05))
    decimals = int(st.session_state.get("decimals", 3))
    missing_policy = st.session_state.get(
        "missing_policy", "Variable-wise drop (per analysis)"
    )

    # 1. Dependent Variable
    dy = st.selectbox(
        "Y (Dependent Variable)",
        df.columns,
        key="l_y",
        help="예측하고자 하는 결과 변수입니다. (이분형, 예: Disease Yes/No)",
    )

    if not dy:
        st.info("Please select a dependent variable (Y).")
        return

    # 2. Event / Control Value
    uniq_y = sorted(df[dy].dropna().unique())
    evs = st.multiselect(
        "Event Value (1)",
        uniq_y,
        key="l_ev",
        help="Y에서 1(사건)으로 간주할 값을 선택하세요. (예: Yes, Dead 등)",
    )
    cts = st.multiselect(
        "Control Value (0)",
        uniq_y,
        key="l_ct",
        help="Y에서 0(대조)으로 간주할 값을 선택하세요. (예: No, Alive 등)",
    )

    if not evs or not cts:
        st.warning("Please select both Event and Control values.")
        return

    if set(evs) & set(cts):
        st.error("Event values and Control values must not overlap.")
        return

    dfl = df.copy()
    dfl["Y"] = ensure_binary_event(dfl[dy], set(evs), set(cts))
    dfl = dfl.dropna(subset=["Y"])
    n_events = int(dfl["Y"].sum())
    st.info(f"Analysis data: N={len(dfl)} (Events={n_events})")

    if len(dfl) < 10 or n_events < 5:
        st.error("Insufficient data or events for logistic regression.")
        return

    # 3. Independent Variables
    ivs = st.multiselect(
        "X (Independent Variables)",
        [c for c in df.columns if c != dy],
        key="l_x",
        help="결과에 영향을 줄 수 있다고 생각되는 설명 변수를 선택하세요.",
    )

    if not ivs:
        st.info("Please select at least one independent variable.")
        return

    # Univariate p-threshold (정보용)
    default_pe = min(max(alpha_global, 0.001), 1.0)
    pe = st.number_input(
        "Univariate p-value threshold (for highlight only)",
        min_value=0.001,
        max_value=1.0,
        value=float(default_pe),
        step=0.01,
        key="l_pe",
        help="단변량에서 이 값보다 작은 변수들을 정보용으로 표시합니다. (변수 포함 여부에는 영향 X)",
    )

    show_vif = st.checkbox(
        "Show multicollinearity check (VIF, multivariate model)",
        value=False,
        key="l_show_vif",
    )

    if not st.button("Run Logistic Regression", key="logistic_run"):
        return

    # 4. 결측값 처리
    model_df = dfl.copy()
    cols_for_policy = list(ivs)
    if cols_for_policy:
        sub = apply_missing_policy(model_df, cols_for_policy, missing_policy)
        model_df = model_df.loc[sub.index].copy()
        for c in cols_for_policy:
            model_df[c] = sub[c]
    model_df = model_df.dropna(subset=["Y"])

    if model_df["Y"].sum() < 5:
        st.error("Too few events after handling missing data.")
        return

    # 5. 단변량 / 다변량 분석 준비
    cat_info = {}  # var -> {"levels": [...], "ref": ...} or None
    uni_results = {}  # var -> {coef_name -> stats dict}
    uni_p_min = {}    # var -> min p

    st.markdown("#### 🔍 Univariate analysis (정보용)")

    with st.spinner("Running univariate logistic regression..."):
        for v in ivs:
            try:
                t = model_df[["Y", v]].dropna()
                if t.empty:
                    continue

                if t[v].dtype == "object" or t[v].nunique() < 10:
                    lv = ordered_levels(t[v])
                    if len(lv) < 2:
                        continue
                    cat_info[v] = {"levels": lv, "ref": lv[0]}
                    X_uni = make_dummies(t, v, lv)
                else:
                    cat_info[v] = {"levels": None, "ref": None}
                    X_uni = pd.to_numeric(t[v], errors="coerce").to_frame(v)

                X_uni = sm.add_constant(X_uni, has_constant="add")
                m_uni = sm.Logit(t["Y"], X_uni).fit(disp=0)

                params = m_uni.params
                conf = m_uni.conf_int()
                pvals = m_uni.pvalues

                uni_results[v] = {}
                p_list = []
                for c in params.index:
                    if c == "const":
                        continue
                    or_ = float(np.exp(params[c]))
                    lo = float(np.exp(conf.loc[c][0]))
                    hi = float(np.exp(conf.loc[c][1]))
                    pv = float(pvals[c])
                    uni_results[v][c] = {
                        "OR": or_,
                        "Lower": lo,
                        "Upper": hi,
                        "p": pv,
                    }
                    p_list.append(pv)

                if p_list:
                    uni_p_min[v] = float(min(p_list))

            except Exception:
                continue

    if uni_p_min:
        sig_vars = [v for v, pv in uni_p_min.items() if pv < pe]
        st.info(f"Univariate p < {pe}: {sig_vars}")
    else:
        st.warning("Univariate analysis could not be computed for selected variables.")

    # 6. 다변량 로지스틱 (모든 선택 변수 포함)
    with st.spinner("Running multivariate logistic regression..."):
        XL = []
        for v in ivs:
            info = cat_info.get(v)
            if info is None or info["levels"] is None:
                XL.append(pd.to_numeric(model_df[v], errors="coerce").to_frame(v))
            else:
                XL.append(make_dummies(model_df[[v]], v, info["levels"]))

        XM = sm.add_constant(pd.concat(XL, axis=1), has_constant="add")
        DM = pd.concat([model_df["Y"], XM], axis=1).dropna()

        if DM["Y"].sum() < 5:
            st.error(
                "Too few events in the multivariate dataset after missing-data handling."
            )
            return

        try:
            f_model = sm.Logit(DM["Y"], DM.drop(columns=["Y"])).fit(disp=0)
        except Exception as e:
            st.error(f"Multivariate Analysis Error: {e}")
            return

    # 7. 다변량 결과 정리
    multi_results = {}
    conf_m = f_model.conf_int()
    for c in f_model.params.index:
        if c == "const":
            continue
        or_ = float(np.exp(f_model.params[c]))
        lo = float(np.exp(conf_m.loc[c][0]))
        hi = float(np.exp(conf_m.loc[c][1]))
        pv = float(f_model.pvalues[c])
        multi_results[c] = {
            "OR": or_,
            "Lower": lo,
            "Upper": hi,
            "p": pv,
        }

    # Forest plot용 DF
    forest_rows = []
    for coef_name, vals in multi_results.items():
        forest_rows.append(
            {
                "Variable": coef_name,
                "OR": vals["OR"],
                "Lower": vals["Lower"],
                "Upper": vals["Upper"],
                "p-value": vals["p"],
            }
        )
    forest_df = pd.DataFrame(forest_rows)

    # 8. SCIE 스타일 결과 테이블 구성 (단변량 + 다변량)
    st.markdown("---")
    st.subheader("📑 Logistic Regression Results (SCIE-style table)")

    rows = []
    for v in ivs:
        info = cat_info.get(v, {"levels": None, "ref": None})
        levels = info["levels"]

        # 연속형 변수
        if levels is None:
            uni_coef = uni_results.get(v, {}).get(v)
            m_coef = multi_results.get(v)

            if uni_coef:
                u_or = uni_coef["OR"]
                u_lo = uni_coef["Lower"]
                u_hi = uni_coef["Upper"]
                u_p = uni_coef["p"]
                uni_str = f"{u_or:.2f} ({u_lo:.2f}-{u_hi:.2f})"
                uni_p_str = format_p(u_p)
            else:
                uni_str = "NA"
                uni_p_str = "NA"

            if m_coef:
                m_or = m_coef["OR"]
                m_lo = m_coef["Lower"]
                m_hi = m_coef["Upper"]
                m_p = m_coef["p"]
                multi_str = f"{m_or:.2f} ({m_lo:.2f}-{m_hi:.2f})"
                multi_p_str = format_p(m_p)
            else:
                multi_str = "NA"
                multi_p_str = "NA"

            rows.append(
                {
                    "Factor": v,
                    "Subgroup": "",
                    "Uni OR (95% CI)": uni_str,
                    "p (uni)": uni_p_str,
                    "Multi OR (95% CI)": multi_str,
                    "p (multi)": multi_p_str,
                }
            )

        # 범주형 변수
        else:
            lv = levels
            ref = lv[0]
            rows.append(
                {
                    "Factor": v,
                    "Subgroup": f"{ref} (Ref)",
                    "Uni OR (95% CI)": "Ref",
                    "p (uni)": "",
                    "Multi OR (95% CI)": "Ref",
                    "p (multi)": "",
                }
            )

            for lev in lv[1:]:
                cn = dummy_colname(v, lev)

                uni_coef = uni_results.get(v, {}).get(cn)
                m_coef = multi_results.get(cn)

                if uni_coef:
                    u_or = uni_coef["OR"]
                    u_lo = uni_coef["Lower"]
                    u_hi = uni_coef["Upper"]
                    u_p = uni_coef["p"]
                    uni_str = f"{u_or:.2f} ({u_lo:.2f}-{u_hi:.2f})"
                    uni_p_str = format_p(u_p)
                else:
                    uni_str = "NA"
                    uni_p_str = "NA"

                if m_coef:
                    m_or = m_coef["OR"]
                    m_lo = m_coef["Lower"]
                    m_hi = m_coef["Upper"]
                    m_p = m_coef["p"]
                    multi_str = f"{m_or:.2f} ({m_lo:.2f}-{m_hi:.2f})"
                    multi_p_str = format_p(m_p)
                else:
                    multi_str = "NA"
                    multi_p_str = "NA"

                rows.append(
                    {
                        "Factor": "",
                        "Subgroup": str(lev),
                        "Uni OR (95% CI)": uni_str,
                        "p (uni)": uni_p_str,
                        "Multi OR (95% CI)": multi_str,
                        "p (multi)": multi_p_str,
                    }
                )

    res_table = pd.DataFrame(rows)
    st.dataframe(res_table, use_container_width=True)

    # 9. Forest Plot (다변량 OR 기준)
    if not forest_df.empty:
        st.markdown("---")
        st.subheader("🌲 Forest Plot (multivariate OR)")
        fig_forest = plot_forest(forest_df)
        if fig_forest:
            st.pyplot(fig_forest)

    # 10. ROC Curve
    st.markdown("---")
    st.subheader("📈 ROC Curve & AUC")
    y_true = DM["Y"]
    y_prob = f_model.predict(DM.drop(columns=["Y"]))
    fig_roc, roc_auc = plot_roc_curve(y_true, y_prob)
    st.pyplot(fig_roc)
    st.info(f"AUC (Area Under Curve): **{roc_auc:.3f}**")

    # 11. VIF (optional)
    if show_vif:
        st.markdown("---")
        st.subheader("📎 Multicollinearity Check (VIF)")
        vif_df = check_vif(DM.drop(columns=["Y"]))
        st.dataframe(vif_df)
        st.caption("일반적으로 VIF > 10이면 다중공선성이 높은 것으로 간주합니다.")

    # 12. Excel (SCIE-style)
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        sheet_name = "Logistic"
        res_table.to_excel(writer, sheet_name=sheet_name, index=False)

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
        factor_fmt = workbook.add_format(
            {
                "border": 1,
                "bold": True,
            }
        )

        # 헤더
        for col_idx, col_name in enumerate(res_table.columns):
            worksheet.write(0, col_idx, col_name, header_fmt)

        # 본문 (Factor 있는 행은 bold)
        for row_idx in range(1, len(res_table) + 1):
            factor_val = str(res_table.iloc[row_idx - 1, 0])
            fmt = factor_fmt if factor_val else body_fmt
            worksheet.set_row(row_idx, None, fmt)

        worksheet.set_column(0, 0, 25)
        worksheet.set_column(1, 1, 20)
        worksheet.set_column(2, len(res_table.columns) - 1, 28)

    st.download_button(
        "📥 Download Logistic Results (SCIE-style)",
        out.getvalue(),
        "Logistic_Results_SCIE.xlsx",
    )
