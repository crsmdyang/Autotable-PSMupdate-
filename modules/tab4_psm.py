import io
import streamlit as st
import pandas as pd
import xlsxwriter

from .missing import apply_missing_policy
from .utils import (
    run_psm,
    calculate_smd,
    analyze_table1_robust,
    suggest_variable_type_single,
)


def render_tab4(df: pd.DataFrame):
    """
    Propensity Score Matching (PSM) tab.
    - allow replacement 옵션 제거 (항상 without replacement)
    - 매칭 후 Table 1 변수 선택 방식도 Table1 탭과 동일 (include + continuous 리스트)
    """
    st.subheader("🎯 Propensity Score Matching (PSM)")

    # 1. Treatment & Covariate selection
    c1, c2 = st.columns(2)
    treat_col = c1.selectbox(
        "Treatment Column",
        df.columns,
        key="psm_treat_col",
        help="치료/대조군을 구분하는 변수입니다. (예: Treatment, Chemo 등)",
    )

    if not treat_col:
        st.info("Please select a treatment column.")
        return

    uniq_treat = sorted(df[treat_col].dropna().unique())
    treated_val = c2.selectbox(
        "Treated value (coded as 1)",
        uniq_treat,
        key="psm_treated_val",
        help="이 값으로 되어 있는 환자를 'Treated(1)'로 간주합니다. 나머지는 Control(0).",
    )

    covariates = st.multiselect(
        "Covariates for propensity score",
        [c for c in df.columns if c != treat_col],
        key="psm_covs",
        help="PSM에서 균형을 맞추고 싶은 공변량을 선택하세요.",
    )

    st.markdown("#### ⚙️ PSM Settings")
    c3, c4 = st.columns(2)
    use_caliper = c3.checkbox(
        "Use caliper",
        value=True,
        key="psm_use_caliper",
        help="체크 해제 시 caliper 없이 가장 가까운 이웃을 매칭합니다.",
    )
    caliper = c4.slider(
        "Caliper (SD of logit PS)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
        key="psm_caliper",
        help="logit(propensity score)의 표준편차 배수로 caliper를 설정합니다.",
    )
    ratio = int(
        st.selectbox(
            "Matching ratio (Treated:Control)",
            [1, 2, 3, 4],
            index=0,
            key="psm_ratio",
            help="각 Treat 환자당 매칭할 Control 환자 수입니다.",
        )
    )

    if st.button("Run PSM", key="psm_run_btn"):
        if not covariates:
            st.error("Please select at least one covariate.")
            st.session_state["psm_run_done"] = False
            return

        policy = st.session_state.get(
            "missing_policy", "Variable-wise drop (per analysis)"
        )
        cols_for_psm = [treat_col] + covariates
        df_use = apply_missing_policy(df, cols_for_psm, policy).copy()
        if df_use.empty:
            st.error("No data left after applying missing-data policy.")
            st.session_state["psm_run_done"] = False
            return

        # Binary treatment indicator
        df_use["__T"] = df_use[treat_col].astype(str).eq(str(treated_val)).astype(int)

        cal = caliper if use_caliper else 0.0

        with st.spinner("Running propensity score model and matching..."):
            # ✅ 항상 without replacement
            matched_df, org_df = run_psm(
                df_use,
                treatment_col="__T",
                covariates=covariates,
                caliper=cal,
                ratio=ratio,
                replace=False,
            )

        if matched_df is None or org_df is None:
            st.error(
                "No matched pairs were found. "
                "Try increasing caliper, reducing covariates, or checking treatment coding."
            )
            st.session_state["psm_run_done"] = False
            return

        st.success(f"PSM completed: matched N = {len(matched_df)}")

        st.session_state["psm_run_done"] = True
        st.session_state["psm_matched_df"] = matched_df
        st.session_state["psm_org_df"] = org_df
        st.session_state["psm_treat_binary"] = "__T"
        st.session_state["psm_treat_label_treated"] = str(treated_val)
        st.session_state["psm_treat_label_control"] = "Others"

        # Balance check (SMD before/after)
        smd_before = calculate_smd(org_df, "__T", covariates)
        smd_after = calculate_smd(matched_df, "__T", covariates)
        smd = smd_before.merge(
            smd_after,
            on="Variable",
            how="outer",
            suffixes=("_Before", "_After"),
        )

        st.markdown("#### 🔍 Covariate Balance (Standardized Mean Difference)")
        st.dataframe(smd, use_container_width=True)

        smd_out = io.BytesIO()
        with pd.ExcelWriter(smd_out, engine="xlsxwriter") as writer:
            smd.to_excel(writer, index=False, sheet_name="SMD")
        st.download_button(
            "📥 Download SMD Table",
            smd_out.getvalue(),
            "PSM_SMD.xlsx",
        )

    # --- Matched Table 1 (like Table1 tab) ---
    if not st.session_state.get("psm_run_done", False):
        return

    st.markdown("---")
    st.subheader("📊 Matched Cohort: Table 1")

    matched_df = st.session_state.get("psm_matched_df")
    if matched_df is None or matched_df.empty:
        st.warning("Matched data not available.")
        return

    treat_binary = st.session_state.get("psm_treat_binary", "__T")
    treated_label = st.session_state.get("psm_treat_label_treated", "Treated")
    control_label = st.session_state.get("psm_treat_label_control", "Control")

    value_map = {"1": treated_label, "0": control_label}

    st.info(
        f"Matched group sizes: "
        f"{treated_label} = {(matched_df[treat_binary] == 1).sum()}, "
        f"{control_label} = {(matched_df[treat_binary] == 0).sum()}"
    )

    st.markdown("#### ⚙️ Variable Configuration for Matched Table 1")

    # covariate 뿐만 아니라 matched_df 안의 다른 변수도 전부 선택 가능
    all_vars = [
        c
        for c in matched_df.columns
        if c not in [treat_binary, "propensity_score", "logit_ps"]
    ]

    # (1) 포함 변수
    include_default = st.session_state.get("psm_include_vars", all_vars)
    include_default = [v for v in include_default if v in all_vars]

    include_vars = st.multiselect(
        "Variables to include in matched Table 1",
        all_vars,
        default=include_default if include_default else all_vars,
        key="psm_include_vars",
        help="Matched Table 1에 포함할 변수를 선택하세요.",
    )

    if not include_vars:
        st.info("Please select at least one variable for matched Table 1.")
        return

    # (2) 연속형 변수
    auto_cont = [
        v for v in include_vars
        if suggest_variable_type_single(matched_df, v) == "Continuous"
    ]
    prev_cont = st.session_state.get("psm_cont_vars", auto_cont)
    prev_cont = [v for v in prev_cont if v in include_vars]
    if not prev_cont:
        prev_cont = auto_cont

    cont_vars = st.multiselect(
        "Continuous variables (나머지는 Categorical로 처리)",
        include_vars,
        default=prev_cont,
        key="psm_cont_vars",
        help="연속형(Mean±SD 또는 Median[IQR])으로 보고 싶은 변수만 선택하세요.",
    )

    cat_vars = [v for v in include_vars if v not in cont_vars]

    if st.button("Generate Matched Table 1", key="psm_t1_btn"):
        with st.spinner("Analyzing matched cohort..."):
            t1_res, err = analyze_table1_robust(
                matched_df,
                group_col=treat_binary,
                value_map=value_map,
                target_cols=include_vars,
                user_cont_vars=cont_vars,
                user_cat_vars=cat_vars,
            )

        if err:
            st.error(f"Matched Table 1 error on variable '{err['var']}': {err['msg']}")
            return

        st.success("Matched Table 1 generated!")
        st.dataframe(t1_res, use_container_width=True)

        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            sheet_name = "Matched_Table1"
            t1_res.to_excel(writer, index=False, sheet_name=sheet_name)
        st.download_button(
            "📥 Download Matched Table 1",
            out.getvalue(),
            "PSM_Matched_Table1.xlsx",
        )
