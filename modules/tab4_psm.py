import streamlit as st
import pandas as pd
import numpy as np
import io
import xlsxwriter
import seaborn as sns
import matplotlib.pyplot as plt

from .utils import (
    calculate_smd,
    run_psm,
    suggest_variable_type_single,
    analyze_table1_robust,
)


def render_tab4(df: pd.DataFrame) -> None:
    """
    Render the Propensity Score Matching (PSM) tab.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe uploaded by the user.
    """
    st.subheader("Propensity Score Matching (PSM)")
    st.info(
        '''
        💡 **PSM** is used to reduce selection bias in observational studies
        by matching treated and control patients with similar propensity scores.

        - **Treatment Variable**: variable indicating treatment assignment (case vs control)
        - **Covariates**: variables used to estimate the propensity score (matching variables)
        '''
    )

    # ------------------------------------------------------------------
    # 1. Basic PSM configuration
    # ------------------------------------------------------------------
    c1, c2 = st.columns(2)

    # Treatment column (use original keys so reset_session_state() works)
    tc = c1.selectbox(
        "Treatment Variable (0/1 or Yes/No)",
        options=df.columns,
        key="p_t",
        help="Variable that distinguishes treatment and control groups.",
    )

    # Value that will be coded as 1 (treated)
    t1 = None
    if tc:
        vals = df[tc].dropna().unique()
        if len(vals) == 2:
            t1 = c2.selectbox(
                "Treated value (coded as 1)",
                options=vals,
                key="p_v",
                help="Select the value that represents the treated group.",
            )
        else:
            st.warning(
                "⚠️ Treatment variable must have exactly 2 unique values "
                "(e.g., 0/1, Yes/No)."
            )

    # Covariates used to estimate propensity score (matching variables)
    covs = st.multiselect(
        "Covariates for propensity score (matching variables)",
        options=[c for c in df.columns if c != tc],
        key="p_c",
        help="Variables used in the logistic model to estimate propensity score.",
    )

    # Caliper width (in SD of logit PS)
    cal = st.slider(
        "Caliper width (SD of logit PS)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
        help="Maximum allowed distance in logit propensity score for matching. "
        "Smaller values → stricter matching → matched N decreases.",
    )

    # ------------------------------------------------------------------
    # 2. Run PSM
    # ------------------------------------------------------------------
    if st.button("Run PSM", key="psm_run_button"):
        if tc is None or t1 is None:
            st.error("Please select a valid treatment variable and treated value.")
        elif not covs:
            st.error("Please select at least one covariate for matching.")
        else:
            with st.spinner("Running PSM..."):
                dp = df.copy()

                # Create binary treatment indicator used inside run_psm
                dp["__T"] = np.where(dp[tc] == t1, 1, 0)

                # Keep original before matching for SMD "Before"
                org_df = dp.copy()

                matched_df, org_out = run_psm(dp, "__T", covs, cal)

                # run_psm may return a reduced data frame for "before" (only
                # treatment + covariates + PS). Use it if available.
                if org_out is not None:
                    org_df = org_out

                if matched_df is None or len(matched_df) == 0:
                    st.error(
                        "Matching failed: no valid matches found. "
                        "Try relaxing the caliper (larger value) or reducing covariates."
                    )
                    # Clear previous results to avoid confusion
                    for k in [
                        "psm_done",
                        "psm_matched_df",
                        "psm_org_df",
                        "psm_covs",
                        "psm_treat_col",
                        "psm_var_config",
                        "psm_t1_selected_vals",
                    ]:
                        st.session_state.pop(k, None)
                else:
                    # Save to session_state for later use (SMD, Table 1, download)
                    st.session_state["psm_done"] = True
                    st.session_state["psm_matched_df"] = matched_df
                    st.session_state["psm_org_df"] = org_df
                    st.session_state["psm_covs"] = covs
                    st.session_state["psm_treat_col"] = tc
                    # Table 1 config will be (re)initialised below as needed

    # ------------------------------------------------------------------
    # 3. Show results if matching has been performed
    # ------------------------------------------------------------------
    required_keys = [
        "psm_done",
        "psm_matched_df",
        "psm_org_df",
        "psm_covs",
        "psm_treat_col",
    ]
    _MISSING = object()
    has_all_keys = all(
        st.session_state.get(k, _MISSING) is not _MISSING for k in required_keys
    )

    if has_all_keys and st.session_state.get("psm_done"):
        m_df: pd.DataFrame = st.session_state["psm_matched_df"]
        org: pd.DataFrame = st.session_state["psm_org_df"]
        covs = st.session_state["psm_covs"]
        tc = st.session_state["psm_treat_col"]

        # --------------------------------------------------------------
        # 3-1. Basic info about matched sample size
        # --------------------------------------------------------------
        st.success(f"✅ Matching complete! Matched N = {len(m_df)}")

        with st.expander("Show sample size before / after matching", expanded=False):
            if "__T" in org.columns:
                org_counts = org["__T"].value_counts().to_dict()
            else:
                org_counts = {}
            if "__T" in m_df.columns:
                m_counts = m_df["__T"].value_counts().to_dict()
            else:
                m_counts = {}

            treated_before = org_counts.get(1, 0)
            control_before = org_counts.get(0, 0)
            treated_after = m_counts.get(1, 0)
            control_after = m_counts.get(0, 0)

            st.write(
                f"- Before: Treated = **{treated_before}**, "
                f"Control = **{control_before}**"
            )
            st.write(
                f"- After: Treated = **{treated_after}**, "
                f"Control = **{control_after}**"
            )
            if treated_before > 0:
                rate = treated_after / treated_before
                st.write(f"- Matched proportion of treated = **{rate:.2%}**")

            st.caption(
                "⚠️ If matched N is much smaller than original, "
                "consider using fewer covariates or a larger caliper."
            )

        # --------------------------------------------------------------
        # 3-2. Balance check (SMD before / after)
        # --------------------------------------------------------------
        if covs:
            st.markdown("### ⚖️ Balance Check (SMD)")
            st.caption(
                "Standardized Mean Difference (SMD) < 0.1 usually indicates good balance."
            )

            sb = calculate_smd(org, "__T", covs)
            sa = calculate_smd(m_df, "__T", covs)
            sm = pd.merge(sb, sa, on="Variable", suffixes=("_Before", "_After"))

            st.dataframe(
                sm.style.format({"SMD_Before": "{:.3f}", "SMD_After": "{:.3f}"})
            )

            # SMD scatter plot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                data=sm,
                x="SMD_Before",
                y="Variable",
                color="red",
                label="Before matching",
                s=100,
                ax=ax,
            )
            sns.scatterplot(
                data=sm,
                x="SMD_After",
                y="Variable",
                color="blue",
                label="After matching",
                s=100,
                ax=ax,
            )
            ax.axvline(0.1, ls="--", color="gray", alpha=0.5)
            ax.axvline(-0.1, ls="--", color="gray", alpha=0.5)
            ax.set_title("SMD before vs after matching")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        # --------------------------------------------------------------
        # 3-3. Download matched dataset
        # --------------------------------------------------------------
        out_p = io.BytesIO()
        with pd.ExcelWriter(out_p, engine="xlsxwriter") as w:
            m_df.drop(columns=["__T", "logit_ps"], errors="ignore").to_excel(
                w, index=False
            )
        st.download_button(
            "📥 Download Matched Data",
            data=out_p.getvalue(),
            file_name="Matched_Data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # --------------------------------------------------------------
        # 3-4. Matched cohort Table 1 (compare variables after matching)
        # --------------------------------------------------------------
        st.markdown("---")
        st.subheader("📊 Matched Cohort Table 1")

        # ---- 3-4-1. Group settings (like Table 1 tab) ----
        st.markdown("#### Group Settings (Matched Data)")

        uvals = m_df[tc].dropna().unique()
        if len(uvals) < 2:
            st.warning(
                "Group variable in matched data must have at least two unique values."
            )
            return

        c_g1, c_g2 = st.columns(2)
        with c_g1:
            default_vals = list(uvals)
            selected_vals = st.multiselect(
                "Groups to include in matched Table 1",
                options=list(uvals),
                default=default_vals,
                key="psm_t1_selected_vals",
            )

        with c_g2:
            st.caption("Rename groups for matched Table 1 (optional).")
            gnames = {}
            for v in selected_vals:
                gnames[v] = st.text_input(
                    f"Display name for {v}",
                    value=str(v),
                    key=f"psm_t1_gname_{v}",
                )

        valid_vals = [v for v in selected_vals if v in uvals]
        if len(valid_vals) < 2:
            st.warning("Please select at least two valid groups for matched Table 1.")
            return

        value_map = {v: gnames.get(v, str(v)) for v in valid_vals}

        # ---- 3-4-2. Variable selection for matched Table 1 ----
        # All available columns in matched data except internal ones and treatment
        avail_cols = [
            c
            for c in m_df.columns
            if c not in ["__T", "logit_ps", "propensity_score", tc]
        ]

        # Initialise variable configuration when columns or covariates change
        _cfg = st.session_state.get("psm_var_config", None)
        need_init = _cfg is None
        if not need_init:
            prev_vars = list(_cfg["Variable"])
            if set(prev_vars) != set(avail_cols):
                need_init = True

        if need_init:
            init_rows = []
            for c in avail_cols:
                # 핵심:
                #   - 매칭에 사용한 covariate(covs)는 기본적으로 Include=False
                #   - 그 외 "나머지 변수들"은 기본적으로 Include=True
                include_flag = c not in covs  # 남은 변수들 기본 선택
                init_rows.append(
                    {
                        "Include": include_flag,
                        "Variable": c,
                        "Type": suggest_variable_type_single(m_df, c),
                    }
                )
            st.session_state["psm_var_config"] = pd.DataFrame(init_rows)

        # Select All / Deselect All buttons
        c_b1, c_b2, _ = st.columns([0.2, 0.2, 0.6])
        if c_b1.button("✅ Select All (Matched)", key="psm_all"):
            st.session_state["psm_var_config"]["Include"] = True
            st.rerun()
        if c_b2.button("⬜ Deselect All (Matched)", key="psm_none"):
            st.session_state["psm_var_config"]["Include"] = False
            st.rerun()

        # Variable editor (user can freely include/exclude ANY variable)
        psm_cfg = st.data_editor(
            st.session_state["psm_var_config"],
            column_config={
                "Include": st.column_config.CheckboxColumn(width="small"),
                "Variable": st.column_config.TextColumn(disabled=True),
                "Type": st.column_config.SelectboxColumn(
                    options=["Continuous", "Categorical"]
                ),
            },
            hide_index=True,
            num_rows="fixed",
            key="psm_editor",
        )
        st.session_state["psm_var_config"] = psm_cfg

        # Parse current selection
        sel = psm_cfg[psm_cfg["Include"] == True]
        t_vars = sel["Variable"].tolist()
        u_cont = sel[sel["Type"] == "Continuous"]["Variable"].tolist()
        u_cat = sel[sel["Type"] == "Categorical"]["Variable"].tolist()

        if st.button("Generate Matched Table 1", key="btn_mt1"):
            if not t_vars:
                st.warning("Please select at least one variable for Table 1.")
            else:
                mt1, err = analyze_table1_robust(
                    m_df, tc, value_map, t_vars, u_cont, u_cat
                )
                if err:
                    st.error(f"Error while generating Table 1: {err}")
                else:
                    st.dataframe(mt1)
                    out_m1 = io.BytesIO()
                    with pd.ExcelWriter(out_m1, engine="xlsxwriter") as w:
                        mt1.to_excel(w, index=False)
                    st.download_button(
                        "📥 Download Matched Table 1",
                        data=out_m1.getvalue(),
                        file_name="Matched_Table1.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

    # When psm_done flag is True but some keys are missing (e.g. after code update)
    elif st.session_state.get("psm_done") and not has_all_keys:
        st.warning(
            "PSM state is incomplete. Please click **Run PSM** again to refresh the results."
        )
