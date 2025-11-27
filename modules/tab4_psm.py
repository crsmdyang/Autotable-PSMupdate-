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
        """
        💡 **PSM** is used to reduce selection bias in observational studies
        by matching treated and control patients with similar propensity scores.

        - **Treatment Variable**: variable indicating treatment assignment (case vs control)
        - **Covariates**: variables used to estimate the propensity score (matching variables)
        """
    )

    # ------------------------------------------------------------------
    # 1. Basic PSM configuration
    # ------------------------------------------------------------------
    c1, c2 = st.columns(2)

    # Treatment column
    tc = c1.selectbox(
        "Treatment Variable (0/1 or Yes/No)",
        options=df.columns,
        key="psm_treatment_col",
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
                key="psm_treated_value",
                help="Select the value that represents the treated group.",
            )
        else:
            st.warning(
                "⚠️ Treatment variable must have exactly 2 unique values "
                "(e.g., 0/1, Yes/No)."
            )

    # Covariates used to estimate propensity score
    covs = st.multiselect(
        "Covariates for propensity score (matching variables)",
        options=[c for c in df.columns if c != tc],
        key="psm_covariates",
        help="Variables used in the logistic model to estimate propensity score.",
    )

    # Caliper width (in SD of logit PS)
    cal = st.slider(
        "Caliper width (SD of logit PS)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
        help="Maximum allowed distance in logit propensity score for matching.",
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

                matched_df, org_df = run_psm(dp, "__T", covs, cal)

                if matched_df is None:
                    st.error(
                        "Matching failed: no valid matches found. "
                        "Try relaxing the caliper or checking the data."
                    )
                else:
                    # Save to session_state for later use (SMD, Table 1, download)
                    st.session_state["psm_done"] = True
                    st.session_state["psm_matched_df"] = matched_df
                    st.session_state["psm_org_df"] = org_df
                    st.session_state["psm_covs"] = covs
                    st.session_state["psm_treat_col"] = tc

    # ------------------------------------------------------------------
    # 3. Show results if matching has been performed
    # ------------------------------------------------------------------
    if st.session_state.get("psm_done"):
        m_df: pd.DataFrame = st.session_state["psm_matched_df"]
        org: pd.DataFrame = st.session_state["psm_org_df"]
        covs = st.session_state["psm_covs"]
        tc = st.session_state["psm_treat_col"]

        st.success(f"✅ Matching complete! Matched N = {len(m_df)}")

        # --------------------------------------------------------------
        # 3-1. Balance check (SMD before / after)
        # --------------------------------------------------------------
        st.markdown("### ⚖️ Balance Check (SMD)")
        st.caption("Standardized Mean Difference (SMD) < 0.1 usually indicates good balance.")

        sb = calculate_smd(org, "__T", covs)
        sa = calculate_smd(m_df, "__T", covs)
        sm = pd.merge(sb, sa, on="Variable", suffixes=("_Before", "_After"))

        st.dataframe(sm.style.format({"SMD_Before": "{:.3f}", "SMD_After": "{:.3f}"}))

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

        # Download matched dataset (drop internal columns)
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
        # 3-2. Matched cohort Table 1 (compare variables after matching)
        # --------------------------------------------------------------
        st.markdown("---")
        st.subheader("📊 Matched Cohort Table 1")

        # All available columns in matched data except internal ones and treatment
        avail_cols = [
            c
            for c in m_df.columns
            if c not in ["__T", "logit_ps", "propensity_score", tc]
        ]

        # Initialise variable configuration when columns or covariates change
        need_init = "psm_var_config" not in st.session_state
        if not need_init:
            prev_vars = list(st.session_state["psm_var_config"]["Variable"])
            if set(prev_vars) != set(avail_cols):
                need_init = True

        if need_init:
            init_rows = []
            for c in avail_cols:
                # 핵심: 매칭에 사용한 covariate(covs)는 기본적으로 제외,
                # 그 외 "나머지 변수들"은 기본적으로 Include=True
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

        # Build value map using original treatment values
        mt_vals = m_df[tc].unique()
        val_map = {v: str(v) for v in mt_vals}

        if st.button("Generate Matched Table 1", key="btn_mt1"):
            if not t_vars:
                st.warning("Please select at least one variable for Table 1.")
            else:
                mt1, err = analyze_table1_robust(
                    m_df, tc, val_map, t_vars, u_cont, u_cat
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
