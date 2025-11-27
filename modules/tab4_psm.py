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
        💡 **PSM**는 관찰연구에서 선택 편향을 줄이기 위해,
        치료군과 대조군의 특성이 비슷하도록 짝을 맞추는 방법입니다.

        - **Treatment Variable**: 치료군/대조군을 구분하는 변수
        - **Covariates**: Propensity score 계산에 사용하는 공변량(매칭 변수)
        """
    )

    # ------------------------------------------------------------------
    # 1. Basic PSM configuration
    # ------------------------------------------------------------------
    c1, c2 = st.columns(2)

    # Treatment column (group variable)
    tc = c1.selectbox(
        "Treatment Variable (0/1 or Yes/No)",
        options=df.columns,
        key="p_t",
        help="치료군과 대조군을 구분하는 변수를 선택하세요.",
    )

    # Which value is treated (coded as 1)
    t1 = None
    if tc:
        vals = df[tc].dropna().unique()
        if len(vals) == 2:
            t1 = c2.selectbox(
                "Treated value (Case = 1)",
                options=vals,
                key="p_v",
                help="선택한 값이 치료군(1)으로 코딩됩니다.",
            )
        else:
            st.warning(
                "⚠️ Treatment variable 은 정확히 2개의 값(예: 0/1, Yes/No)만 가져야 합니다."
            )

    # Covariates used *only* for propensity score estimation
    covs = st.multiselect(
        "Covariates for propensity score (매칭에 사용할 공변량)",
        options=[c for c in df.columns if c != tc],
        key="p_c",
        help="이 변수들로 propensity score를 계산하고 매칭을 수행합니다.",
    )

    # Caliper width in SD of logit(PS)
    cal = st.slider(
        "Caliper width (SD of logit PS)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
        help="logit(propensity score)의 표준편차 × 값 만큼 거리 이내에서만 매칭합니다.",
    )

    # ------------------------------------------------------------------
    # 2. Run PSM
    # ------------------------------------------------------------------
    if st.button("Run PSM", key="psm_run"):
        if tc is None or t1 is None:
            st.error("Treatment variable과 treated value를 먼저 올바르게 선택하세요.")
        elif not covs:
            st.error("최소 1개 이상의 covariate를 선택해야 합니다.")
        else:
            with st.spinner("PSM 수행 중..."):
                dp = df.copy()

                # 내부용 이진 treatment 변수 (__T: 1=treated, 0=control)
                dp["__T"] = np.where(dp[tc] == t1, 1, 0)

                # run_psm는 내부적으로 결측(dropna)과 caliper를 적용합니다.
                m_df, org = run_psm(dp, "__T", covs, cal)

                if m_df is None:
                    st.error(
                        "Matching에 실패했습니다. caliper를 넓히거나, covariate 개수를 줄이거나, "
                        "결측값을 정리한 뒤 다시 시도해 보세요."
                    )
                else:
                    # 새로 매칭할 때마다 이전 Table 1 설정은 초기화
                    if "psm_var_config" in st.session_state:
                        del st.session_state["psm_var_config"]

                    st.session_state["psm_done"] = True
                    st.session_state["psm_matched_df"] = m_df
                    st.session_state["psm_org_df"] = org
                    # covs, tc는 위젯 key(p_t, p_c)에서 항상 복원되므로 별도 저장 불필요

    # ------------------------------------------------------------------
    # 3. Show results if matching has been performed
    # ------------------------------------------------------------------
    required_keys = ["psm_done", "psm_matched_df", "psm_org_df"]
    if not all(k in st.session_state for k in required_keys):
        return
    if not st.session_state.get("psm_done"):
        return

    m_df: pd.DataFrame = st.session_state["psm_matched_df"]
    org: pd.DataFrame = st.session_state["psm_org_df"]

    # 최신 위젯 상태(=실제 사용자가 선택해 둔 값)를 다시 가져옴
    tc = st.session_state.get("p_t")
    covs = st.session_state.get("p_c", [])

    if tc is None or tc not in df.columns:
        st.error("현재 선택된 treatment variable이 유효하지 않습니다. 다시 선택 후 매칭을 실행하세요.")
        return

    st.success(f"✅ Matching complete! (Matched N = {len(m_df)})")

    # ------------------------------------------------------------------
    # 3-1. Balance check: SMD before / after (for covariates only)
    # ------------------------------------------------------------------
    if covs:
        st.markdown("### ⚖️ Balance Check (SMD)")
        st.caption("일반적으로 |SMD| < 0.1 이면 공변량 균형이 잘 맞았다고 봅니다.")

        # org, m_df 모두 __T(0/1)를 사용해서 SMD 계산
        sb = calculate_smd(org, "__T", covs)
        sa = calculate_smd(m_df, "__T", covs)
        sm = pd.merge(sb, sa, on="Variable", suffixes=("_Before", "_After"))

        st.dataframe(
            sm.style.format({"SMD_Before": "{:.3f}", "SMD_After": "{:.3f}"})
        )

        # SMD plot
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
        ax.set_title("Standardized Mean Difference (SMD)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # ------------------------------------------------------------------
    # 3-2. Download matched dataset
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3-3. Matched Cohort Table 1 (사용자가 비교하고 싶은 변수 선택)
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Matched Cohort Table 1")

    # Matched data에서 사용할 수 있는 변수들:
    #  - 내부용 __T, logit_ps, propensity_score, treatment variable 은 제외
    avail_cols = [
        c
        for c in m_df.columns
        if c not in ["__T", "logit_ps", "propensity_score", tc]
    ]

    # 세션에 저장된 설정이 없거나, 컬럼 구성이 달라졌으면 초기화
    need_init = True
    if "psm_var_config" in st.session_state:
        prev = st.session_state["psm_var_config"]
        if set(prev["Variable"].tolist()) == set(avail_cols):
            need_init = False

    if need_init:
        init_rows = []
        for c in avail_cols:
            # 핵심: 매칭에 사용한 covariate 는 기본적으로 Include=False,
            # 그 외 "나머지" 변수들은 기본적으로 Include=True
            include_flag = c not in covs
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

    # Variable editor
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
        use_container_width=True,
        num_rows="fixed",
        key="psm_editor",
    )
    st.session_state["psm_var_config"] = psm_cfg

    # 현재 선택 상태 파싱
    sel = psm_cfg[psm_cfg["Include"] == True]
    t_vars = sel["Variable"].tolist()
    u_cont = sel[sel["Type"] == "Continuous"]["Variable"].tolist()
    u_cat = sel[sel["Type"] == "Categorical"]["Variable"].tolist()

    # value_map: treatment variable 의 실제 값 → 문자열
    mt_vals = m_df[tc].unique()
    val_map = {v: str(v) for v in mt_vals}

    if st.button("Generate Matched Table 1", key="btn_mt1"):
        if not t_vars:
            st.warning("Table 1에 포함할 변수를 한 개 이상 선택해야 합니다.")
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
