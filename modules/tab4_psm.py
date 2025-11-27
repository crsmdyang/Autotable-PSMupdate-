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

    - PS 기반 1:1 / 1:n 매칭 (nearest neighbor, caliper 사용)
    - SMD before / after 확인
    - 매칭 코호트에 대해 Tab1과 동일한 UX로 Table 1 생성
    """
    st.subheader("Propensity Score Matching (PSM)")
    st.info(
        "💡 **PSM**은 관찰연구에서 선택 편향을 줄이기 위해, "
        "치료군과 대조군의 propensity score(치료를 받을 확률)를 비슷하게 맞춰주는 방법입니다.\n\n"
        "- Treatment Variable: 치료군/대조군을 구분하는 이분형 변수\n"
        "- Covariates: PS 계산 및 매칭에 사용할 공변량\n"
        "- Caliper & Matching ratio(배율): 매칭의 엄격함과 대조군:치료군 비율을 조절합니다."
    )

    # ------------------------------------------------------------------
    # 1. PSM 설정
    # ------------------------------------------------------------------
    c1, c2 = st.columns(2)

    # 1-1. Treatment 변수 및 Treated value 선택
    treat_col = c1.selectbox(
        "Treatment Variable (이분형)",
        options=df.columns,
        key="psm_treat_col",
        help="치료군과 대조군을 구분하는 이분형 변수를 선택하세요 (값이 정확히 2개여야 합니다).",
    )

    treated_value = None
    can_use_treat = False
    if treat_col:
        vals = df[treat_col].dropna().unique()
        if len(vals) == 2:
            treated_value = c2.selectbox(
                "Treated value (Case로 간주할 값)",
                options=list(vals),
                key="psm_treat_val",
                help="이 값을 갖는 환자를 치료군(1)으로, 나머지를 대조군(0)으로 사용합니다.",
            )
            can_use_treat = True
        else:
            c2.warning("⚠️ Treatment 변수는 정확히 2개의 고유값만 가져야 합니다 (예: 0/1, Yes/No).")

    # 1-2. 매칭에 사용할 공변량 선택
    cov_options = [c for c in df.columns if c != treat_col]
    covariates = st.multiselect(
        "Covariates for matching (PS 계산 및 매칭에 사용할 공변량)",
        options=cov_options,
        key="psm_covariates",
        help="너무 많은 공변량을 동시에 선택하면 NA가 많을 경우 매칭 가능한 환자 수가 줄어듭니다.",
    )

    # 1-3. PSM 통계 설정 (caliper & 1:n ratio)
    with st.expander("⚙️ PSM 통계 설정", expanded=True):
        caliper = st.slider(
            "Caliper width (SD of logit PS)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.01,
            key="psm_caliper",
            help="logit(propensity score)의 표준편차를 1로 보았을 때의 caliper 폭입니다. "
                 "보통 0.2를 기본값으로 사용합니다.",
        )
        match_ratio = st.number_input(
            "Matching ratio (Controls per Treated, 1:n)",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
            key="psm_ratio",
            help="대조군:치료군 비율입니다. 1이면 1:1 매칭, 2이면 1:2 매칭을 시도합니다. "
                 "비율이 커질수록 대조군 수는 늘어나지만 매칭이 더 어려워질 수 있습니다.",
        )
        allow_replace = st.checkbox(
            "Allow matching with replacement (같은 대조군이 여러 치료군과 매칭될 수 있음)",
            value=False,
            key="psm_replace",
        )

    # ------------------------------------------------------------------
    # 2. PSM 실행
    # ------------------------------------------------------------------
    if st.button("Run PSM", key="psm_run_btn"):
        if not can_use_treat or treated_value is None:
            st.error("Treatment 변수와 Treated value를 올바르게 선택해 주세요.")
        elif not covariates:
            st.error("매칭에 사용할 Covariates를 최소 1개 이상 선택해 주세요.")
        else:
            with st.spinner("Running PSM (nearest-neighbor matching)..."):
                # 내부용 이진 처리변수 생성 (__T: 1=treated, 0=control)
                work_df = df.copy()
                work_df["__T"] = np.where(work_df[treat_col] == treated_value, 1, 0)

                matched_df, ps_data = run_psm(
                    work_df,
                    treatment_col="__T",
                    covariates=covariates,
                    caliper=caliper,
                    ratio=int(match_ratio),
                    replace=bool(allow_replace),
                )

                if matched_df is None or ps_data is None:
                    st.error(
                        "매칭에 실패했습니다. (해당 caliper/배율에서 매칭 가능한 쌍이 부족하거나, "
                        "공변량에 결측치가 많을 수 있습니다.)\n\n"
                        "- caliper 값을 키우거나 (예: 0.3–0.5)\n"
                        "- 공변량 개수를 줄여서 다시 시도해 보세요."
                    )
                    # 이전 결과 정리 (widget key와 겹치지 않는 prefix 사용)
                    for k in [
                        "psm_result_done",
                        "psm_result_matched_df",
                        "psm_result_psdata_df",
                        "psm_result_treat_col",
                        "psm_result_treated_value",
                        "psm_result_covariates",
                        "psm_result_caliper",
                        "psm_result_ratio",
                        "psm_result_replace",
                        "psm_var_config",
                        "psm_var_signature",
                        "psm_group_labels",
                    ]:
                        st.session_state.pop(k, None)
                else:
                    st.session_state["psm_result_done"] = True
                    st.session_state["psm_result_matched_df"] = matched_df
                    st.session_state["psm_result_psdata_df"] = ps_data
                    st.session_state["psm_result_treat_col"] = treat_col
                    st.session_state["psm_result_treated_value"] = treated_value
                    st.session_state["psm_result_covariates"] = covariates
                    st.session_state["psm_result_caliper"] = caliper
                    st.session_state["psm_result_ratio"] = int(match_ratio)
                    st.session_state["psm_result_replace"] = bool(allow_replace)
                    # Table 1 설정 초기화
                    st.session_state.pop("psm_var_config", None)
                    st.session_state.pop("psm_var_signature", None)
                    st.session_state.pop("psm_group_labels", None)

    # ------------------------------------------------------------------
    # 3. 매칭 결과 표시 (SMD / 다운로드 / Table 1)
    # ------------------------------------------------------------------
    state = st.session_state
    if not state.get("psm_result_done", False):
        return

    matched_df = state.get("psm_result_matched_df")
    ps_data = state.get("psm_result_psdata_df")
    treat_col = state.get("psm_result_treat_col")
    treated_value = state.get("psm_result_treated_value")
    covariates = state.get("psm_result_covariates")
    caliper = state.get("psm_result_caliper")
    match_ratio = state.get("psm_result_ratio")
    allow_replace = state.get("psm_result_replace")

    if (
        matched_df is None
        or ps_data is None
        or treat_col is None
        or treated_value is None
        or covariates is None
    ):
        st.warning("PSM 결과 정보가 완전하지 않습니다. 다시 한 번 Run PSM을 눌러 주세요.")
        return

    # 3-0. N 요약 및 설정 요약
    n_all = len(df)
    n_cc = len(ps_data)
    n_cc_treated = int(ps_data["__T"].sum())
    n_cc_control = n_cc - n_cc_treated

    n_matched = len(matched_df)
    n_matched_treated = int((matched_df["__T"] == 1).sum())
    n_matched_control = n_matched - n_matched_treated

    st.success(
        f"✅ Matching complete!  "
        f"(원본 N={n_all} → complete-case N={n_cc} [treated={n_cc_treated}, control={n_cc_control}] "
        f"→ 매칭 후 N={n_matched} [treated={n_matched_treated}, control={n_matched_control}])"
    )
    st.caption(
        f"PSM 설정: caliper = {caliper:.2f} × SD(logit PS), "
        f"ratio = 1:{match_ratio}, "
        f"{'with' if allow_replace else 'without'} replacement"
    )

    # --------------------------------------------------------------
    # 3-1. Balance Check: SMD before/after
    # --------------------------------------------------------------
    st.markdown("### ⚖️ Balance Check (SMD)")
    st.caption("일반적으로 |SMD| < 0.1 이면 두 군의 공변량 분포가 잘 맞는 것으로 봅니다.")

    smd_before = calculate_smd(ps_data, "__T", covariates)
    smd_after = calculate_smd(matched_df, "__T", covariates)

    smd_before = smd_before.rename(columns={"SMD": "SMD_Before"})
    smd_after = smd_after.rename(columns={"SMD": "SMD_After"})
    smd_merged = pd.merge(smd_before, smd_after, on="Variable", how="outer")

    st.dataframe(
        smd_merged.style.format({"SMD_Before": "{:.3f}", "SMD_After": "{:.3f}"}),
        use_container_width=True,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=smd_merged,
        x="SMD_Before",
        y="Variable",
        label="Before Matching",
        s=100,
        ax=ax,
    )
    sns.scatterplot(
        data=smd_merged,
        x="SMD_After",
        y="Variable",
        label="After Matching",
        s=100,
        ax=ax,
    )
    ax.axvline(0.1, ls="--", color="gray", alpha=0.5)
    ax.axvline(-0.1, ls="--", color="gray", alpha=0.5)
    ax.set_title("Standardized Mean Differences (Before vs After)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    # 매칭된 데이터 다운로드 (내부 컬럼(__T, logit_ps, propensity_score) 제외 가능)
    out_buf = io.BytesIO()
    with pd.ExcelWriter(out_buf, engine="xlsxwriter") as w:
        matched_df.drop(
            columns=["__T", "logit_ps", "propensity_score"],
            errors="ignore",
        ).to_excel(w, index=False)
    st.download_button(
        "📥 Download Matched Data",
        data=out_buf.getvalue(),
        file_name="Matched_Data.xlsx",
    )

    # --------------------------------------------------------------
    # 3-2. Matched Cohort Table 1 (Tab1과 동일한 방식의 변수 선택)
    # --------------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Matched Cohort Table 1")

    # 3-2-1. 그룹 라벨 설정 (Table 1 헤더용)
    group_values = list(matched_df[treat_col].dropna().unique())
    if len(group_values) < 2:
        st.warning("매칭된 데이터에서 Treatment 변수의 그룹 수가 2 미만입니다.")
        return

    group_labels = state.get("psm_group_labels") or {gv: str(gv) for gv in group_values}
    for gv in group_values:
        group_labels.setdefault(gv, str(gv))
    state["psm_group_labels"] = group_labels

    with st.expander("⚙️ Group Labels for Matched Table 1", expanded=True):
        updated_labels = {}
        for gv in group_values:
            default_label = group_labels.get(gv, str(gv))
            label = st.text_input(
                f"Label for group value '{gv}'",
                value=default_label,
                key=f"psm_group_label_{treat_col}_{str(gv)}",
            )
            clean_label = label.strip() if label.strip() else str(gv)
            updated_labels[gv] = clean_label
        state["psm_group_labels"] = updated_labels

    value_map = state["psm_group_labels"]

    # 3-2-2. Table 1 변수 선택 (Tab1과 유사한 UX)
    candidate_cols = [
        c
        for c in matched_df.columns
        if c not in ["__T", "logit_ps", "propensity_score", treat_col]
    ]

    var_signature = (tuple(sorted(candidate_cols)), tuple(sorted(covariates)))
    prev_sig = state.get("psm_var_signature")
    need_init_cfg = prev_sig != var_signature or state.get("psm_var_config") is None

    if need_init_cfg:
        init_rows = []
        for col in candidate_cols:
            include_default = col not in covariates  # 매칭 공변량은 기본 False, 나머지는 기본 True
            init_rows.append(
                {
                    "Include": include_default,
                    "Variable": col,
                    "Type": suggest_variable_type_single(matched_df, col),
                }
            )
        state["psm_var_config"] = pd.DataFrame(init_rows)
        state["psm_var_signature"] = var_signature

    # Select All / Deselect All
    c_sa1, c_sa2, _ = st.columns([0.2, 0.2, 0.6])
    if c_sa1.button("✅ Select All (Matched)", key="psm_tbl_all"):
        cfg = state.get("psm_var_config")
        if cfg is not None:
            cfg["Include"] = True
            state["psm_var_config"] = cfg
        st.rerun()
    if c_sa2.button("⬜ Deselect All (Matched)", key="psm_tbl_none"):
        cfg = state.get("psm_var_config")
        if cfg is not None:
            cfg["Include"] = False
            state["psm_var_config"] = cfg
        st.rerun()

    # 편집 가능한 변수 설정 테이블
    cfg_df = st.data_editor(
        state["psm_var_config"],
        column_config={
            "Include": st.column_config.CheckboxColumn(width="small"),
            "Variable": st.column_config.TextColumn(disabled=True),
            "Type": st.column_config.SelectboxColumn(
                options=["Continuous", "Categorical"]
            ),
        },
        hide_index=True,
        num_rows="fixed",
        key="psm_tbl_editor",
        use_container_width=True,
    )
    state["psm_var_config"] = cfg_df

    # Include=True 인 변수들만 사용
    sel = cfg_df[cfg_df["Include"] == True]
    target_vars = sel["Variable"].tolist()
    cont_vars = sel[sel["Type"] == "Continuous"]["Variable"].tolist()
    cat_vars = sel[sel["Type"] == "Categorical"]["Variable"].tolist()

    if st.button("Generate Matched Table 1", key="psm_tbl_run"):
        if not target_vars:
            st.warning("Table 1에 포함할 변수를 최소 1개 이상 선택해야 합니다.")
        else:
            t1_res, err_info = analyze_table1_robust(
                matched_df,
                group_col=treat_col,
                value_map=value_map,
                target_cols=target_vars,
                user_cont_vars=cont_vars,
                user_cat_vars=cat_vars,
            )
            if err_info:
                var_name = err_info.get("var", "Unknown")
                msg = err_info.get("msg", "")
                st.error(f"🚨 Table 1 생성 중 오류: 변수 '{var_name}' 에서 문제가 발생했습니다.")
                if msg:
                    st.warning(f"Details: {msg}")
            else:
                st.dataframe(t1_res, use_container_width=True)
                out_t1 = io.BytesIO()
                with pd.ExcelWriter(out_t1, engine="xlsxwriter") as w:
                    t1_res.to_excel(w, index=False)
                st.download_button(
                    "📥 Download Matched Table 1",
                    data=out_t1.getvalue(),
                    file_name="Matched_Table1.xlsx",
                )
