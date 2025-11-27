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
    # --------------------------------------------------------------
    # 0. Intro
    # --------------------------------------------------------------
    st.subheader("Propensity Score Matching (PSM)")
    st.info(
        "\n".join(
            [
                "💡 **PSM**은 관찰 연구에서 선택 편향을 줄이기 위해,"
                " 치료군과 대조군의 propensity score(치료를 받을 확률)를 맞춰주는 방법입니다.",
                "",
                "- **Treatment Variable**: 치료군/대조군을 구분하는 변수 (값이 2개여야 함)",
                "- **Covariates for Matching**: PS 계산 및 매칭에 사용할 공변량",
                "- 매칭 후에는 **Matched Cohort Table 1**에서 공변량뿐 아니라 다른 변수들도 자유롭게 비교할 수 있습니다.",
            ]
        )
    )

    # --------------------------------------------------------------
    # 1. PSM 설정 입력
    # --------------------------------------------------------------
    c1, c2 = st.columns(2)

    # 1-1. Treatment 변수 선택
    treatment_col = c1.selectbox(
        "Treatment Variable (이분형)",
        options=df.columns,
        key="psm_treatment_col",
        help="치료군과 대조군을 구분하는 변수를 선택하세요 (값이 정확히 2개여야 합니다).",
    )

    treated_value = None
    can_use_treatment = False

    if treatment_col:
        unique_vals = df[treatment_col].dropna().unique()
        if len(unique_vals) == 2:
            can_use_treatment = True
            treated_value = c2.selectbox(
                "Treated value (case로 간주할 값)",
                options=unique_vals,
                key="psm_treated_value",
                help="이 값을 갖는 환자를 치료군(1)으로, 나머지를 대조군(0)으로 사용합니다.",
            )
        else:
            c2.warning("⚠️ 선택한 Treatment 변수는 **정확히 2개의 값**만 가져야 합니다.")

    # 1-2. 공변량 선택 (Treatment 변수 제외)
    covariates = st.multiselect(
        "Covariates for Matching (PS 계산 및 매칭에 사용할 변수)",
        options=[c for c in df.columns if c != treatment_col],
        key="psm_covariates",
        help="너무 많은 공변량을 동시에 선택하면 NA가 많을 경우 매칭 가능한 환자 수가 줄어듭니다.",
    )

    # 1-3. Caliper 설정
    caliper = st.slider(
        "Caliper width (SD of logit PS)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
        help="logit(propensity score)의 표준편차를 1로 보았을 때의 caliper 폭입니다. 보통 0.2를 많이 사용합니다.",
    )

    # --------------------------------------------------------------
    # 2. PSM 실행
    # --------------------------------------------------------------
    if st.button("Run PSM", key="psm_run_button"):
        # 기본 입력 체크
        if not can_use_treatment or treated_value is None:
            st.error("Treatment 변수와 Treated value를 올바르게 선택해 주세요.")
        elif not covariates:
            st.error("매칭에 사용할 공변량을 최소 1개 이상 선택해 주세요.")
        else:
            with st.spinner("Running PSM (1:1 matching with caliper)..."):
                # 내부용 이진 처리 변수 생성 (__T: 1=treated, 0=control)
                df_psm = df.copy()
                df_psm["__T"] = np.where(df_psm[treatment_col] == treated_value, 1, 0)

                # run_psm는 df, treatment_col(0/1), covariates, caliper 를 인자로 받습니다.
                matched_df, ps_data = run_psm(
                    df_psm,
                    treatment_col="__T",
                    covariates=covariates,
                    caliper=caliper,
                )

                if matched_df is None or ps_data is None:
                    st.error(
                        "매칭에 실패했습니다. (해당 caliper 내에서 매칭 가능한 쌍이 없거나, "
                        "공변량에 결측치가 많아서 분석 가능한 표본이 부족할 수 있습니다.)\n\n"
                        "- caliper 값을 조금 키워 보거나 (예: 0.3–0.5)\n"
                        "- 공변량 개수를 줄여서 다시 시도해 보세요."
                    )
                    # 실패 시 이전 결과는 삭제
                    st.session_state.pop("psm_done", None)
                    st.session_state.pop("psm_matched_df", None)
                    st.session_state.pop("psm_psdata_df", None)
                    st.session_state.pop("psm_treatment_col", None)
                    st.session_state.pop("psm_treated_value", None)
                    st.session_state.pop("psm_covariates", None)
                    st.session_state.pop("psm_var_config", None)
                    st.session_state.pop("psm_var_signature", None)
                    st.session_state.pop("psm_group_labels", None)
                else:
                    # 성공 시 결과 저장
                    st.session_state["psm_done"] = True
                    st.session_state["psm_matched_df"] = matched_df
                    st.session_state["psm_psdata_df"] = ps_data
                    st.session_state["psm_treatment_col"] = treatment_col
                    st.session_state["psm_treated_value"] = treated_value
                    st.session_state["psm_covariates"] = covariates
                    # 매칭이 바뀌었으므로 Table 1 / 그룹명 설정 초기화
                    st.session_state.pop("psm_var_config", None)
                    st.session_state.pop("psm_var_signature", None)
                    st.session_state.pop("psm_group_labels", None)

    # --------------------------------------------------------------
    # 3. 매칭 결과 표시 (SMD / 다운로드 / Table 1)
    # --------------------------------------------------------------
    if not st.session_state.get("psm_done", False):
        return

    matched_df = st.session_state.get("psm_matched_df")
    ps_data = st.session_state.get("psm_psdata_df")
    treatment_col = st.session_state.get("psm_treatment_col")
    treated_value = st.session_state.get("psm_treated_value")
    covariates = st.session_state.get("psm_covariates")

    # 세션이 꼬였거나 중간에 일부 키만 삭제된 경우를 방지
    if (
        matched_df is None
        or ps_data is None
        or treatment_col is None
        or treated_value is None
        or covariates is None
    ):
        st.warning("PSM 결과 정보가 완전하지 않습니다. 다시 한 번 Run PSM을 눌러 주세요.")
        return

    # --------------------------------------------------------------
    # 3-0. N 설명
    # --------------------------------------------------------------
    st.success(f"✅ Matching complete! Matched N = {len(matched_df)}")

    # run_psm 안에서 사용된 complete-case 데이터(ps_data)를 기준으로 N을 보여줌
    n_all = len(df)
    n_cc = len(ps_data)
    n_cc_treated = int(ps_data["__T"].sum())
    n_cc_control = n_cc - n_cc_treated

    n_matched = len(matched_df)
    # matched_df는 df_psm로부터 만들어졌기 때문에 __T 컬럼이 남아 있음
    n_matched_treated = int((matched_df["__T"] == 1).sum())
    n_matched_control = n_matched - n_matched_treated

    st.caption(
        f"원본 데이터: N={n_all}  →  "
        f"PSM에 사용된 complete-case 데이터: N={n_cc} (treated={n_cc_treated}, control={n_cc_control})  →  "
        f"매칭 후: N={n_matched} (treated={n_matched_treated}, control={n_matched_control})"
    )

    # --------------------------------------------------------------
    # 3-1. Balance Check (SMD)
    # --------------------------------------------------------------
    st.markdown("### ⚖️ Balance Check (SMD)")
    st.caption("표준화 차이(|SMD|) < 0.1 이면 두 군의 공변량 분포가 잘 맞는 것으로 봅니다.")

    # 매칭 전: run_psm 내부에서 사용된 complete-case 데이터 기준
    smd_before = calculate_smd(ps_data, "__T", covariates)
    # 매칭 후: matched_df 기준
    smd_after = calculate_smd(matched_df, "__T", covariates)

    smd_before = smd_before.rename(columns={"SMD": "SMD_Before"})
    smd_after = smd_after.rename(columns={"SMD": "SMD_After"})
    smd_merged = pd.merge(smd_before, smd_after, on="Variable", how="outer")

    st.dataframe(
        smd_merged.style.format({"SMD_Before": "{:.3f}", "SMD_After": "{:.3f}"}),
        use_container_width=True,
    )

    # SMD 플롯
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=smd_merged,
        x="SMD_Before",
        y="Variable",
        label="Before matching",
        s=100,
        ax=ax,
    )
    sns.scatterplot(
        data=smd_merged,
        x="SMD_After",
        y="Variable",
        label="After matching",
        s=100,
        ax=ax,
    )
    ax.axvline(0.1, ls="--", color="gray", alpha=0.5)
    ax.axvline(-0.1, ls="--", color="gray", alpha=0.5)
    ax.set_title("Standardized Mean Differences (Before vs After)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    # 매칭된 데이터 다운로드 (내부 컬럼 제외)
    out_p = io.BytesIO()
    with pd.ExcelWriter(out_p, engine="xlsxwriter") as w:
        matched_df.drop(
            columns=["__T", "logit_ps", "propensity_score"],
            errors="ignore",
        ).to_excel(w, index=False)
    st.download_button(
        "📥 Download Matched Data",
        data=out_p.getvalue(),
        file_name="Matched_Data.xlsx",
    )

    # --------------------------------------------------------------
    # 3-2. Matched Cohort Table 1
    # --------------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Matched Cohort Table 1")

    # 3-2-1. 그룹명 설정 (Table 1 헤더에 사용)
    group_values = list(matched_df[treatment_col].dropna().unique())
    if not group_values:
        st.warning("매칭된 데이터에서 Treatment 변수의 값이 없습니다.")
        return

    # 초기 그룹 라벨 설정
    stored_labels = st.session_state.get("psm_group_labels")
    if stored_labels is None:
        stored_labels = {gv: str(gv) for gv in group_values}
        st.session_state["psm_group_labels"] = stored_labels
    else:
        # 만약 새로운 그룹 값이 생겼다면 기본 라벨을 채워 넣기
        for gv in group_values:
            if gv not in stored_labels:
                stored_labels[gv] = str(gv)
        st.session_state["psm_group_labels"] = stored_labels

    with st.expander("⚙️ Matched Group Names (Table 1 Header 설정)", expanded=True):
        new_labels = {}
        for gv in group_values:
            default_label = stored_labels.get(gv, str(gv))
            label = st.text_input(
                f"Label for group value '{gv}'",
                value=default_label,
                key=f"psm_group_label_{treatment_col}_{str(gv)}",
            )
            label_clean = label.strip() if label.strip() else str(gv)
            new_labels[gv] = label_clean
        st.session_state["psm_group_labels"] = new_labels

    value_map = st.session_state["psm_group_labels"]

    # 3-2-2. Table 1에 사용할 변수 선택
    # 매칭된 데이터에서 Table 1 후보 변수들:
    #  - treatment_col, 내부 변수(__T, logit_ps, propensity_score)는 제외
    available_cols = [
        c
        for c in matched_df.columns
        if c not in ["__T", "logit_ps", "propensity_score", treatment_col]
    ]

    # covariates / 후보 변수 구성이 바뀌었는지 확인하기 위한 signature
    current_signature = (tuple(sorted(available_cols)), tuple(sorted(covariates)))

    need_init = False
    if "psm_var_config" not in st.session_state:
        need_init = True
    else:
        prev_sig = st.session_state.get("psm_var_signature")
        if prev_sig != current_signature:
            need_init = True

    if need_init:
        init_rows = []
        for c in available_cols:
            # 매칭에 사용한 공변량(covariates)은 기본적으로 Include=False,
            # 나머지 변수들은 기본적으로 Include=True
            include_flag = c not in covariates
            init_rows.append(
                {
                    "Include": include_flag,
                    "Variable": c,
                    "Type": suggest_variable_type_single(matched_df, c),
                }
            )
        st.session_state["psm_var_config"] = pd.DataFrame(init_rows)
        st.session_state["psm_var_signature"] = current_signature

    # Select All / Deselect All 버튼
    c_b1, c_b2, _ = st.columns([0.2, 0.2, 0.6])
    if c_b1.button("✅ Select All (Matched)", key="psm_all"):
        cfg = st.session_state.get("psm_var_config")
        if cfg is not None:
            cfg["Include"] = True
            st.session_state["psm_var_config"] = cfg
        st.rerun()
    if c_b2.button("⬜ Deselect All (Matched)", key="psm_none"):
        cfg = st.session_state.get("psm_var_config")
        if cfg is not None:
            cfg["Include"] = False
            st.session_state["psm_var_config"] = cfg
        st.rerun()

    # 변수/타입 편집용 테이블
    var_config_df = st.data_editor(
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
        use_container_width=True,
    )
    st.session_state["psm_var_config"] = var_config_df

    # Include=True 로 남긴 변수들만 Table 1 대상으로 사용
    selected_rows = var_config_df[var_config_df["Include"] == True]
    target_vars = selected_rows["Variable"].tolist()
    user_cont_vars = selected_rows[selected_rows["Type"] == "Continuous"][
        "Variable"
    ].tolist()
    user_cat_vars = selected_rows[selected_rows["Type"] == "Categorical"][
        "Variable"
    ].tolist()

    if st.button("Generate Matched Table 1", key="btn_mt1"):
        if not target_vars:
            st.warning("Table 1에 포함할 변수를 최소 1개 이상 선택하세요.")
        else:
            t1_res, error_info = analyze_table1_robust(
                matched_df,
                treatment_col,
                value_map,
                target_vars,
                user_cont_vars,
                user_cat_vars,
            )
            if error_info:
                st.error(
                    f"🚨 Table 1 생성 중 오류: 변수 '{error_info.get('var')}' 에서 문제가 발생했습니다."
                )
                if "msg" in error_info:
                    st.warning(f"Details: {error_info['msg']}")
            else:
                st.success("Matched Cohort Table 1 Generated!")
                st.dataframe(t1_res, use_container_width=True)

                out_m1 = io.BytesIO()
                with pd.ExcelWriter(out_m1, engine="xlsxwriter") as w:
                    t1_res.to_excel(w, index=False)
                st.download_button(
                    "📥 Download Matched Table 1",
                    data=out_m1.getvalue(),
                    file_name="Matched_Table1.xlsx",
                )
