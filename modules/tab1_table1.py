import io
import pandas as pd
import streamlit as st
import xlsxwriter

from .missing import apply_missing_policy
from .utils import analyze_table1_robust, suggest_variable_type_single


def render_tab1(df: pd.DataFrame) -> None:
    """
    Table 1 (Baseline Characteristics)
    - 변수 포함: multiselect
    - 연속형 지정: 포함된 변수 중에서 multiselect로 Continuous 선택
    - 나머지는 자동으로 Categorical 처리
    """
    st.subheader("📊 Table 1: Baseline Characteristics")

    # 1. Group Column
    group_col = st.selectbox(
        "Group Column (그룹 변수)",
        df.columns,
        key="t1_group_col",
        help="비교할 그룹을 나누는 변수입니다. (예: Treatment, Sex 등)",
    )
    if not group_col:
        st.info("Please select a group column.")
        return

    # 2. Group Values
    unique_vals = sorted(df[group_col].dropna().astype(str).unique())

    c1, _ = st.columns(2)
    with c1:
        st.write(f"**Unique values in '{group_col}':** {unique_vals}")

    selected_vals = st.multiselect(
        "Select Group Values to Compare",
        unique_vals,
        default=unique_vals[:2] if len(unique_vals) >= 2 else unique_vals,
        key="t1_selected_vals",
        help="비교에 사용할 그룹 값을 2개 이상 선택하세요.",
    )
    if len(selected_vals) < 2:
        st.warning("⚠️ Please select at least 2 group values for comparison.")
        return

    # 3. Group Label Map
    value_map = {}
    st.markdown("##### Group Labels (Optional)")
    cols = st.columns(len(selected_vals))
    for i, val in enumerate(selected_vals):
        new_label = cols[i].text_input(
            f"Label for '{val}'",
            value=str(val),
            key=f"t1_label_{val}",
            help="논문 Table 1에 표시될 그룹 이름입니다.",
        )
        value_map[val] = new_label

    st.write("---")

    # 4. Variable Configuration
    st.markdown("#### ⚙️ Variable Configuration")

    all_vars = [c for c in df.columns if c != group_col]

    # (1) Table 1에 포함할 변수 선택
    include_default = st.session_state.get("t1_include_vars", all_vars)
    include_default = [v for v in include_default if v in all_vars]

    include_vars = st.multiselect(
        "Variables to include in Table 1",
        all_vars,
        default=include_default if include_default else all_vars,
        key="t1_include_vars",
        help="Table 1에 포함할 변수를 선택하세요.",
    )

    if not include_vars:
        st.info("Please select at least one variable to analyze.")
        return

    # (2) 포함된 변수 중에서 연속형 변수 선택 (나머지는 범주형)
    auto_cont = [
        v for v in include_vars
        if suggest_variable_type_single(df, v) == "Continuous"
    ]
    prev_cont = st.session_state.get("t1_cont_vars", auto_cont)
    prev_cont = [v for v in prev_cont if v in include_vars]
    if not prev_cont:
        prev_cont = auto_cont

    cont_vars = st.multiselect(
        "Continuous variables (나머지는 Categorical로 처리)",
        include_vars,
        default=prev_cont,
        key="t1_cont_vars",
        help="연속형(Mean±SD 또는 Median[IQR])으로 보고 싶은 변수만 선택하세요.",
    )

    cat_vars = [v for v in include_vars if v not in cont_vars]

    # 4.5 Missing value policy (결측치 처리 방식)
    st.markdown("#### 🧩 Missing Data Handling")
    
    missing_options = [
        "Variable-wise drop (per analysis)",
        "Complete-case (drop rows with ANY missing)",
        "Categorical: treat missing as 'Missing' (numeric untouched)",
        "Simple imputation (numeric=median, categorical=mode)",
    ]
    
    default_policy = st.session_state.get("missing_policy", missing_options[0])
    
    policy = st.selectbox(
        "Missing value policy (결측치 처리 방식)",
        missing_options,
        index=missing_options.index(default_policy) if default_policy in missing_options else 0,
        key="missing_policy",
        help=(
            "Variable-wise: 변수별 분석 시 해당 변수에서만 결측 제외(표본수 최대화)\n"
            "Complete-case: 포함 변수 중 결측이 하나라도 있으면 해당 행 제거(표본수 감소)\n"
            "Categorical Missing: 범주형 결측을 'Missing' 범주로 포함\n"
            "Simple imputation: 수치형=중앙값, 범주형=최빈값으로 대체"
        ),
    )

    # 5. 분석 실행
    if st.button("Generate Table 1", key="t1_btn_run"):
        cols_for_analysis = [group_col] + include_vars
        df_use = apply_missing_policy(df, cols_for_analysis, policy)

        with st.spinner("Analyzing... (including normality tests)"):
            t1_res, error_info = analyze_table1_robust(
                df_use,
                group_col,
                value_map,
                include_vars,
                cont_vars,
                cat_vars,
            )

        if error_info:
            st.error(f"🚨 **Data Error: '{error_info['var']}'**")
            st.warning(f"Details: {error_info['msg']}")
            return

        st.success("Table 1 Generated!")

        # 화면 표시용 스타일: 상위 변수는 bold
        def style_table1(df_table: pd.DataFrame):
            def highlight_head(row):
                ch = str(row["Characteristic"])
                is_head = not ch.startswith("  ")
                return [
                    "font-weight: 700;" if is_head else ""
                    for _ in row
                ]

            return df_table.style.apply(highlight_head, axis=1)

        st.dataframe(style_table1(t1_res), use_container_width=True)

        # Excel 다운로드 (SCIE 스타일 기본틀)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            sheet_name = "Table1"
            t1_res.to_excel(writer, index=False, sheet_name=sheet_name)
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]

            header_fmt = workbook.add_format(
                {
                    "bold": True,
                    "align": "center",
                    "valign": "vcenter",
                    "font_name": "Times New Roman",
                    "font_size": 10,
                    "bottom": 1,
                }
            )
            body_fmt = workbook.add_format(
                {
                    "font_name": "Times New Roman",
                    "font_size": 10,
                }
            )
            head_row_fmt = workbook.add_format(
                {
                    "font_name": "Times New Roman",
                    "font_size": 10,
                    "bold": True,
                }
            )

            # 헤더 스타일
            for col_idx, col_name in enumerate(t1_res.columns):
                worksheet.write(0, col_idx, col_name, header_fmt)

            # 본문 행 스타일 (상위 변수 bold)
            for row_idx in range(1, len(t1_res) + 1):
                char_val = str(t1_res.iloc[row_idx - 1, 0])
                if not char_val.startswith("  "):
                    worksheet.set_row(row_idx, None, head_row_fmt)
                else:
                    worksheet.set_row(row_idx, None, body_fmt)

            # 열 폭 자동 조정
            for col_idx, col_name in enumerate(t1_res.columns):
                max_len = max(
                    [len(str(col_name))]
                    + [len(str(v)) for v in t1_res.iloc[:, col_idx]]
                )
                worksheet.set_column(col_idx, col_idx, max_len + 2)

        st.download_button(
            "📥 Download Excel (SCIE style)",
            output.getvalue(),
            "Table1_Robust.xlsx",
        )


