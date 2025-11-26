# Page Config & Imports
import io

import numpy as np
import pandas as pd
import streamlit as st

from modules.tab1_table1 import render_tab1
from modules.tab2_cox import render_tab2
from modules.tab3_logistic import render_tab3
from modules.tab4_psm import render_tab4
from modules.tab5_methods import render_tab5

# ------------------------------------------------------------
# 기본 페이지 설정
# ------------------------------------------------------------
st.set_page_config(
    page_title="Medical Statistics Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# 공통 CSS (다크 + 파스텔 느낌)
# ------------------------------------------------------------
BASE_CSS = """
<style>
/* Global Font */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Headers */
h1, h2, h3 {
    color: #00ADB5 !important;
    font-weight: 700;
}

/* Buttons */
.stButton > button {
    background-color: #00ADB5;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background-color: #007A80;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

/* Cards/Containers */
.stDataFrame, .stTable {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #393E46;
}

/* Info Boxes */
.stAlert {
    border-radius: 8px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #222831;
    border-right: 1px solid #393E46;
}

/* ✅ 모든 체크박스 크게 (Table1 / PSM / Sidebar 전부) */
input[type="checkbox"] {
    transform: scale(1.6);
    margin-right: 6px;
    cursor: pointer;
}
</style>
"""


st.markdown(BASE_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------
# 유틸 함수
# ------------------------------------------------------------

def reset_session_state(new_file_id: str) -> None:
    """새 파일/시트를 로드했을 때 관련 상태를 초기화."""
    last_id = st.session_state.get("last_file_id")
    if last_id != new_file_id:
        keys_to_clear = [
            # Table 1
            "t1_group_col",
            "t1_selected_vals",
            "t1_include_vars",
            "t1_cont_vars",
            # PSM
            "psm_matched_df",
            "psm_org_df",
            "psm_covs",
            "psm_treat_col",
            "psm_include_vars",
            "psm_cont_vars",
            "psm_run_done",
            # 예전 키들 정리 (혹시 남아있으면)
            "var_config_df",
            "psm_var_config",
        ]
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state["last_file_id"] = new_file_id
        # ⚠️ 여기서는 st.rerun() 안 씀 → 클릭할 때 불필요한 재실행 방지


def load_data(uploaded_file):
    """CSV / Excel 파일을 DataFrame으로 로드."""
    try:
        df = None
        file_ext = uploaded_file.name.split(".")[-1].lower()

        # 파일 포인터 초기화
        uploaded_file.seek(0)

        selected_sheet = None

        if file_ext == "csv":
            use_header = st.checkbox(
                "Use first row as header",
                value=True,
                key="csv_use_header",
                help="체크를 끄면 첫 행도 데이터로 취급합니다.",
            )
            header_opt = 0 if use_header else None
            df = pd.read_csv(uploaded_file, header=header_opt)

        elif file_ext in ["xlsx", "xls"]:
            # Excel 파일 로드 (openpyxl)
            xl = pd.ExcelFile(uploaded_file, engine="openpyxl")
            sheet_names = xl.sheet_names

            # 시트 선택
            selected_sheet = sheet_names[0]
            if len(sheet_names) > 1:
                selected_sheet = st.selectbox(
                    "Select Sheet", sheet_names, key="sheet_selector"
                )

            use_header = st.checkbox(
                "Use first row as header",
                value=True,
                key="excel_use_header",
                help="체크를 끄면 첫 행도 데이터로 취급합니다.",
            )
            header_opt = 0 if use_header else None
            df = xl.parse(selected_sheet, header=header_opt)

        else:
            st.error("Unsupported file format. Please upload CSV or XLSX.")
            return None, None

        return df, selected_sheet

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, None


# ------------------------------------------------------------
# 메인 앱
# ------------------------------------------------------------

def main():
    st.title("📊 Medical Statistics Analysis Tool")
    st.caption("Integrated Statistical Analysis Platform for Medical Research")

    # ---- Sidebar ----
    with st.sidebar:
        st.header("📂 Data Upload & Settings")
        st.info("Upload Excel (.xlsx) or CSV (.csv) file.")
        uploaded_file = st.file_uploader("Select File", type=["xlsx", "csv"])

        st.write("---")
        st.markdown("### 🧹 Missing Data")
        missing_policy = st.radio(
            "How to handle missing values?",
            [
                "Variable-wise drop (per analysis)",
                "Complete-case (drop rows with any NA in used vars)",
                "Categorical: treat NA as 'Missing'",
                "Simple imputation (median/mode)",
            ],
            key="missing_policy",
            help="Table 1, Cox, Logistic, PSM 모두에 공통 적용할 결측값 처리 방식을 선택합니다.",
        )

        st.write("---")
        st.markdown("### ℹ️ Help")
        st.markdown(
            """
        - **Table 1**: Baseline characteristics (t-test, Chi-square, etc.)
        - **Cox Regression**: Survival analysis (Kaplan-Meier, Cox PH)
        - **Logistic Regression**: Binary outcome prediction (ROC curve)
        - **PSM**: Propensity score matching
        """
        )

    # ---- 메인 영역 ----
    if uploaded_file is not None:
        df, sheet_name = load_data(uploaded_file)

        if df is not None:
            # 파일 ID 생성 (이름 + 크기 + 시트명)
            current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if sheet_name:
                current_file_id += f"_{sheet_name}"

            # 새 파일이면 세션 리셋
            reset_session_state(current_file_id)

            st.success("File uploaded successfully!")
            st.write(f"**Shape**: {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head())

            # 탭 생성
            tab1, tab2, tab3, tab4, tab_methods = st.tabs(
                [
                    "📊 Table 1 (Baseline)",
                    "⏱️ Cox Regression",
                    "💊 Logistic Regression",
                    "⚖️ PSM (Matching)",
                    "📝 Methods Draft",
                ]
            )

            # 각 탭 렌더링
            with tab1:
                render_tab1(df)
            with tab2:
                render_tab2(df)
            with tab3:
                render_tab3(df)
            with tab4:
                render_tab4(df)
            with tab_methods:
                render_tab5()
    else:
        st.info("👈 Please upload a data file from the sidebar.")


if __name__ == "__main__":
    main()
