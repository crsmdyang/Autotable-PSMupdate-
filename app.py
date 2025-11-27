# Page Config
import streamlit as st
import os
from datetime import datetime


st.set_page_config(
    page_title="Medical Statistics Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
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
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        border-radius: 8px 8px 0 0;
        background-color: #393E46;
        color: #EEEEEE;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ADB5 !important;
        color: white !important;
    }
    
    /* DataFrames */
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
    </style>
    """, unsafe_allow_html=True)

# Imports
try:
    import pandas as pd
    import numpy as np
    import io
    from modules.tab1_table1 import render_tab1
    from modules.tab2_cox import render_tab2
    from modules.tab3_logistic import render_tab3
    from modules.tab4_psm import render_tab4
    from modules.tab5_methods import render_tab5
except ImportError as e:
    st.error(f"Module Import Error: {e}")
    st.stop()

# ------------------------------------------------------------------
# Simple User Authentication (Login / Signup / Password reset / ID find)
# ------------------------------------------------------------------
USER_DB_PATH = "users_db.csv"
USER_DB_COLUMNS = [
    "user_id",
    "password",
    "hospital",
    "affiliation",
    "position",
    "name",
    "role",
    "created_at",
]


def _init_user_db():
    """Ensure that the user DB exists and has at least one admin account."""
    if not os.path.exists(USER_DB_PATH):
        df = pd.DataFrame(columns=USER_DB_COLUMNS)
        # 기본 관리자 계정 (ID/비밀번호는 필요에 따라 수정하세요)
        df.loc[len(df)] = [
            "admin",            # user_id
            "admin1234",        # password
            "Admin Hospital",   # hospital
            "Admin",            # affiliation
            "관리자",            # position
            "Administrator",    # name
            "admin",            # role
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ]
        df.to_csv(USER_DB_PATH, index=False, encoding="utf-8")


def load_user_db():
    """Load user database as DataFrame."""
    _init_user_db()
    try:
        df = pd.read_csv(USER_DB_PATH, dtype=str, encoding="utf-8")
    except Exception:
        df = pd.DataFrame(columns=USER_DB_COLUMNS)
    # 컬럼 누락 시 보정
    for col in USER_DB_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[USER_DB_COLUMNS]


def save_user_db(df):
    """Save user database DataFrame."""
    try:
        df.to_csv(USER_DB_PATH, index=False, encoding="utf-8")
    except Exception as e:
        st.error(f"사용자 DB 저장 중 오류가 발생했습니다: {e}")


def render_auth_block():
    """Render login / signup / password reset / ID find UI.

    Returns
    -------
    user : dict or None
        Logged-in user info if authenticated, else None.
    users_df : pd.DataFrame
        Current user database.
    """
    users_df = load_user_db()
    current_user = st.session_state.get("current_user")

    # 이미 로그인 된 경우: 간단 정보 + 로그아웃 버튼
    if current_user is not None:
        with st.sidebar:
            st.success(
                f"로그인: {current_user.get('name', '')} "
                f"({current_user.get('user_id', '')})"
            )
            if st.button("로그아웃", key="btn_logout"):
                st.session_state.pop("current_user", None)
                st.experimental_rerun()
        return current_user, users_df

    # 로그인되지 않은 상태: 로그인/회원가입/비밀번호 찾기/아이디 찾기 탭
    st.markdown("### 🔐 로그인")
    tab_login, tab_signup, tab_reset_pw, tab_find_id = st.tabs(
        ["로그인", "회원가입", "비밀번호 재설정", "아이디 찾기"]
    )

    # -------------------------------
    # 로그인 탭
    # -------------------------------
    with tab_login:
        login_id = st.text_input("아이디", key="login_id")
        login_pw = st.text_input("비밀번호", type="password", key="login_pw")

        if st.button("로그인", key="btn_login"):
            row = users_df[users_df["user_id"] == login_id]
            if row.empty:
                st.error("존재하지 않는 아이디입니다.")
            else:
                stored_pw = str(row.iloc[0]["password"])
                if stored_pw != login_pw:
                    st.error("비밀번호가 올바르지 않습니다.")
                else:
                    user = row.iloc[0].to_dict()
                    st.session_state["current_user"] = user
                    st.success(f"{user.get('name', '')}님 환영합니다.")
                    st.experimental_rerun()

    # -------------------------------
    # 회원가입 탭
    # -------------------------------
    with tab_signup:
        st.markdown("#### 새 계정 생성")

        reg_id = st.text_input("아이디", key="reg_id")
        reg_pw = st.text_input("비밀번호", type="password", key="reg_pw")
        reg_pw2 = st.text_input("비밀번호 확인", type="password", key="reg_pw2")
        reg_hospital = st.text_input("병원명", key="reg_hospital")
        reg_affiliation = st.text_input("소속 (예: 대장항문외과)", key="reg_affiliation")
        reg_position = st.text_input("직책 (예: 교수)", value="교수", key="reg_position")
        reg_name = st.text_input("이름", key="reg_name")

        if st.button("회원가입", key="btn_signup"):
            if not reg_id or not reg_pw or not reg_name:
                st.error("아이디, 비밀번호, 이름은 필수 입력 항목입니다.")
            elif reg_pw != reg_pw2:
                st.error("비밀번호가 일치하지 않습니다.")
            elif (users_df["user_id"] == reg_id).any():
                st.error("이미 사용 중인 아이디입니다.")
            else:
                new_row = pd.DataFrame(
                    [
                        {
                            "user_id": reg_id,
                            "password": reg_pw,
                            "hospital": reg_hospital,
                            "affiliation": reg_affiliation,
                            "position": reg_position,
                            "name": reg_name,
                            "role": "user",
                            "created_at": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                        }
                    ]
                )
                users_df = pd.concat([users_df, new_row], ignore_index=True)
                save_user_db(users_df)
                st.success("회원가입이 완료되었습니다. 이제 로그인 해 주세요.")
                # 회원가입 후 바로 로그인 탭으로 이동
                st.session_state["auth_active_tab"] = "login"
                st.experimental_rerun()

    # -------------------------------
    # 비밀번호 재설정 탭
    # -------------------------------
    with tab_reset_pw:
        st.markdown("#### 비밀번호 재설정")
        rp_id = st.text_input("아이디", key="rp_id")
        rp_name = st.text_input("이름", key="rp_name")
        rp_hospital = st.text_input("병원명", key="rp_hospital")
        new_pw = st.text_input("새 비밀번호", type="password", key="rp_new_pw")
        new_pw2 = st.text_input("새 비밀번호 확인", type="password", key="rp_new_pw2")

        if st.button("비밀번호 재설정", key="btn_reset_pw"):
            cond = (
                (users_df["user_id"] == rp_id)
                & (users_df["name"] == rp_name)
                & (users_df["hospital"] == rp_hospital)
            )
            row = users_df[cond]
            if row.empty:
                st.error("일치하는 회원 정보를 찾을 수 없습니다.")
            elif new_pw != new_pw2 or not new_pw:
                st.error("새 비밀번호가 일치하지 않거나 비어 있습니다.")
            else:
                users_df.loc[cond, "password"] = new_pw
                save_user_db(users_df)
                st.success("비밀번호가 재설정되었습니다. 새 비밀번호로 로그인해 주세요.")
                # 비밀번호 재설정 후 로그인 탭으로 이동
                st.session_state["auth_active_tab"] = "login"
                st.experimental_rerun()

    # -------------------------------
    # 아이디 찾기 탭
    # -------------------------------
    with tab_find_id:
        st.markdown("#### 아이디 찾기")
        fid_name = st.text_input("이름", key="fid_name")
        fid_hospital = st.text_input("병원명", key="fid_hospital")

        if st.button("아이디 찾기", key="btn_find_id"):
            cond = (
                (users_df["name"] == fid_name)
                & (users_df["hospital"] == fid_hospital)
            )
            rows = users_df[cond]
            if rows.empty:
                st.error("일치하는 회원 정보를 찾을 수 없습니다.")
            else:
                ids = rows["user_id"].dropna().unique().tolist()
                if len(ids) == 1:
                    st.success(f"해당 정보로 등록된 아이디는 **{ids[0]}** 입니다.")
                else:
                    joined_ids = ", ".join(ids)
                    st.success(
                        f"해당 정보로 등록된 아이디는 다음과 같습니다: **{joined_ids}**"
                    )

    # 로그인/회원가입/비밀번호/아이디 찾기만 보여주는 상태
    return None, users_df


def render_admin_view(users_df):
    """Admin 전용: 가입자 목록 화면"""
    st.title("👨‍💼 관리자 화면")
    st.caption("가입한 사용자 목록을 확인할 수 있는 관리자 전용 화면입니다.")

    if users_df is None or users_df.empty:
        st.info("아직 가입된 사용자가 없습니다.")
        return

    display_cols = [
        "user_id",
        "name",
        "hospital",
        "affiliation",
        "position",
        "role",
        "created_at",
    ]
    existing_cols = [c for c in display_cols if c in users_df.columns]

    st.dataframe(users_df[existing_cols], use_container_width=True)
    st.caption(
        "※ 기본 admin 계정(ID: admin / PW: admin1234)은 필요 시 "
        "users_db.csv 에서 수정 가능합니다."
    )

# ------------------------------------------------------------------
# 기존 세션/데이터 로딩 함수
# ------------------------------------------------------------------
def reset_session_state(new_file_id):
    """Resets session state variables when a new file is uploaded."""
    if (
        "last_file_id" not in st.session_state
        or st.session_state["last_file_id"] != new_file_id
    ):
        keys_to_clear = [
            "var_config_df",
            "psm_var_config",
            "psm_done",
            "psm_matched_df",
            "psm_original_w_score",
            "t1_group_col",
            "t1_selected_vals",
            "p_t",
            "p_v",
            "p_c",
        ]
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state["last_file_id"] = new_file_id
        st.rerun()


def load_data(uploaded_file):
    """Loads data from CSV or Excel file."""
    try:
        df = None
        file_ext = uploaded_file.name.split(".")[-1].lower()

        # Reset pointer to beginning
        uploaded_file.seek(0)

        selected_sheet = None

        if file_ext == "csv":
            use_header = st.checkbox(
                "Use first row as header", value=True, key="csv_use_header"
            )
            header_opt = 0 if use_header else None
            df = pd.read_csv(uploaded_file, header=header_opt)

        elif file_ext in ["xlsx", "xls"]:
            xl = pd.ExcelFile(uploaded_file, engine="openpyxl")
            sheet_names = xl.sheet_names

            selected_sheet = sheet_names[0]
            if len(sheet_names) > 1:
                selected_sheet = st.selectbox(
                    "Select Sheet", sheet_names, key="sheet_selector"
                )

            use_header = st.checkbox(
                "Use first row as header", value=True, key="excel_use_header"
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

# ------------------------------------------------------------------
# 메인 함수: 로그인 → 모드 선택(통계/관리자) → 통계 탭
# ------------------------------------------------------------------
def main():
    st.title("📊 Medical Statistics Analysis Tool")
    st.caption("Integrated Statistical Analysis Platform for Medical Research")

    # 1) 로그인 / 회원가입 / 비밀번호 / 아이디 찾기
    user, users_df = render_auth_block()
    if user is None:
        # 로그인 화면을 보여준 상태에서는 아래 분석 화면을 렌더링하지 않음
        return

    # 2) Sidebar: 사용자 정보 + 모드 선택 + (통계 모드일 때) 데이터 업로드
    uploaded_file = None
    mode = "통계 분석"
    with st.sidebar:
        st.header("👤 사용자 정보")
        st.markdown(
            f"**이름:** {user.get('name', '')}  \\n"
            f"**아이디:** {user.get('user_id', '')}  \\n"
            f"**병원:** {user.get('hospital', '')}"
        )
        st.write("---")

        # 관리자라면 모드 선택 가능
        if user.get("role") == "admin":
            mode = st.radio(
                "모드 선택",
                ["통계 분석", "관리자 화면"],
                key="sidebar_mode",
            )
        else:
            mode = "통계 분석"

        st.write("---")

        if mode == "통계 분석":
            st.header("📂 Data Upload & Settings")
            st.info("Upload Excel (.xlsx) or CSV (.csv) file.")
            uploaded_file = st.file_uploader("Select File", type=["xlsx", "csv"])

            st.write("---")
            st.markdown("### ℹ️ Help")
            st.markdown(
                "- **Table 1**: Baseline Characteristics (T-test, Chi-square, etc.)\\n"
                "- **Cox Regression**: Survival Analysis (Kaplan-Meier, Cox PH)\\n"
                "- **Logistic Regression**: Binary Outcome Prediction (ROC Curve)\\n"
                "- **PSM**: Propensity Score Matching"
            )

    # 3) 관리자 화면
    if user.get("role") == "admin" and mode == "관리자 화면":
        render_admin_view(users_df)
        return

    # 4) 통계 분석 화면 (기존 로직)
    if uploaded_file is not None:
        # Load Data
        df, sheet_name = load_data(uploaded_file)

        if df is not None:
            # Generate File ID
            current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if sheet_name:
                current_file_id += f"_{sheet_name}"

            # Reset Session if needed
            reset_session_state(current_file_id)

            st.success("File uploaded successfully!")
            st.dataframe(df.head())

            # Create Tabs
            tab1, tab2, tab3, tab4, tab_methods = st.tabs(
                [
                    "📊 Table 1 (Baseline)",
                    "⏱️ Cox Regression",
                    "💊 Logistic Regression",
                    "⚖️ PSM (Matching)",
                    "📝 Methods Draft",
                ]
            )

            # Render Tabs
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
