# Page Config
import streamlit as st
import hashlib
from pathlib import Path

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
    
    /* Cards/Containers */
    .stDataFrame, .stTable {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #393E46;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #222831;
        color: white;
        border-radius: 8px 8px 0 0;
        padding: 0.4rem 0.8rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ADB5 !important;
        color: white !important;
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

# =============================================================
# Simple File-based Authentication
# =============================================================

USERS_FILE = Path("users.csv")
USER_COLUMNS = [
    "username",
    "password_hash",
    "hospital",
    "department",
    "position",
    "name",
    "is_admin",
]


def _hash_password(password: str) -> str:
    """Return SHA256 hash of the password."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _save_users(df: pd.DataFrame) -> None:
    """Save users dataframe to CSV, keeping only known columns."""
    # Ensure all columns exist
    for col in USER_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col != "is_admin" else False
    df = df[USER_COLUMNS].copy()
    df.to_csv(USERS_FILE, index=False)


def _load_users() -> pd.DataFrame:
    """Load users CSV and ensure at least one admin exists."""
    if USERS_FILE.exists():
        try:
            users = pd.read_csv(USERS_FILE)
        except Exception:
            users = pd.DataFrame(columns=USER_COLUMNS)
    else:
        users = pd.DataFrame(columns=USER_COLUMNS)

    # Ensure mandatory columns
    for col in USER_COLUMNS:
        if col not in users.columns:
            users[col] = "" if col != "is_admin" else False

    # Normalize is_admin to bool
    if "is_admin" in users.columns:
        users["is_admin"] = (
            users["is_admin"].astype(str).str.lower().isin(["true", "1", "yes"])
        )
    else:
        users["is_admin"] = False

    # Ensure at least one admin user
    if not users["is_admin"].any():
        default_admin = {
            "username": "admin",
            "password_hash": _hash_password("admin1234"),
            "hospital": "",
            "department": "",
            "position": "Admin",
            "name": "Administrator",
            "is_admin": True,
        }
        users = pd.concat([users, pd.DataFrame([default_admin])], ignore_index=True)
        _save_users(users)

    return users


def _authenticate(users: pd.DataFrame, username: str, password: str):
    """Return user dict if username/password match, else None."""
    if not username or not password:
        return None
    pw_hash = _hash_password(password)
    mask = (users["username"].astype(str) == username) & (
        users["password_hash"].astype(str) == pw_hash
    )
    if not mask.any():
        return None
    return users.loc[mask].iloc[0].to_dict()


def _register_user(
    users: pd.DataFrame,
    username: str,
    pw1: str,
    pw2: str,
    hospital: str,
    department: str,
    position: str,
    name: str,
):
    """Register a new non-admin user."""
    if not username:
        return False, "아이디를 입력해주세요."
    if not pw1 or not pw2:
        return False, "비밀번호를 입력해주세요."
    if pw1 != pw2:
        return False, "비밀번호 확인이 일치하지 않습니다."

    if username in users["username"].astype(str).tolist():
        return False, "이미 사용 중인 아이디입니다."

    new_user = {
        "username": username,
        "password_hash": _hash_password(pw1),
        "hospital": hospital or "",
        "department": department or "",
        "position": position or "",
        "name": name or "",
        "is_admin": False,
    }

    updated = pd.concat([users, pd.DataFrame([new_user])], ignore_index=True)
    _save_users(updated)
    return True, "회원가입이 완료되었습니다. 이제 로그인해 주세요."


def _reset_password(
    users: pd.DataFrame,
    username: str,
    name: str,
    hospital: str,
    pw1: str,
    pw2: str,
):
    """Simple password reset using (username, name, hospital) verification."""
    if not username:
        return False, "아이디를 입력해주세요."
    if not name:
        return False, "이름을 입력해주세요."
    if not hospital:
        return False, "병원명을 입력해주세요."
    if not pw1 or not pw2:
        return False, "새 비밀번호를 입력해주세요."
    if pw1 != pw2:
        return False, "비밀번호 확인이 일치하지 않습니다."

    mask = (
        users["username"].astype(str) == username
    ) & (
        users["name"].astype(str) == name
    ) & (
        users["hospital"].astype(str) == hospital
    )
    if not mask.any():
        return False, "입력하신 정보와 일치하는 사용자가 없습니다. 관리자에게 문의해주세요."

    users.loc[mask, "password_hash"] = _hash_password(pw1)
    _save_users(users)
    return True, "비밀번호가 변경되었습니다. 새 비밀번호로 로그인해 주세요."


def require_login():
    """Render login / signup / reset UI and return logged-in user dict if any.

    - 로그인 상태라면 바로 사용자 정보를 반환
    - 로그인하지 않았다면 로그인/회원가입/비밀번호 재설정 탭을 보여주고 None 반환
    """
    # 이미 로그인 되어 있으면 곧바로 반환
    user = st.session_state.get("auth_user")
    if user is not None:
        return user

    users = _load_users()

    st.title("🔐 로그인")
    st.caption("의료 통계 분석 도구 사용을 위해 먼저 로그인 해주세요.")

    tab_login, tab_signup, tab_reset = st.tabs(["로그인", "회원가입", "비밀번호 찾기 / 재설정"])

    # 로그인 탭
    with tab_login:
        with st.form("login_form"):
            username = st.text_input("아이디", key="login_username")
            password = st.text_input("비밀번호", type="password", key="login_password")
            submitted = st.form_submit_button("로그인")
        if submitted:
            user = _authenticate(users, username, password)
            if user:
                st.session_state["auth_user"] = user
                st.success("로그인되었습니다.")
                st.rerun()
            else:
                st.error("아이디 또는 비밀번호가 올바르지 않습니다.")

    # 회원가입 탭
    with tab_signup:
        with st.form("signup_form"):
            su_username = st.text_input("아이디", key="signup_username")
            su_pw1 = st.text_input("비밀번호", type="password", key="signup_pw1")
            su_pw2 = st.text_input("비밀번호 확인", type="password", key="signup_pw2")
            su_hosp = st.text_input("병원명", key="signup_hospital")
            su_dept = st.text_input("소속", key="signup_department")
            su_pos = st.text_input("직책 (예: 교수)", key="signup_position")
            su_name = st.text_input("이름", key="signup_name")
            su_submit = st.form_submit_button("회원가입")
        if su_submit:
            success, msg = _register_user(
                users,
                su_username,
                su_pw1,
                su_pw2,
                su_hosp,
                su_dept,
                su_pos,
                su_name,
            )
            if success:
                st.success(msg)
            else:
                st.error(msg)

    # 비밀번호 재설정 탭
    with tab_reset:
        with st.form("reset_form"):
            r_username = st.text_input("아이디", key="reset_username")
            r_name = st.text_input("이름", key="reset_name")
            r_hosp = st.text_input("병원명", key="reset_hospital")
            r_pw1 = st.text_input("새 비밀번호", type="password", key="reset_pw1")
            r_pw2 = st.text_input("새 비밀번호 확인", type="password", key="reset_pw2")
            r_submit = st.form_submit_button("비밀번호 재설정")
        if r_submit:
            success, msg = _reset_password(
                users,
                r_username,
                r_name,
                r_hosp,
                r_pw1,
                r_pw2,
            )
            if success:
                st.success(msg)
            else:
                st.error(msg)

    return None


# =============================================================
# 기존 데이터 업로드 및 분석 함수들
# =============================================================

def reset_session_state(new_file_id):
    """Resets session state variables when a new file is uploaded."""
    if "last_file_id" not in st.session_state or st.session_state["last_file_id"] != new_file_id:
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

        selected_sheet = None

        if file_ext == "csv":
            use_header = st.checkbox("Use first row as header", value=True, key="csv_use_header")
            header_opt = 0 if use_header else None
            df = pd.read_csv(uploaded_file, header=header_opt)

        elif file_ext in ["xlsx", "xls"]:
            xl = pd.ExcelFile(uploaded_file, engine="openpyxl")
            sheet_names = xl.sheet_names

            selected_sheet = sheet_names[0]
            if len(sheet_names) > 1:
                selected_sheet = st.selectbox("Select Sheet", sheet_names, key="sheet_selector")

            use_header = st.checkbox("Use first row as header", value=True, key="excel_use_header")
            header_opt = 0 if use_header else None
            df = xl.parse(selected_sheet, header=header_opt)

        else:
            st.error("Unsupported file format. Please upload CSV or XLSX.")
            return None, None

        return df, selected_sheet

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, None


def main():
    # 1) 인증 먼저 수행
    user = require_login()
    if user is None:
        # 로그인 화면만 보여주고 종료
        return

    # 2) 로그인 이후 메인 타이틀 / 사용자 정보 표시
    st.title("📊 Medical Statistics Analysis Tool")
    st.caption("Integrated Statistical Analysis Platform for Medical Research")

    # 사용자 정보 및 로그아웃 버튼은 사이드바에 표시
    with st.sidebar:
        st.markdown("### 👤 사용자 정보")
        st.write(f"**이름:** {user.get('name', '')}")
        st.write(f"**아이디:** {user.get('username', '')}")
        if user.get("hospital") or user.get("department"):
            st.write(f"**소속:** {user.get('hospital', '')} {user.get('department', '')}")
        if user.get("position"):
            st.write(f"**직책:** {user.get('position', '')}")

        if st.button("로그아웃", key="logout_btn"):
            st.session_state.pop("auth_user", None)
            st.rerun()

        st.write("---")
        st.header("📂 Data Upload & Settings")
        st.info("Upload Excel (.xlsx) or CSV (.csv) file.")
        uploaded_file = st.file_uploader("Select File", type=["xlsx", "csv"])

        st.write("---")
        st.markdown("### ℹ️ Help")
        st.markdown(
            """
        - **Table 1**: Baseline Characteristics (T-test, Chi-square, etc.)
        - **Cox Regression**: Survival Analysis (Kaplan-Meier, Cox PH)
        - **Logistic Regression**: Binary Outcome Prediction (ROC Curve)
        - **PSM**: Propensity Score Matching
        """
        )

    # 3) 관리자 전용: 가입자 목록 보기
    if user.get("is_admin"):
        st.markdown("---")
        st.subheader("👑 관리자 패널: 가입자 목록")
        admin_users = _load_users().copy()
        if not admin_users.empty:
            admin_view = admin_users.drop(columns=["password_hash"], errors="ignore")
            st.dataframe(admin_view, use_container_width=True)
        else:
            st.info("아직 가입된 사용자가 없습니다.")

    # 4) 데이터 업로드 후 통계 분석 탭 표시
    if uploaded_file is not None:
        df, sheet_name = load_data(uploaded_file)

        if df is not None:
            current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if sheet_name:
                current_file_id += f"_{sheet_name}"

            reset_session_state(current_file_id)

            st.success("File uploaded successfully!")
            st.dataframe(df.head())

            # 탭 구성 (현재는 관리자/일반 사용자 모두 동일한 탭 구조)
            tab1, tab2, tab3, tab4, tab_methods = st.tabs(
                [
                    "📊 Table 1 (Baseline)",
                    "⏱️ Cox Regression",
                    "💊 Logistic Regression",
                    "⚖️ PSM (Matching)",
                    "📝 Methods Draft",
                ]
            )

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
