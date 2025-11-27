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
    
    /* Buttons - smaller size */
    .stButton > button {
        background-color: #00ADB5;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.3rem 0.7rem;
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #007A80;
        box-shadow: 0 3px 5px rgba(0,0,0,0.15);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.4rem 0.8rem;
        border-radius: 8px 8px 0 0;
        background-color: #222831;
        color: #EEEEEE;
        font-size: 0.9rem;
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
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #222831;
        border-right: 1px solid #393E46;
    }
    </style>
    """, unsafe_allow_html=True)

# Imports for analysis modules
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
# Authentication (file-based, hashed password)
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
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _save_users(df: pd.DataFrame) -> None:
    # ensure all columns exist
    for col in USER_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col != "is_admin" else False
    df = df[USER_COLUMNS].copy()
    df.to_csv(USERS_FILE, index=False)


def _load_users() -> pd.DataFrame:
    if USERS_FILE.exists():
        try:
            users = pd.read_csv(USERS_FILE)
        except Exception:
            users = pd.DataFrame(columns=USER_COLUMNS)
    else:
        users = pd.DataFrame(columns=USER_COLUMNS)

    # normalize columns
    for col in USER_COLUMNS:
        if col not in users.columns:
            users[col] = "" if col != "is_admin" else False

    # normalize is_admin to bool
    users["is_admin"] = (
        users["is_admin"].astype(str).str.lower().isin(["true", "1", "yes"])
    )

    # --- 기본 관리자 계정 설정 / 마이그레이션 ---
    default_admin_username = "admin"
    old_default_pw_hash = _hash_password("admin1234")          # 이전 기본값
    new_default_pw_hash = _hash_password("asdqwe123!@#")       # 새 기본값

    admin_mask = users["username"].astype(str) == default_admin_username

    if not admin_mask.any():
        # 관리자 계정이 전혀 없으면 새로 생성
        default_admin = {
            "username": default_admin_username,
            "password_hash": new_default_pw_hash,
            "hospital": "",
            "department": "",
            "position": "Admin",
            "name": "Administrator",
            "is_admin": True,
        }
        users = pd.concat([users, pd.DataFrame([default_admin])], ignore_index=True)
        _save_users(users)
    else:
        # admin 행이 있는데, 예전 기본 해시를 쓰고 있으면 새 비번으로 교체
        if (users.loc[admin_mask, "password_hash"] == old_default_pw_hash).any():
            users.loc[admin_mask, "password_hash"] = new_default_pw_hash
            _save_users(users)

    return users


def _authenticate(users: pd.DataFrame, username: str, password: str):
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


def _find_ids(users: pd.DataFrame, name: str, hospital: str):
    if not name or not hospital:
        return []
    mask = (
        users["name"].astype(str) == name
    ) & (
        users["hospital"].astype(str) == hospital
    )
    return users.loc[mask, "username"].astype(str).tolist()


def require_login():
    """로그인/회원가입/비번·아이디 찾기 화면을 띄우고, 로그인되면 user dict 반환."""
    user = st.session_state.get("auth_user")
    if user is not None:
        return user

    users = _load_users()

    st.title("🔐 로그인")
    st.caption("의료 통계 분석 도구 사용을 위해 먼저 로그인 해주세요.")

    tab_login, tab_signup, tab_reset = st.tabs(
        ["로그인", "회원가입", "비밀번호 / 아이디 찾기"]
    )

    # 로그인
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

    # 회원가입
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
                # 회원가입 후 바로 로그인 화면으로
                st.rerun()
            else:
                st.error(msg)

    # 비밀번호 / 아이디 찾기
    with tab_reset:
        col_id, col_pw = st.columns(2)

        with col_id:
            st.markdown("#### 아이디 찾기")
            with st.form("find_id_form"):
                fid_name = st.text_input("이름", key="findid_name")
                fid_hosp = st.text_input("병원명", key="findid_hospital")
                fid_submit = st.form_submit_button("아이디 찾기")
            if fid_submit:
                ids = _find_ids(users, fid_name, fid_hosp)
                if ids:
                    st.success("입력하신 정보로 등록된 아이디:")
                    for u in ids:
                        st.write(f"- **{u}**")
                else:
                    st.error("해당하는 아이디를 찾을 수 없습니다.")

        with col_pw:
            st.markdown("#### 비밀번호 재설정")
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
                    # 재설정 후 로그인 화면으로
                    st.rerun()
                else:
                    st.error(msg)

    return None


def render_admin_panel(current_user: dict):
    """관리자 화면: 가입자 목록 + 관리자 계정(ID/비번) 변경."""
    st.title("👑 관리자 화면")
    st.caption("가입된 사용자 목록 및 관리자 계정 설정을 확인/변경할 수 있습니다.")

    users = _load_users()

    # 관리자 계정 설정
    st.markdown("### 🔧 관리자 계정 설정")
    admin_mask = users["username"].astype(str) == str(current_user.get("username"))
    if not admin_mask.any():
        st.warning("현재 계정 정보가 사용자 목록에서 발견되지 않았습니다.")
    else:
        admin_row = users.loc[admin_mask].iloc[0]
        with st.form("admin_account_form"):
            new_username = st.text_input(
                "관리자 아이디",
                value=str(admin_row["username"]),
                key="admin_username",
            )
            new_pw1 = st.text_input(
                "새 비밀번호 (변경하지 않으면 비워두세요)",
                type="password",
                key="admin_pw1",
            )
            new_pw2 = st.text_input(
                "새 비밀번호 확인",
                type="password",
                key="admin_pw2",
            )
            submitted = st.form_submit_button("관리자 계정 저장")

        if submitted:
            # 아이디 체크
            if not new_username:
                st.error("아이디는 비워둘 수 없습니다.")
            else:
                username_conflict = users[
                    (users["username"].astype(str) == new_username)
                    & (~admin_mask)
                ]
                if not username_conflict.empty:
                    st.error("이미 사용 중인 아이디입니다.")
                else:
                    # 아이디 업데이트
                    users.loc[admin_mask, "username"] = new_username
                    # 비밀번호 변경 요청이 있으면 처리
                    if new_pw1 or new_pw2:
                        if new_pw1 != new_pw2:
                            st.error("새 비밀번호가 서로 일치하지 않습니다.")
                        else:
                            users.loc[admin_mask, "password_hash"] = _hash_password(
                                new_pw1
                            )
                    _save_users(users)
                    # 세션 정보 갱신
                    updated_user = users.loc[admin_mask].iloc[0].to_dict()
                    st.session_state["auth_user"] = updated_user
                    st.success("관리자 계정 정보가 업데이트되었습니다.")
                    st.rerun()

    st.markdown("### 👥 가입자 목록")
    if users.empty:
        st.info("아직 가입된 사용자가 없습니다.")
    else:
        display_cols = [
            "username",
            "name",
            "hospital",
            "department",
            "position",
            "is_admin",
        ]
        existing_cols = [c for c in display_cols if c in users.columns]
        st.dataframe(users[existing_cols], use_container_width=True)


# =============================================================
# Data loading & session reset
# =============================================================

def reset_session_state(new_file_id: str):
    """파일이 바뀔 때 통계 관련 세션 초기화."""
    if st.session_state.get("last_file_id") != new_file_id:
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
            st.session_state.pop(k, None)
        st.session_state["last_file_id"] = new_file_id
        st.rerun()


def load_data(uploaded_file):
    """Loads data from CSV or Excel file."""
    try:
        df = None
        file_ext = uploaded_file.name.split(".")[-1].lower()
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


# =============================================================
# Main app
# =============================================================

def main():
    # 1) 로그인 처리
    user = require_login()
    if user is None:
        return

    # 2) Sidebar: 사용자 정보 / 모드 선택 / 업로드
    with st.sidebar:
        st.markdown("### 👤 사용자 정보")
        st.write(f"**이름:** {user.get('name', '')}")
        st.write(f"**아이디:** {user.get('username', '')}")
        if user.get("hospital") or user.get("department"):
            st.write(
                f"**소속:** {user.get('hospital', '')} {user.get('department', '')}"
            )
        if user.get("position"):
            st.write(f"**직책:** {user.get('position', '')}")

        if st.button("로그아웃", key="logout_btn"):
            st.session_state.pop("auth_user", None)
            st.rerun()

        st.write("---")
        # 관리자라면 통계/관리자 모드 선택
        if user.get("is_admin"):
            mode = st.radio("모드 선택", ["통계 분석", "관리자 화면"], key="mode_radio")
        else:
            mode = "통계 분석"

        uploaded_file = None
        if mode == "통계 분석":
            st.header("📂 Data Upload & Settings")
            st.info("Upload Excel (.xlsx) or CSV (.csv) file.")
            uploaded_file = st.file_uploader("Select File", type=["xlsx", "csv"])

            st.write("---")
            st.markdown("### ℹ️ Help")
            st.markdown(
                "- **Table 1**: Baseline Characteristics (T-test, Chi-square, etc.)\n"
                "- **Cox Regression**: Survival Analysis (Kaplan-Meier, Cox PH)\n"
                "- **Logistic Regression**: Binary Outcome Prediction (ROC Curve)\n"
                "- **PSM**: Propensity Score Matching"
            )

    # 3) 관리자 화면
    if user.get("is_admin") and mode == "관리자 화면":
        render_admin_panel(user)
        return

    # 4) 통계 분석 화면
    st.title("📊 Medical Statistics Analysis Tool")
    st.caption("Integrated Statistical Analysis Platform for Medical Research")

    if uploaded_file is not None:
        df, sheet_name = load_data(uploaded_file)
        if df is not None:
            current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if sheet_name:
                current_file_id += f"_{sheet_name}"
            reset_session_state(current_file_id)

            st.success("File uploaded successfully!")
            st.dataframe(df.head())

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
