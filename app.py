# Page Config
import streamlit as st

st.set_page_config(
    page_title="Medical Statistics Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown(
    """
    <style>
    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background */
    body {
        background-color: #0F172A;
        color: #E5E7EB;
    }

    /* Main container */
    .main {
        background-color: #111827;
        color: #E5E7EB;
    }

    /* Cards */
    .stCard {
        background-color: #111827;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #1F2937;
    }

    /* Dataframes */
    .stDataFrame, .stTable {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #393E46;
    }

    /* Alerts */
    .stAlert {
        border-radius: 8px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1F2937;
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
        box-shadow: 0 4px 6px rgba(0, 173, 181, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Imports
try:
    import pandas as pd
    import numpy as np
    import io
    import os
    import hashlib
    from datetime import datetime

    from modules.tab1_table1 import render_tab1
    from modules.tab2_cox import render_tab2
    from modules.tab3_logistic import render_tab3
    from modules.tab4_psm import render_tab4
    from modules.tab5_methods import render_tab5
except ImportError as e:
    st.error(f"Module Import Error: {e}")
    st.stop()

# ---------------------------------------------------------------------
# Simple user management (CSV-based)
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DB_PATH = os.path.join(BASE_DIR, "users_db.csv")

USER_COLUMNS = [
    "user_id",
    "password_hash",
    "hospital",
    "department",
    "position",
    "full_name",
    "is_admin",
    "created_at",
]

DEFAULT_ADMIN_ID = "admin"
DEFAULT_ADMIN_PASSWORD = "admin1234!"  # 개발용 기본 관리자 계정


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def load_user_db() -> pd.DataFrame:
    if os.path.exists(USER_DB_PATH):
        try:
            df = pd.read_csv(USER_DB_PATH)
        except Exception:
            df = pd.DataFrame(columns=USER_COLUMNS)
    else:
        # 최초 실행 시 기본 관리자 계정 생성
        df = pd.DataFrame(columns=USER_COLUMNS)
        admin_row = {
            "user_id": DEFAULT_ADMIN_ID,
            "password_hash": _hash_password(DEFAULT_ADMIN_PASSWORD),
            "hospital": "",
            "department": "",
            "position": "Admin",
            "full_name": "Administrator",
            "is_admin": True,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        df = pd.concat([df, pd.DataFrame([admin_row])], ignore_index=True)
        df.to_csv(USER_DB_PATH, index=False)
    # ensure columns exist
    for c in USER_COLUMNS:
        if c not in df.columns:
            df[c] = "" if c != "is_admin" else False
    df["is_admin"] = df["is_admin"].astype(bool)
    return df[USER_COLUMNS]


def save_user_db(df: pd.DataFrame) -> None:
    df = df[USER_COLUMNS]
    df.to_csv(USER_DB_PATH, index=False)


def register_user(
    user_id: str,
    password: str,
    hospital: str,
    department: str,
    position: str,
    full_name: str,
) -> tuple[bool, str]:
    user_id = user_id.strip()
    full_name = full_name.strip()

    if not user_id or not password or not full_name:
        return False, "아이디, 비밀번호, 이름은 필수입니다."

    df = load_user_db()
    if (df["user_id"] == user_id).any():
        return False, "이미 사용 중인 아이디입니다."

    new_row = {
        "user_id": user_id,
        "password_hash": _hash_password(password),
        "hospital": hospital.strip(),
        "department": department.strip(),
        "position": position.strip(),
        "full_name": full_name,
        "is_admin": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_user_db(df)
    return True, "회원가입이 완료되었습니다. 이제 로그인 해 주세요."


def authenticate_user(
    user_id: str, password: str
) -> tuple[bool, str | None, dict | None]:
    df = load_user_db()
    row = df[df["user_id"] == user_id]
    if row.empty:
        return False, "존재하지 않는 아이디입니다.", None

    row = row.iloc[0]
    if row["password_hash"] != _hash_password(password):
        return False, "비밀번호가 올바르지 않습니다.", None

    profile = {
        "user_id": row["user_id"],
        "full_name": row["full_name"],
        "hospital": row["hospital"],
        "department": row["department"],
        "position": row["position"],
        "is_admin": bool(row["is_admin"]),
    }
    return True, None, profile


def reset_password(
    user_id: str,
    full_name: str,
    hospital: str,
    new_password: str,
) -> tuple[bool, str]:
    full_name = full_name.strip()
    hospital = hospital.strip()
    df = load_user_db()
    mask = (
        (df["user_id"] == user_id.strip())
        & (df["full_name"].astype(str) == full_name)
        & (df["hospital"].astype(str) == hospital)
    )
    if not mask.any():
        return False, "입력하신 정보와 일치하는 계정을 찾을 수 없습니다."

    df.loc[mask, "password_hash"] = _hash_password(new_password)
    save_user_db(df)
    return True, "비밀번호가 변경되었습니다. 새 비밀번호로 로그인 해 주세요."


def find_user_ids(full_name: str, hospital: str) -> list[str]:
    full_name = full_name.strip()
    hospital = hospital.strip()
    df = load_user_db()
    mask = (df["full_name"].astype(str) == full_name) & (
        df["hospital"].astype(str) == hospital
    )
    ids = df.loc[mask, "user_id"].dropna().astype(str).unique().tolist()
    return ids


def show_login_page():
    st.title("📊 Medical Statistics Analysis Tool")
    st.caption("로그인 후 자동 통계 프로그램을 사용할 수 있습니다.")

    # 현재 선택된 메뉴 상태
    if "auth_menu" not in st.session_state:
        st.session_state["auth_menu"] = "로그인"

    # 상단 메뉴 버튼 (로그인 / 회원가입 / 비밀번호 재설정 / 아이디 찾기)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("로그인", use_container_width=True):
            st.session_state["auth_menu"] = "로그인"
            st.rerun()
    with col2:
        if st.button("회원가입", use_container_width=True):
            st.session_state["auth_menu"] = "회원가입"
            st.rerun()
    with col3:
        if st.button("비밀번호 재설정", use_container_width=True):
            st.session_state["auth_menu"] = "비밀번호 재설정"
            st.rerun()
    with col4:
        if st.button("아이디 찾기", use_container_width=True):
            st.session_state["auth_menu"] = "아이디 찾기"
            st.rerun()

    st.write("---")

    menu = st.session_state.get("auth_menu", "로그인")

    # ------------------- 로그인 -------------------
    if menu == "로그인":
        with st.form("login_form"):
            user_id = st.text_input("아이디", key="login_user_id")
            password = st.text_input("비밀번호", type="password", key="login_password")
            submitted = st.form_submit_button("로그인")

        if submitted:
            ok, msg, profile = authenticate_user(user_id, password)
            if not ok:
                st.error(msg)
            else:
                st.success("로그인되었습니다.")
                st.session_state["auth_logged_in"] = True
                st.session_state["auth_user_profile"] = profile
                st.rerun()

        st.markdown(
            "기본 관리자 계정: `admin` / `admin1234!`  \\n"
            "(최초 접속 후 관리자 계정은 테스트용으로만 사용하시고, "
            "별도 계정을 만들어 사용하시는 것을 권장합니다.)"
        )

    # ------------------- 회원가입 -------------------
    elif menu == "회원가입":
        with st.form("signup_form"):
            new_id = st.text_input("아이디", key="signup_user_id")
            pw1 = st.text_input("비밀번호", type="password", key="signup_pw1")
            pw2 = st.text_input("비밀번호 확인", type="password", key="signup_pw2")
            hospital = st.text_input("병원명", key="signup_hospital")
            dept = st.text_input("소속(예: 대장항문외과)", key="signup_dept")
            position = st.text_input("직책(예: 교수)", key="signup_position")
            full_name = st.text_input("이름", key="signup_full_name")
            submitted_signup = st.form_submit_button("회원가입")

        if submitted_signup:
            if pw1 != pw2:
                st.error("비밀번호와 비밀번호 확인이 일치하지 않습니다.")
            else:
                ok, msg = register_user(
                    user_id=new_id,
                    password=pw1,
                    hospital=hospital,
                    department=dept,
                    position=position,
                    full_name=full_name,
                )
                if ok:
                    st.success(msg)
                    # 회원가입 성공 후 바로 로그인 화면으로 이동
                    st.session_state["auth_menu"] = "로그인"
                    st.rerun()
                else:
                    st.error(msg)

    # ------------------- 비밀번호 재설정 -------------------
    elif menu == "비밀번호 재설정":
        with st.form("reset_pw_form"):
            r_user_id = st.text_input("아이디", key="reset_user_id")
            r_name = st.text_input("이름", key="reset_full_name")
            r_hosp = st.text_input("병원명", key="reset_hospital")
            r_pw1 = st.text_input("새 비밀번호", type="password", key="reset_pw1")
            r_pw2 = st.text_input("새 비밀번호 확인", type="password", key="reset_pw2")
            submitted_reset = st.form_submit_button("비밀번호 재설정")

        if submitted_reset:
            if not r_pw1 or not r_pw2:
                st.error("새 비밀번호를 입력해주세요.")
            elif r_pw1 != r_pw2:
                st.error("비밀번호와 비밀번호 확인이 일치하지 않습니다.")
            else:
                ok, msg = reset_password(
                    user_id=r_user_id,
                    full_name=r_name,
                    hospital=r_hosp,
                    new_password=r_pw1,
                )
                if ok:
                    st.success(msg)
                    # 비밀번호 재설정 후 바로 로그인 화면으로 이동
                    st.session_state["auth_menu"] = "로그인"
                    st.rerun()
                else:
                    st.error(msg)

    # ------------------- 아이디 찾기 -------------------
    elif menu == "아이디 찾기":
        with st.form("find_id_form"):
            f_name = st.text_input("이름", key="find_full_name")
            f_hosp = st.text_input("병원명", key="find_hospital")
            submitted_find = st.form_submit_button("아이디 찾기")

        if submitted_find:
            if not f_name or not f_hosp:
                st.error("이름과 병원명을 모두 입력해주세요.")
            else:
                ids = find_user_ids(f_name, f_hosp)
                if not ids:
                    st.warning("입력하신 정보와 일치하는 아이디를 찾을 수 없습니다.")
                elif len(ids) == 1:
                    st.success(f"찾은 아이디: **{ids[0]}**")
                else:
                    st.success(
                        "여러 개의 계정이 있습니다: "
                        + ", ".join(f"**{uid}**" for uid in ids)
                    )


def admin_panel(current_profile: dict):
    st.title("👨‍⚕️ Admin Panel")
    st.caption("가입한 사용자 목록 및 관리자 계정 설정 화면입니다.")

    df_users = load_user_db()
    if df_users.empty:
        st.info("아직 가입한 사용자가 없습니다.")
    else:
        st.subheader("가입자 목록")
        st.dataframe(
            df_users[
                [
                    "user_id",
                    "full_name",
                    "hospital",
                    "department",
                    "position",
                    "is_admin",
                    "created_at",
                ]
            ],
            use_container_width=True,
        )

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
            df_users.to_excel(w, index=False)
        st.download_button(
            "📥 Download user list",
            data=buf.getvalue(),
            file_name="user_list.xlsx",
        )

    st.write("---")
    st.subheader("관리자 계정 변경")

    with st.form("admin_change_form"):
        st.caption("현재 로그인한 관리자 계정의 아이디/비밀번호를 변경합니다.")
        current_id = st.text_input(
            "현재 아이디",
            value=current_profile.get("user_id", ""),
            disabled=True,
        )
        current_pw = st.text_input("현재 비밀번호", type="password", key="admin_current_pw")
        new_id = st.text_input("새 아이디 (변경하지 않을 경우 비워두기 가능)", key="admin_new_id")
        new_pw1 = st.text_input("새 비밀번호", type="password", key="admin_new_pw1")
        new_pw2 = st.text_input("새 비밀번호 확인", type="password", key="admin_new_pw2")
        submitted_admin = st.form_submit_button("관리자 계정 변경")

    if submitted_admin:
        df = load_user_db()
        mask = (df["user_id"] == current_profile.get("user_id")) & (df["is_admin"])
        if not mask.any():
            st.error("현재 관리자 계정을 찾을 수 없습니다.")
        elif df.loc[mask, "password_hash"].iloc[0] != _hash_password(current_pw):
            st.error("현재 비밀번호가 올바르지 않습니다.")
        elif new_pw1 != new_pw2 or not new_pw1:
            st.error("새 비밀번호가 비어 있거나 일치하지 않습니다.")
        else:
            final_id = new_id.strip() or current_profile.get("user_id")
            # 아이디 중복 체크 (자기 자신 제외)
            if (df["user_id"] == final_id).any() and final_id != current_profile.get("user_id"):
                st.error("이미 사용 중인 아이디입니다.")
            else:
                df.loc[mask, "user_id"] = final_id
                df.loc[mask, "password_hash"] = _hash_password(new_pw1)
                save_user_db(df)

                # 세션 프로필도 업데이트
                updated_profile = dict(current_profile)
                updated_profile["user_id"] = final_id
                st.session_state["auth_user_profile"] = updated_profile

                st.success("관리자 계정 정보가 변경되었습니다. 변경된 정보로 계속 사용하실 수 있습니다.")

# ---------------------------------------------------------------------
# Original data loading / analysis app
# ---------------------------------------------------------------------


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


def run_analysis_app():
    st.title("📊 Medical Statistics Analysis Tool")
    st.caption("Integrated Statistical Analysis Platform for Medical Research")

    # Sidebar
    with st.sidebar:
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


def main():
    # 초기 auth state 설정
    if "auth_logged_in" not in st.session_state:
        st.session_state["auth_logged_in"] = False

    if not st.session_state["auth_logged_in"]:
        # 로그인 화면만 보여줌
        show_login_page()
        return

    # 로그인된 상태: 상단에 사용자 정보와 로그아웃 버튼 표시
    profile = st.session_state.get("auth_user_profile", {}) or {}
    user_name = profile.get("full_name") or profile.get("user_id") or "User"
    is_admin = bool(profile.get("is_admin", False))

    top_col1, top_col2 = st.columns([0.8, 0.2])
    with top_col1:
        st.markdown(f"👤 **{user_name}** 님이 로그인 중입니다.")
    with top_col2:
        if st.button("로그아웃", key="logout_btn"):
            # auth 관련 키만 정리
            for k in ["auth_logged_in", "auth_user_profile", "auth_menu"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    # Admin 모드 선택 (관리자만)
    mode = "Analysis"
    if is_admin:
        mode = st.sidebar.radio(
            "Mode",
            options=["Analysis", "Admin Panel"],
            index=0,
            key="app_mode_radio",
        )

    if mode == "Admin Panel" and is_admin:
        admin_panel(profile)
    else:
        run_analysis_app()


if __name__ == "__main__":
    main()
