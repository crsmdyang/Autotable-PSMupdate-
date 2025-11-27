import streamlit as st
import os
from datetime import datetime

st.set_page_config(
    page_title="Medical Statistics Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# 기본 스타일 (버튼 작게)
# =========================
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #00ADB5 !important;
        font-weight: 700;
    }
    /* 버튼 크기 조정 */
    .stButton > button {
        background-color: #00ADB5;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.3rem 0.8rem;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #007A80;
        box-shadow: 0 3px 5px rgba(0,0,0,0.15);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.4rem 0.9rem;
        border-radius: 8px 8px 0 0;
        background-color: #393E46;
        color: #EEEEEE;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ADB5 !important;
        color: white !important;
    }
    .stDataFrame, .stTable {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #393E46;
    }
    [data-testid="stSidebar"] {
        background-color: #222831;
        border-right: 1px solid #393E46;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# 모듈 로딩
# =========================
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

# =========================
# 사용자 DB (CSV 기반)
# =========================
USER_DB_PATH = "users_db.csv"
USER_DB_COLUMNS = [
    "user_id",
    "password",       # 평문 비밀번호 (배포용이면 해시 권장)
    "hospital",
    "affiliation",
    "position",
    "name",
    "role",           # "admin" or "user"
    "created_at",
    "last_login_at",  # 마지막 로그인 시각
]

DEFAULT_ADMIN_ID = "admin"
DEFAULT_ADMIN_PASSWORD = "asdqwe123!@#"    # ✅ 요구하신 기본 관리자 PW
OLD_DEFAULT_ADMIN_PASSWORD = "admin1234"   # 이전 버전 기본 PW (마이그레이션용)


def _init_user_db():
    """users_db.csv가 없으면 기본 admin 계정을 만든다."""
    if not os.path.exists(USER_DB_PATH):
        df = pd.DataFrame(columns=USER_DB_COLUMNS)
        df.loc[len(df)] = [
            DEFAULT_ADMIN_ID,           # user_id
            DEFAULT_ADMIN_PASSWORD,     # password
            "Admin Hospital",           # hospital
            "Admin",                    # affiliation
            "관리자",                    # position
            "Administrator",            # name
            "admin",                    # role
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # created_at
            "",                         # last_login_at
        ]
        df.to_csv(USER_DB_PATH, index=False, encoding="utf-8")


def save_user_db(df: pd.DataFrame) -> None:
    """사용자 DB를 CSV로 저장."""
    for col in USER_DB_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[USER_DB_COLUMNS]
    df.to_csv(USER_DB_PATH, index=False, encoding="utf-8")


def load_user_db() -> pd.DataFrame:
    """CSV에서 사용자 DB를 읽어온다 (없으면 생성 + 예전 admin PW를 새 PW로 교체)."""
    _init_user_db()
    try:
        df = pd.read_csv(USER_DB_PATH, dtype=str, encoding="utf-8")
    except Exception:
        df = pd.DataFrame(columns=USER_DB_COLUMNS)

    for col in USER_DB_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # 예전 기본 PW(admin1234) 사용 시, 새 PW로 마이그레이션
    mask_old_admin = (
        (df["user_id"] == DEFAULT_ADMIN_ID)
        & (df["role"] == "admin")
        & (df["password"] == OLD_DEFAULT_ADMIN_PASSWORD)
    )
    if mask_old_admin.any():
        df.loc[mask_old_admin, "password"] = DEFAULT_ADMIN_PASSWORD
        save_user_db(df)

    return df[USER_DB_COLUMNS]


# =========================
# Auth UI (버튼으로 화면 전환)
# =========================
def render_auth_block():
    """
    로그인 / 회원가입 / 비밀번호 재설정 / 아이디 찾기

    Returns
    -------
    user : dict or None
    users_df : pd.DataFrame
    """
    users_df = load_user_db()
    current_user = st.session_state.get("current_user")

    # 이미 로그인된 경우
    if current_user is not None:
        return current_user, users_df

    st.markdown("### 🔐 로그인 / 회원 관리")

    # 메뉴 상태
    if "auth_menu" not in st.session_state:
        st.session_state["auth_menu"] = "로그인"

    # 상단 버튼 네 개로 메뉴 전환
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("로그인", use_container_width=True, key="btn_menu_login"):
            st.session_state["auth_menu"] = "로그인"
            st.rerun()
    with col2:
        if st.button("회원가입", use_container_width=True, key="btn_menu_signup"):
            st.session_state["auth_menu"] = "회원가입"
            st.rerun()
    with col3:
        if st.button("비밀번호 재설정", use_container_width=True, key="btn_menu_reset_pw"):
            st.session_state["auth_menu"] = "비밀번호 재설정"
            st.rerun()
    with col4:
        if st.button("아이디 찾기", use_container_width=True, key="btn_menu_find_id"):
            st.session_state["auth_menu"] = "아이디 찾기"
            st.rerun()

    st.write("---")
    menu = st.session_state.get("auth_menu", "로그인")

    # ------------------ 로그인 ------------------
    if menu == "로그인":
        # 회원가입/비번 재설정 후 안내 메시지
        if st.session_state.pop("signup_done", False):
            st.success("회원가입이 완료되었습니다. 이제 로그인해 주세요.")
        if st.session_state.pop("pw_reset_done", False):
            st.success("비밀번호가 재설정되었습니다. 새 비밀번호로 로그인해 주세요.")

        with st.form("login_form"):
            login_id = st.text_input("아이디", key="login_id")
            login_pw = st.text_input("비밀번호", type="password", key="login_pw")
            submitted = st.form_submit_button("로그인")

        if submitted:
            row = users_df[users_df["user_id"] == login_id]
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if row.empty:
                # admin 계정이 DB에 없는데 기본 admin으로 로그인 시도하는 경우 → 새로 생성
                if login_id == DEFAULT_ADMIN_ID and login_pw == DEFAULT_ADMIN_PASSWORD:
                    new_admin = {
                        "user_id": DEFAULT_ADMIN_ID,
                        "password": DEFAULT_ADMIN_PASSWORD,
                        "hospital": "Admin Hospital",
                        "affiliation": "Admin",
                        "position": "관리자",
                        "name": "Administrator",
                        "role": "admin",
                        "created_at": now_str,
                        "last_login_at": now_str,
                    }
                    users_df = pd.concat(
                        [users_df, pd.DataFrame([new_admin])], ignore_index=True
                    )
                    save_user_db(users_df)
                    st.session_state["current_user"] = new_admin
                    st.rerun()
                else:
                    st.error("존재하지 않는 아이디입니다.")
            else:
                stored_pw = str(row.iloc[0]["password"])
                idx = row.index[0]

                # 일반 로그인 (PW 일치)
                if stored_pw == login_pw:
                    users_df.loc[idx, "last_login_at"] = now_str
                    save_user_db(users_df)
                    user = users_df.loc[idx].to_dict()
                    st.session_state["current_user"] = user
                    st.rerun()
                else:
                    # 🔑 admin / asdqwe123!@# 로 로그인한 경우는 강제로 admin PW를 덮어써서 살려준다
                    if login_id == DEFAULT_ADMIN_ID and login_pw == DEFAULT_ADMIN_PASSWORD:
                        users_df.loc[idx, "password"] = DEFAULT_ADMIN_PASSWORD
                        users_df.loc[idx, "role"] = "admin"
                        users_df.loc[idx, "last_login_at"] = now_str
                        save_user_db(users_df)
                        user = users_df.loc[idx].to_dict()
                        st.session_state["current_user"] = user
                        st.rerun()
                    else:
                        st.error("비밀번호가 올바르지 않습니다.")

    # ------------------ 회원가입 ------------------
    elif menu == "회원가입":
        with st.form("signup_form"):
            reg_id = st.text_input("아이디", key="reg_id")
            reg_pw = st.text_input("비밀번호", type="password", key="reg_pw")
            reg_pw2 = st.text_input("비밀번호 확인", type="password", key="reg_pw2")
            reg_hospital = st.text_input("병원명", key="reg_hospital")
            reg_affiliation = st.text_input("소속 (예: 대장항문외과)", key="reg_affiliation")
            reg_position = st.text_input("직책 (예: 교수)", value="교수", key="reg_position")
            reg_name = st.text_input("이름", key="reg_name")
            submitted = st.form_submit_button("회원가입")

        if submitted:
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
                            "last_login_at": "",
                        }
                    ]
                )
                users_df = pd.concat([users_df, new_row], ignore_index=True)
                save_user_db(users_df)
                st.session_state["signup_done"] = True
                st.session_state["auth_menu"] = "로그인"
                st.rerun()

    # ------------------ 비밀번호 재설정 ------------------
    elif menu == "비밀번호 재설정":
        with st.form("reset_pw_form"):
            rp_id = st.text_input("아이디", key="rp_id")
            rp_name = st.text_input("이름", key="rp_name")
            rp_hosp = st.text_input("병원명", key="rp_hosp")
            new_pw = st.text_input("새 비밀번호", type="password", key="rp_new_pw")
            new_pw2 = st.text_input("새 비밀번호 확인", type="password", key="rp_new_pw2")
            submitted = st.form_submit_button("비밀번호 재설정")

        if submitted:
            cond = (
                (users_df["user_id"] == rp_id)
                & (users_df["name"] == rp_name)
                & (users_df["hospital"] == rp_hosp)
            )
            row = users_df[cond]
            if row.empty:
                st.error("입력하신 정보와 일치하는 계정을 찾을 수 없습니다.")
            elif not new_pw or new_pw != new_pw2:
                st.error("새 비밀번호가 비어 있거나 일치하지 않습니다.")
            else:
                users_df.loc[cond, "password"] = new_pw
                save_user_db(users_df)
                # ✅ 안내 + 로그인 화면으로 이동
                st.session_state["pw_reset_done"] = True
                st.session_state["auth_menu"] = "로그인"
                st.rerun()

    # ------------------ 아이디 찾기 ------------------
    elif menu == "아이디 찾기":
        with st.form("find_id_form"):
            fid_name = st.text_input("이름", key="fid_name")
            fid_hosp = st.text_input("병원명", key="fid_hosp")
            submitted = st.form_submit_button("아이디 찾기")

        if submitted:
            cond = (users_df["name"] == fid_name) & (users_df["hospital"] == fid_hosp)
            rows = users_df[cond]
            if rows.empty:
                st.error("입력하신 정보와 일치하는 아이디가 없습니다.")
            else:
                ids = rows["user_id"].dropna().unique().tolist()
                if len(ids) == 1:
                    st.success(f"해당 정보로 등록된 아이디는 **{ids[0]}** 입니다.")
                else:
                    st.success(
                        "해당 정보로 등록된 아이디:\n\n"
                        + ", ".join(f"**{uid}**" for uid in ids)
                    )

    return None, users_df


# =========================
# 관리자 화면
# =========================
def render_admin_view(users_df: pd.DataFrame, current_user: dict):
    st.title("👨‍💼 관리자 화면")
    st.caption("가입한 사용자 목록 및 관리자 계정 설정")

    # --- 관리자 계정 설정 변경 ---
    with st.expander("🔐 내 관리자 계정 설정 변경", expanded=True):
        new_admin_id = st.text_input(
            "새 관리자 아이디",
            value=current_user.get("user_id", ""),
            key="admin_new_id",
        )
        new_admin_pw = st.text_input(
            "새 비밀번호 (변경 시에만 입력)",
            type="password",
            key="admin_new_pw",
        )
        new_admin_pw2 = st.text_input(
            "새 비밀번호 확인",
            type="password",
            key="admin_new_pw2",
        )
        submitted = st.button("관리자 계정 업데이트", key="btn_admin_update")

    if submitted:
        if not new_admin_id:
            st.error("아이디는 비워둘 수 없습니다.")
        else:
            # 다른 사람과 아이디 중복 여부 확인
            conflict = users_df[
                (users_df["user_id"] == new_admin_id)
                & (users_df["user_id"] != current_user.get("user_id"))
            ]
            if not conflict.empty:
                st.error("이미 사용 중인 아이디입니다.")
            else:
                # 비밀번호 변경 여부
                if new_admin_pw or new_admin_pw2:
                    if new_admin_pw != new_admin_pw2:
                        st.error("새 비밀번호와 확인이 일치하지 않습니다.")
                        return
                    if not new_admin_pw:
                        st.error("새 비밀번호가 비어 있습니다.")
                        return
                    final_pw = new_admin_pw
                else:
                    final_pw = current_user.get("password", "")

                # DB 업데이트
                mask = users_df["user_id"] == current_user.get("user_id")
                users_df.loc[mask, "user_id"] = new_admin_id
                users_df.loc[mask, "password"] = final_pw
                save_user_db(users_df)

                updated_user = current_user.copy()
                updated_user["user_id"] = new_admin_id
                updated_user["password"] = final_pw
                st.session_state["current_user"] = updated_user

                st.success("관리자 계정 정보가 변경되었습니다.")
                st.rerun()

    st.markdown("---")

    # --- 가입자 목록 (접속일자 포함) ---
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
        "last_login_at",  # ✅ 마지막 접속일자
    ]
    existing_cols = [c for c in display_cols if c in users_df.columns]

    st.markdown("#### 가입자 목록")
    st.dataframe(users_df[existing_cols], use_container_width=True)
    st.caption("※ 비밀번호는 표시되지 않습니다.")


# =========================
# 데이터 로딩 & 세션 초기화
# =========================
def reset_session_state(new_file_id):
    """새 파일 업로드 시 분석 관련 state 초기화."""
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
    """CSV / Excel 파일을 pandas DataFrame으로 로딩."""
    try:
        df = None
        file_ext = uploaded_file.name.split(".")[-1].lower()
        uploaded_file.seek(0)
        selected_sheet = None

        if file_ext == "csv":
            use_header = st.checkbox(
                "Use first row as header", value=True, key="csv_use_header"
            )
            header_opt = 0 if use_header else None
            df = pd.read_csv(uploaded_file, header=header_opt)

        elif file_ext in ["xlsx", "xls"]:
            import openpyxl  # 엔진 보장
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


# =========================
# 메인 앱
# =========================
def main():
    st.title("📊 Medical Statistics Analysis Tool")
    st.caption("자동 통계 및 PSM 분석 도구")

    # 1) 인증
    user, users_df = render_auth_block()
    if user is None:
        return

    # 2) Sidebar: 사용자 정보 + 모드 선택 + 업로드
    uploaded_file = None
    mode = "통계 분석"

    with st.sidebar:
        st.header("👤 사용자 정보")
        st.markdown(
            f"**이름:** {user.get('name', '')}  \n"
            f"**아이디:** {user.get('user_id', '')}  \n"
            f"**병원:** {user.get('hospital', '')}"
        )
        if user.get("affiliation"):
            st.caption(f"소속: {user.get('affiliation', '')}")
        if user.get("position"):
            st.caption(f"직책: {user.get('position', '')}")

        if st.button("로그아웃", key="logout_btn"):
            st.session_state.pop("current_user", None)
            st.rerun()

        st.write("---")

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
                "- **Table 1**: Baseline Characteristics (T-test, Chi-square, etc.)\n"
                "- **Cox Regression**: Survival Analysis (Kaplan-Meier, Cox PH)\n"
                "- **Logistic Regression**: Binary Outcome Prediction (ROC Curve)\n"
                "- **PSM**: Propensity Score Matching"
            )

    # 관리자 모드
    if user.get("role") == "admin" and mode == "관리자 화면":
        render_admin_view(users_df, user)
        return

    # 통계 분석 모드
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
        st.info("👈 왼쪽 사이드바에서 데이터 파일을 업로드 해 주세요.")


if __name__ == "__main__":
    main()
