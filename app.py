# app.py

import os
from datetime import datetime
import io

import streamlit as st
import pandas as pd
import numpy as np

from modules.tab1_table1 import render_tab1
from modules.tab2_cox import render_tab2
from modules.tab3_logistic import render_tab3
from modules.tab4_psm import render_tab4
from modules.tab5_methods import render_tab5

# ------------------------------------------------
# Page Config & CSS
# ------------------------------------------------
st.set_page_config(
    page_title="Medical Statistics Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

    /* 버튼 전체 크기 축소 */
    .stButton > button {
        background-color: #00ADB5;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.25rem 0.6rem;
        font-size: 0.9rem;
        font-weight: 500;
        min-height: 0px;
        line-height: 1.2;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #007A80;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.3rem 0.8rem;
        border-radius: 6px 6px 0 0;
        background-color: #393E46;
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
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------
# User DB & Auth
# ------------------------------------------------
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
    "last_login",  # 🔹 마지막 접속일자
]

DEFAULT_ADMIN_ID = "admin"
DEFAULT_ADMIN_PASSWORD = "asdqwe123!@#"  # 기본 관리자 비밀번호


def load_user_db() -> pd.DataFrame:
    """
    users_db.csv를 로드하고,
    - 파일이 없으면 기본 admin 계정 생성
    - role='admin' 계정이 하나도 없으면 기본 admin 계정 추가
    - 과거 기본 비번(admin1234) or 비어 있는 admin 계정은 새 비번으로 한번 업데이트
    """
    if os.path.exists(USER_DB_PATH):
        try:
            df = pd.read_csv(USER_DB_PATH, dtype=str, encoding="utf-8")
        except Exception:
            df = pd.DataFrame(columns=USER_DB_COLUMNS)
    else:
        df = pd.DataFrame(columns=USER_DB_COLUMNS)

    # 컬럼 보정
    for col in USER_DB_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    changed = False

    # 1) admin role이 하나도 없으면 기본 관리자 계정 생성
    admins = df[df["role"] == "admin"]
    if admins.empty:
        new_admin = {
            "user_id": DEFAULT_ADMIN_ID,
            "password": DEFAULT_ADMIN_PASSWORD,
            "hospital": "",
            "affiliation": "Admin",
            "position": "관리자",
            "name": "Administrator",
            "role": "admin",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_login": "",
        }
        df = pd.concat([df, pd.DataFrame([new_admin])], ignore_index=True)
        changed = True
    else:
        # 2) 예전 기본 비번(admin1234) 또는 공백인 admin 계정은 새 비번으로 업데이트
        mask_old = (
            (df["role"] == "admin")
            & (df["user_id"] == DEFAULT_ADMIN_ID)
            & (df["password"].isin(["admin1234", ""]))
        )
        if mask_old.any():
            df.loc[mask_old, "password"] = DEFAULT_ADMIN_PASSWORD
            changed = True

    if changed:
        df.to_csv(USER_DB_PATH, index=False, encoding="utf-8")

    return df[USER_DB_COLUMNS]


def save_user_db(df: pd.DataFrame) -> None:
    for col in USER_DB_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[USER_DB_COLUMNS]
    df.to_csv(USER_DB_PATH, index=False, encoding="utf-8")


# ---------------------- Auth UI ----------------------
def render_auth_block():
    """
    로그인 / 회원가입 / 비밀번호 재설정 / 아이디 찾기 UI.

    Returns
    -------
    user : dict or None
    users_df : pd.DataFrame
    """
    users_df = load_user_db()
    current_user = st.session_state.get("current_user")

    # 이미 로그인 된 상태
    if current_user is not None:
        with st.sidebar:
            st.success(
                f"로그인: {current_user.get('name', '')} "
                f"({current_user.get('user_id', '')})"
            )
            if st.button("로그아웃", key="btn_logout"):
                st.session_state.pop("current_user", None)
                st.rerun()
        return current_user, users_df

    # 로그인되지 않은 상태
    st.markdown("### 🔐 로그인 / 회원 관리")

    # 이전 액션에서 전달된 안내 메시지
    info_msg = st.session_state.pop("auth_info_msg", None)
    if info_msg:
        st.success(info_msg)

    # 메뉴 상태
    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "로그인"

    auth_mode = st.radio(
        "메뉴 선택",
        ["로그인", "회원가입", "비밀번호 재설정", "아이디 찾기"],
        key="auth_mode",
        horizontal=True,
    )

    # ------ 로그인 ------
    if auth_mode == "로그인":
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
                    # 🔹 마지막 접속일자 기록
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    users_df.loc[
                        users_df["user_id"] == login_id, "last_login"
                    ] = now_str
                    save_user_db(users_df)

                    user = users_df[users_df["user_id"] == login_id].iloc[0].to_dict()
                    st.session_state["current_user"] = user
                    st.success(f"{user.get('name', '')}님 환영합니다.")
                    st.rerun()

    # ------ 회원가입 ------
    elif auth_mode == "회원가입":
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
                            "last_login": "",
                        }
                    ]
                )
                users_df = pd.concat([users_df, new_row], ignore_index=True)
                save_user_db(users_df)
                # 🔹 안내 메시지 + 로그인 화면으로 이동
                st.session_state["auth_info_msg"] = (
                    "회원가입이 완료되었습니다. 이제 로그인해 주세요."
                )
                st.session_state["auth_mode"] = "로그인"
                st.rerun()

    # ------ 비밀번호 재설정 ------
    elif auth_mode == "비밀번호 재설정":
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
                # 🔹 안내 메시지 + 로그인 화면으로 이동
                st.session_state["auth_info_msg"] = (
                    "비밀번호가 재설정되었습니다. 새 비밀번호로 로그인해 주세요."
                )
                st.session_state["auth_mode"] = "로그인"
                st.rerun()

    # ------ 아이디 찾기 ------
    elif auth_mode == "아이디 찾기":
        st.markdown("#### 아이디 찾기")
        fid_name = st.text_input("이름", key="fid_name")
        fid_hospital = st.text_input("병원명", key="fid_hospital")

        if st.button("아이디 찾기", key="btn_find_id"):
            cond = (users_df["name"] == fid_name) & (
                users_df["hospital"] == fid_hospital
            )
            rows = users_df[cond]
            if rows.empty:
                st.error("입력하신 정보와 일치하는 아이디가 없습니다.")
            else:
                ids = rows["user_id"].dropna().unique().tolist()
                if len(ids) == 1:
                    st.success(f"해당 정보로 등록된 아이디는 **{ids[0]}** 입니다.")
                else:
                    joined_ids = ", ".join(ids)
                    st.success(
                        f"해당 정보로 등록된 아이디는 다음과 같습니다: **{joined_ids}**"
                    )

    return None, users_df


def render_admin_view(users_df: pd.DataFrame, current_user: dict) -> None:
    """관리자: 가입자 목록 + 관리자 계정 설정."""
    st.title("👨‍⚕️ 관리자 화면")
    st.caption("가입자 목록 확인 및 관리자 계정 설정 변경")

    # 가입자 목록 (비밀번호는 노출 X)
    if users_df is None or users_df.empty:
        st.info("아직 가입된 사용자가 없습니다.")
    else:
        display_cols = [
            "user_id",
            "name",
            "hospital",
            "affiliation",
            "position",
            "role",
            "created_at",   # 가입일자
            "last_login",   # 🔹 마지막 접속일자
        ]
        existing_cols = [c for c in display_cols if c in users_df.columns]
        st.subheader("📋 가입자 목록")
        st.dataframe(users_df[existing_cols], use_container_width=True)

    # 관리자 계정 변경
    st.markdown("---")
    st.subheader("🔑 관리자 계정 설정 변경")

    cur_admin_id = current_user.get("user_id", "")
    st.caption(f"현재 관리자 아이디: **{cur_admin_id}**")

    with st.form("admin_settings_form"):
        new_admin_id = st.text_input(
            "새 관리자 아이디", value=cur_admin_id, key="admin_new_id"
        )
        cur_pw = st.text_input("현재 비밀번호", type="password", key="admin_cur_pw")
        new_pw = st.text_input("새 비밀번호", type="password", key="admin_new_pw")
        new_pw2 = st.text_input(
            "새 비밀번호 확인", type="password", key="admin_new_pw2"
        )
        submitted = st.form_submit_button("관리자 계정 변경")

    if submitted:
        db = load_user_db()
        mask = (db["user_id"] == cur_admin_id) & (db["role"] == "admin")
        row = db[mask]

        if row.empty:
            st.error("관리자 계정을 찾을 수 없습니다. (users_db.csv 확인 필요)")
        else:
            stored_pw = str(row.iloc[0]["password"])
            if stored_pw != cur_pw:
                st.error("현재 비밀번호가 올바르지 않습니다.")
            elif not new_admin_id:
                st.error("새 관리자 아이디를 입력해주세요.")
            elif not new_pw:
                st.error("새 비밀번호를 입력해주세요.")
            elif new_pw != new_pw2:
                st.error("새 비밀번호 확인이 일치하지 않습니다.")
            elif (db["user_id"] == new_admin_id).any() and new_admin_id != cur_admin_id:
                st.error("이미 사용 중인 아이디입니다.")
            else:
                db.loc[mask, "user_id"] = new_admin_id
                db.loc[mask, "password"] = new_pw
                save_user_db(db)
                updated_user = db.loc[mask].iloc[0].to_dict()
                st.session_state["current_user"] = updated_user
                st.success("관리자 아이디/비밀번호가 변경되었습니다.")


# ------------------------------------------------
# Data & Session Handling
# ------------------------------------------------
def reset_session_state(new_file_id: str) -> None:
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
            import openpyxl  # 안전하게 엔진 확보

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


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    st.title("📊 Medical Statistics Analysis Tool")
    st.caption("자동 통계 및 PSM 분석 도구")

    # 1) 인증
    user, users_df = render_auth_block()
    if user is None:
        # 로그인/회원 관리 화면만 보여주는 상태
        return

    # 2) Sidebar: 사용자 정보 + 모드 선택
    uploaded_file = None
    mode = "통계 분석"

    with st.sidebar:
        st.header("👤 사용자 정보")
        st.markdown(
            f"**이름:** {user.get('name', '')}  \n"
            f"**아이디:** {user.get('user_id', '')}  \n"
            f"**병원:** {user.get('hospital', '')}"
        )
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

    # 3) 관리자 모드
    if user.get("role") == "admin" and mode == "관리자 화면":
        render_admin_view(users_df, user)
        return

    # 4) 통계 분석 모드
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
        st.info("👈 왼쪽에서 데이터 파일을 업로드 해주세요.")


if __name__ == "__main__":
    main()
