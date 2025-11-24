import streamlit as st
import pandas as pd
import io

# 모듈 불러오기
from modules.tab1_table1 import render_tab1
from modules.tab2_cox import render_tab2
from modules.tab3_logistic import render_tab3
from modules.tab4_psm import render_tab4
from modules.tab5_methods import render_tab5

# 페이지 설정
st.set_page_config(page_title="Dr.Stats Ultimate: Medical Statistics", layout="wide")
st.title("Dr.Stats Ultimate: Medical Statistics Tool")

# 1. 파일 업로드 (공통 영역)
uploaded_file = st.file_uploader("📂 데이터 파일 업로드 (Excel/CSV)", type=['xlsx', 'xls', 'csv'])

if uploaded_file:
    # 파일 및 시트 처리
    selected_sheet = None
    if uploaded_file.name.endswith(('.xlsx', '.xls')):
        try:
            xl = pd.ExcelFile(uploaded_file)
            sheet_names = xl.sheet_names
            if len(sheet_names) > 1:
                selected_sheet = st.selectbox("📑 시트 선택", sheet_names)
            else:
                selected_sheet = sheet_names[0]
        except Exception as e:
            st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
            st.stop()
    
    # 고유 ID 생성 (파일 변경 감지용)
    file_id = f"{uploaded_file.name}_{selected_sheet if selected_sheet else 'csv'}_{uploaded_file.size}"
    
    # 데이터 로드 및 세션 초기화
    if 'current_file_id' not in st.session_state or st.session_state['current_file_id'] != file_id:
        try:
            if selected_sheet:
                df_load = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            else:
                df_load = pd.read_csv(uploaded_file)
            
            df_load.columns = df_load.columns.astype(str).str.strip()
            st.session_state['df'] = df_load
            st.session_state['current_file_id'] = file_id
            
            # 각종 설정 초기화
            keys_to_clear = ['var_config_df', 'current_target_hash', 'psm_var_config', 'psm_done', 'psm_matched_df', 'psm_original_w_score']
            for k in keys_to_clear:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()
        except Exception as e:
            st.error(f"데이터 로드 실패: {e}")
            st.stop()

    df = st.session_state.get('df')

    if df is not None:
        # 데이터 에디터 (수정 기능) - 항상 상단 노출
        with st.expander("✏️ 원본 데이터 미리보기 및 수정", expanded=False):
            st.info("데이터 오류(문자/숫자 혼합)가 있으면 여기서 직접 수정하세요. 수정 시 즉시 반영됩니다.")
            edited_df = st.data_editor(st.session_state['df'], num_rows="dynamic", use_container_width=True, key='main_editor')
            if not edited_df.equals(st.session_state['df']):
                st.session_state['df'] = edited_df
                st.rerun()
        
        st.divider()

        # 탭 구성
        tab1, tab2, tab3, tab4, tab_methods = st.tabs([
            "📊 Table 1 (기초통계)", 
            "⏱️ Cox Regression", 
            "💊 Logistic Regression", 
            "⚖️ PSM (매칭)", 
            "📝 Methods 작문"
        ])

        # 각 탭의 기능은 모듈에게 위임
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
    st.info("👈 좌측 상단 메뉴 혹은 위쪽 버튼을 통해 데이터 파일을 업로드해주세요.")