import streamlit as st
import pandas as pd
import io
import xlsxwriter
from .utils import suggest_variable_type_single, analyze_table1_robust

def render_tab1(df):
    st.subheader("Table 1: Baseline Characteristics")
    group_col = st.selectbox("그룹 변수 선택", df.columns, key='t1_group')
    
    if group_col:
        unique_vals = df[group_col].dropna().unique()
        col1, col2 = st.columns(2)
        with col1:
            selected_vals = st.multiselect("비교할 그룹 값 (2개 이상)", unique_vals, default=unique_vals[:2] if len(unique_vals)>=2 else unique_vals)
        
        all_cols = [c for c in df.columns if c != group_col]
        
        # 변수 설정 초기화
        if 'var_config_df' not in st.session_state:
            initial_data = []
            for col in all_cols:
                initial_data.append({
                    "Include": True,
                    "Variable": col,
                    "Type": suggest_variable_type_single(df, col)
                })
            st.session_state['var_config_df'] = pd.DataFrame(initial_data)
        
        st.write("---")
        st.markdown("#### ⚙️ 분석 변수 및 타입 설정")
        
        # 전체 선택/해제 버튼
        col_btn1, col_btn2, _ = st.columns([0.15, 0.15, 0.7])
        if col_btn1.button("✅ 전체 선택", key='btn_select_all'):
            st.session_state['var_config_df']['Include'] = True
            st.rerun()
        if col_btn2.button("⬜ 전체 해제", key='btn_deselect_all'):
            st.session_state['var_config_df']['Include'] = False
            st.rerun()

        # 에디터
        edited_config = st.data_editor(
            st.session_state['var_config_df'],
            column_config={
                "Include": st.column_config.CheckboxColumn("Include?", width="small", default=True),
                "Variable": st.column_config.TextColumn("Variable Name", width="medium", disabled=True),
                "Type": st.column_config.SelectboxColumn("Data Type", width="medium", options=["Continuous", "Categorical"], required=True)
            },
            hide_index=True,
            use_container_width=True,
            num_rows="fixed", 
            key='var_manager_editor'
        )
        
        st.session_state['var_config_df'] = edited_config

        # 선택된 변수 추출
        selected_rows = edited_config[edited_config['Include'] == True]
        target_vars = selected_rows['Variable'].tolist()
        user_cont_vars = selected_rows[selected_rows['Type'] == 'Continuous']['Variable'].tolist()
        user_cat_vars = selected_rows[selected_rows['Type'] == 'Categorical']['Variable'].tolist()
        
        value_map = {v: str(v) for v in selected_vals}

        if len(selected_vals) >= 2 and target_vars:
            if st.button("Table 1 생성", key='btn_t1'):
                with st.spinner("분석 중... (정규성 검정 포함)"):
                    t1_res, error_info = analyze_table1_robust(
                        df, group_col, value_map, target_vars, 
                        user_cont_vars, user_cat_vars
                    )
                    
                    if error_info:
                        st.error(f"🚨 **데이터 오류: '{error_info['var']}'**")
                        st.warning(f"오류 내용: {error_info['msg']}")
                    else:
                        st.dataframe(t1_res, use_container_width=True)
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            t1_res.to_excel(writer, index=False)
                        st.download_button("📥 엑셀 다운로드", output.getvalue(), "Table1_Robust.xlsx")