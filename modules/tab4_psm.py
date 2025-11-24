import streamlit as st
import pandas as pd
import numpy as np
import io
import xlsxwriter
import seaborn as sns
import matplotlib.pyplot as plt
from .utils import calculate_smd, run_psm, suggest_variable_type_single, analyze_table1_robust

def render_tab4(df):
    st.header("⚖️ Propensity Score Matching")
    c1, c2 = st.columns(2)
    tc = c1.selectbox("치료 변수 (0/1)", df.columns, key='p_t')
    if tc:
        vls = df[tc].dropna().unique()
        if len(vls)==2: t1 = c2.selectbox("치료군 값(1)", vls, key='p_v')
        else: st.warning("2개 값 필수")
    
    covs = st.multiselect("매칭 공변량", [c for c in df.columns if c!=tc], key='p_c')
    cal = st.slider("Caliper", 0.0, 1.0, 0.2)

    if st.button("PSM 실행"):
        if not covs: st.error("공변량 선택 필수")
        else:
            with st.spinner("Matching..."):
                dp = df.copy(); dp['__T'] = np.where(dp[tc]==t1, 1, 0)
                m_df, org = run_psm(dp, '__T', covs, cal)
                
                if m_df is None: st.error("매칭 실패")
                else:
                    st.session_state['psm_done'] = True
                    st.session_state['psm_matched_df'] = m_df
                    st.session_state['psm_org_df'] = org
                    st.session_state['psm_covs'] = covs
                    st.session_state['psm_treat_col'] = tc

    # PSM 결과 및 Table 1 생성 UI
    if st.session_state.get('psm_done'):
        m_df = st.session_state['psm_matched_df']
        org = st.session_state['psm_org_df']
        covs = st.session_state['psm_covs']
        tc = st.session_state['psm_treat_col']
        
        st.success(f"매칭 완료! N={len(m_df)}")
        
        # 1. Balance Check
        sb = calculate_smd(org, '__T', covs)
        sa = calculate_smd(m_df, '__T', covs)
        sm = pd.merge(sb, sa, on='Variable', suffixes=('_Before', '_After'))
        st.dataframe(sm.style.format("{:.3f}"))
        
        fig, ax = plt.subplots()
        sns.scatterplot(data=sm, x='SMD_Before', y='Variable', color='red', label='Before')
        sns.scatterplot(data=sm, x='SMD_After', y='Variable', color='blue', label='After')
        ax.axvline(0.1, ls='--'); st.pyplot(fig)

        out_p = io.BytesIO()
        with pd.ExcelWriter(out_p, engine='xlsxwriter') as w:
            m_df.drop(columns=['__T', 'logit_ps'], errors='ignore').to_excel(w, index=False)
        st.download_button("📥 매칭 데이터 저장", out_p.getvalue(), "Matched.xlsx")

        # [Matched Table 1 Generator]
        st.markdown("---")
        st.subheader("📊 Matched Cohort Table 1")
        
        # 매칭 데이터에서 사용할 수 있는 모든 변수
        avail_cols = [c for c in m_df.columns if c not in ['__T', 'logit_ps', 'propensity_score', tc]]
        
        if 'psm_var_config' not in st.session_state:
            init_d = []
            for c in avail_cols:
                init_d.append({"Include": (c in covs), "Variable": c, "Type": suggest_variable_type_single(m_df, c)})
            st.session_state['psm_var_config'] = pd.DataFrame(init_d)
        
        # 전체 선택/해제 버튼
        c_b1, c_b2, _ = st.columns([0.2,0.2,0.6])
        if c_b1.button("✅ 전체 선택 (Matched)", key='psm_all'):
            st.session_state['psm_var_config']['Include'] = True; st.rerun()
        if c_b2.button("⬜ 전체 해제 (Matched)", key='psm_none'):
            st.session_state['psm_var_config']['Include'] = False; st.rerun()

        # 변수 설정 에디터
        psm_cfg = st.data_editor(
            st.session_state['psm_var_config'],
            column_config={
                "Include": st.column_config.CheckboxColumn(width="small"),
                "Variable": st.column_config.TextColumn(disabled=True),
                "Type": st.column_config.SelectboxColumn(options=["Continuous", "Categorical"])
            },
            hide_index=True, use_container_width=True, num_rows="fixed", key='psm_editor'
        )
        st.session_state['psm_var_config'] = psm_cfg
        
        # 선택된 변수 파싱
        sel = psm_cfg[psm_cfg['Include']==True]
        t_vars = sel['Variable'].tolist()
        u_cont = sel[sel['Type']=='Continuous']['Variable'].tolist()
        u_cat = sel[sel['Type']=='Categorical']['Variable'].tolist()
        
        # 값 매핑 (원래 치료 변수 값 사용)
        mt_vals = m_df[tc].unique()
        val_map = {v: str(v) for v in mt_vals}
        
        if st.button("Generate Matched Table 1", key='btn_mt1'):
            if not t_vars: st.warning("변수를 선택하세요.")
            else:
                mt1, err = analyze_table1_robust(m_df, tc, val_map, t_vars, u_cont, u_cat)
                if err: st.error(err)
                else:
                    st.dataframe(mt1)
                    out_m1 = io.BytesIO()
                    with pd.ExcelWriter(out_m1, engine='xlsxwriter') as w: mt1.to_excel(w, index=False)
                    st.download_button("📥 Matched Table 1 저장", out_m1.getvalue(), "Matched_Table1.xlsx")