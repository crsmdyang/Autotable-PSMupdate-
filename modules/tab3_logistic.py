import streamlit as st
import pandas as pd
import numpy as np
import io
import xlsxwriter
import statsmodels.api as sm
from .utils import ensure_binary_event, ordered_levels, make_dummies, check_vif, plot_forest

def render_tab3(df):
    st.subheader("Binary Logistic Regression")
    dy = st.selectbox("Y (종속변수)", df.columns, key='l_y')
    if dy:
        evs = st.multiselect("Event(1)", df[dy].unique(), key='l_ev')
        cts = st.multiselect("Control(0)", df[dy].unique(), key='l_ct')
        if evs and cts:
            dfl = df.copy(); dfl['Y'] = ensure_binary_event(dfl[dy], set(evs), set(cts))
            dfl = dfl.dropna(subset=['Y'])
            ivs = st.multiselect("X (독립변수)", [c for c in df.columns if c!=dy], key='l_x')
            pe = st.number_input("P-enter", 0.05, key='l_pe')
            
            if st.button("Logistic 실행"):
                sig_vars = []
                for v in ivs:
                    try:
                        t = dfl[['Y', v]].dropna()
                        if t[v].dtype=='object' or t[v].nunique()<10: X = make_dummies(t, v, ordered_levels(t[v]))
                        else: X = pd.to_numeric(t[v], errors='coerce').to_frame()
                        X = sm.add_constant(X)
                        m = sm.Logit(t['Y'], X).fit(disp=0)
                        if min([m.pvalues[c] for c in m.pvalues.index if c!='const']) < pe: sig_vars.append(v)
                    except: pass
                
                if sig_vars:
                    XL = []
                    for v in sig_vars:
                        if dfl[v].dtype=='object': XL.append(make_dummies(dfl[[v]], v, ordered_levels(dfl[v])))
                        else: XL.append(pd.to_numeric(dfl[v], errors='coerce'))
                    XM = sm.add_constant(pd.concat(XL, axis=1)); DM = pd.concat([dfl['Y'], XM], axis=1).dropna()
                    try:
                        res = sm.Logit(DM['Y'], DM.drop(columns=['Y'])).fit(disp=0)
                        conf = res.conf_int(); conf['OR'] = res.params.apply(np.exp)
                        conf['Lo'] = conf[0].apply(np.exp); conf['Hi'] = conf[1].apply(np.exp); conf['p'] = res.pvalues
                        res_df = conf[['OR', 'Lo', 'Hi', 'p']].drop('const', errors='ignore')
                        st.dataframe(res_df.style.format("{:.3f}"))
                        fig = plot_forest(res_df, "Logistic OR", "OR"); st.pyplot(fig)
                        
                        out_l = io.BytesIO()
                        with pd.ExcelWriter(out_l, engine='xlsxwriter') as w: res_df.to_excel(w)
                        st.download_button("📥 로지스틱 저장", out_l.getvalue(), "Logistic.xlsx")
                    except Exception as e: st.error(f"Error: {e}")