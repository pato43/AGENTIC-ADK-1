# app.py
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.express as px

# ---------- CONFIG GENERAL ----------
st.set_page_config(
    page_title="Agente ADK — Retail MX",
    layout="wide",
    page_icon="🛒"
)

# Tema oscuro: colores de alto contraste
st.markdown("""
<style>
body {background-color: #0f1116; color: #f0f2f6;}
h1,h2,h3,h4,h5 {color: #f8fafc;}
.section-title { font-size: 1.1rem; font-weight: 800; margin-top: 1rem; color: #f8fafc;}
.kpi-card {
    border-radius: 14px; padding: 14px; font-weight: bold;
    color: white; text-align: center;
}
.kpi-sales {background: linear-gradient(135deg, #2563eb, #1e40af);}
.kpi-units {background: linear-gradient(135deg, #16a34a, #166534);}
.kpi-ticket {background: linear-gradient(135deg, #f97316, #9a3412);}
.spinner-msg {
    padding: 10px; background: #1e293b; border-radius: 8px;
    margin-bottom: 6px; font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- GENERADOR DE DATOS SINTÉTICOS ----------
def make_data(seed=23, days=120):
    rng = np.random.default_rng(seed)
    end = datetime.today().date()
    dates = pd.date_range(end - timedelta(days=days-1), end, freq="D")
    regions = ["Centro", "Norte", "Occidente", "Sureste"]
    cats = ["Abarrotes", "Electrónica", "Hogar", "Moda"]
    rows = []
    for d in dates:
        for r in regions:
            for c in cats:
                sales = max(50, 500 + 4*(d.toordinal()%200) + rng.normal(0,80))
                units = max(1, int(sales/25 + rng.normal(0,4)))
                rows.append([d, r, c, sales, units])
    return pd.DataFrame(rows, columns=["date","region","category","sales","units"])

df = make_data()

# ---------- PANTALLA INICIAL ----------
if "show_dashboard" not in st.session_state:
    st.session_state.show_dashboard = False

if not st.session_state.show_dashboard:
    st.markdown("## 🛒 Agente ADK — Retail MX")
    st.caption("Análisis conversacional de datos de ventas omnicanal en grandes retailers.")
    query = st.text_input("Escribe tu consulta:", placeholder="Ej. Ventas del último trimestre por categoría y región")
    if query:
        with st.spinner("Procesando consulta..."):
            msgs = [
                "Agente ADK: entendiendo intención del usuario...",
                "Agente ADK: consultando datos en BigQuery...",
                "Agente ADK: ejecutando SQL y agregaciones...",
                "Agente ADK: generando visualizaciones..."
            ]
            for m in msgs:
                st.markdown(f'<div class="spinner-msg">{m}</div>', unsafe_allow_html=True)
                time.sleep(1.2)
        st.session_state.show_dashboard = True
        st.session_state.query = query
        st.rerun()

# ---------- DASHBOARD ----------
if st.session_state.show_dashboard:
    st.markdown(f"### Consulta: {st.session_state.query}")
    st.markdown('<div class="section-title">KPIs Clave</div>', unsafe_allow_html=True)
    total_sales = df["sales"].sum()
    total_units = df["units"].sum()
    ticket_medio = total_sales / total_units

    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="kpi-card kpi-sales">Ventas<br>${total_sales:,.0f} MXN</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card kpi-units">Unidades<br>{total_units:,}</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card kpi-ticket">Ticket medio<br>${ticket_medio:,.2f}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Tendencia de ventas</div>', unsafe_allow_html=True)
    ts = df.groupby("date", as_index=False)["sales"].sum()
    fig_ts = px.line(ts, x="date", y="sales", title="", markers=True,
                     color_discrete_sequence=["#38bdf8"])
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown('<div class="section-title">Ventas por región y categoría</div>', unsafe_allow_html=True)
    seg = df.groupby(["region","category"], as_index=False)["sales"].sum()
    fig_seg = px.bar(seg, x="region", y="sales", color="category",
                     color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig_seg, use_container_width=True)
