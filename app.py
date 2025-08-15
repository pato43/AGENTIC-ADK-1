# app.py
# Conversational BI Agent (ADK) • Caso Retail MX • Streamlit + Plotly
# 100% ilustrativo: no usa LLM/ADK reales ni se conecta a fuentes externas.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ------------------------
# CONFIG & THEME
# ------------------------
st.set_page_config(
    page_title="Conversational BI Agent (ADK) — Retail MX",
    layout="wide",
    page_icon="🛒"
)

# Estilos (dashboard horizontal, tarjetas KPI, “píldoras” de estado)
st.markdown("""
<style>
.block-container {padding: 0.5rem 1rem 1rem 1rem;}
header {visibility: hidden;}  /* Oculta el header default para look limpio */
.kpi-card {
  border-radius: 16px; padding: 14px 16px; color: #fff; font-weight: 700;
  background: linear-gradient(135deg, #6E00FF, #00D2FF);
  box-shadow: 0 8px 24px rgba(0,0,0,0.15);
}
.kpi-card.alt { background: linear-gradient(135deg, #FF5F6D, #FFC371); }
.kpi-card.alt2 { background: linear-gradient(135deg, #00C9FF, #92FE9D); color:#0d1730;}
.kpi-title { font-size: 0.8rem; opacity: 0.95; font-weight:600;}
.kpi-value { font-size: 1.6rem; line-height: 1.2; }
.kpi-sub { font-size: 0.85rem; opacity: 0.9; font-weight:500;}
.section-title{
  font-size: 1.05rem; font-weight: 800; letter-spacing: .2px;
  margin: 0 0 8px 0; color: #0d1730;
}
hr.section { border: none; height: 1px; background: linear-gradient(90deg, #6E00FF, #00D2FF); margin: 8px 0 16px;}
.pill {
  display:inline-block; padding:5px 12px; border-radius:999px; font-size:.8rem;
  background: #0d1730; color:white; margin:0 6px 6px 0; font-weight:600;
}
.brand{
  display:flex; gap:12px; align-items:center; padding:10px 0 6px 0;
}
.brand .title{
  font-weight:900; font-size:1.2rem; letter-spacing:.2px; color:#0d1730;
}
.badge{
  font-size:.75rem; font-weight:800; padding:4px 8px; border-radius:8px; color:#fff;
  background: linear-gradient(135deg, #0EA5E9, #22D3EE);
}
</style>
""", unsafe_allow_html=True)

PALETTE = px.colors.qualitative.Prism

# ------------------------
# DATOS SINTÉTICOS (retail MX omnicanal)
# ------------------------
@st.cache_data(show_spinner=False)
def make_data(seed=23, days=420):
    rng = np.random.default_rng(seed)
    end = datetime.today().date()
    dates = pd.date_range(end - timedelta(days=days-1), end, freq="D")

    # Contexto retail MX (inspirado en gran retailer omnicanal)
    regions = ["Centro", "Norte", "Occidente", "Sureste"]
    cats = ["Abarrotes", "Perecederos", "Hogar", "Electrónica", "Farmacia", "Moda"]
    channels = ["Tienda", "Ecommerce", "Pickup", "Mayorista"]

    rows = []
    for d in dates:
        for r in regions:
            for c in cats:
                # Tendencia + estacionalidades típicas retail
                base = 600 + 4.5*(d.toordinal() % 200)
                weekly = 150*np.sin((d.dayofyear/7)*2*np.pi)     # ciclo semanal
                season = 180*np.sin((d.dayofyear/30)*2*np.pi)    # ciclo mensual
                region_b = {"Centro":1.18, "Norte":1.04, "Occidente":0.97, "Sureste":0.92}[r]
                cat_b = {"Abarrotes":1.15,"Perecederos":1.05,"Hogar":1.0,"Electrónica":1.35,"Farmacia":0.95,"Moda":1.1}[c]
                noise = rng.normal(0, 90)
                sales = max(60, (base + weekly + season)*region_b*cat_b + noise)
                units = max(1, int(sales/28 + rng.normal(0, 5)))
                ch = rng.choice(channels, p=[0.48, 0.32, 0.12, 0.08])
                rows.append([d, r, c, ch, round(float(sales),2), units])

    df = pd.DataFrame(rows, columns=["date","region","category","channel","sales","units"])

    # Cohortes de clientes (altas previas)
    n_customers = 2200
    cust_ids = np.arange(1, n_customers+1)
    signup = pd.to_datetime(rng.choice(pd.date_range(dates.min()-pd.Timedelta(days=150), dates.min(), freq="D"),
                                       size=n_customers))
    customers = pd.DataFrame({"customer_id": cust_ids, "signup_date": signup})
    df["customer_id"] = rng.choice(cust_ids, size=len(df))
    return df, customers

df, customers = make_data()

# ------------------------
# ENCABEZADO / BRANDING (inspirado en WALMEX)
# ------------------------
top = st.container()
with top:
    c0, c1, c2 = st.columns([2.2, 2, 1], gap="large")
    with c0:
        st.markdown(
            '<div class="brand"><div class="title">🧠 Conversational BI Agent — ADK</div>'
            '<div class="badge">Retail MX • Omnicanal</div></div>',
            unsafe_allow_html=True
        )
        st.caption("Caso inspirado en operación omnicanal de un retailer grande en México (ventas diarias por región, categoría y canal).")
    with c1:
        # Entrada de “chat” para disparar el flujo
        user_msg = st.text_input("Escribe tu instrucción",
                                 value="Muéstrame ventas del último trimestre por categoría y región")
    with c2:
        run = st.button("▶ Ejecutar consulta", use_container_width=True, type="primary")

# ------------------------
# FILTROS GLOBALES (opcional)
# ------------------------
filt = st.container()
with filt:
    f1, f2, f3, f4 = st.columns([1.5,1.5,1,1], gap="medium")
    with f1:
        region_sel = st.multiselect("Región", sorted(df["region"].unique()), default=None)
    with f2:
        cat_sel = st.multiselect("Categoría", sorted(df["category"].unique()), default=None)
    with f3:
        ch_sel = st.multiselect("Canal", sorted(df["channel"].unique()), default=None)
    with f4:
        horizon_days = st.slider("Ventana (días)", min_value=60, max_value=180, value=120, step=10)

df_viz = df.copy()
if region_sel: df_viz = df_viz[df_viz["region"].isin(region_sel)]
if cat_sel: df_viz = df_viz[df_viz["category"].isin(cat_sel)]
if ch_sel: df_viz = df_viz[df_viz["channel"].isin(ch_sel)]

cut_date = df_viz["date"].max() - pd.Timedelta(days=horizon_days-1)
df_recent = df_viz[df_viz["date"] >= cut_date].copy()

# ------------------------
# “PENSANDO” / BITÁCORA (simulación ADK)
# ------------------------
sql_text = """
-- NL→SQL (simulado) • Retail MX (BigQuery)
WITH base AS (
  SELECT date, region, category, channel, sales, units
  FROM `mx_retail.analytics.daily_sales`
  WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
)
SELECT
  DATE_TRUNC(date, WEEK) AS week,
  region, category, channel,
  SUM(sales) AS sales, SUM(units) AS units,
  AVG(sales/NULLIF(units,0)) AS avg_price
FROM base
GROUP BY week, region, category, channel
ORDER BY week;
""".strip()

status_area = st.empty()
log_area = st.container()

def simulate_agent_run():
    with status_area:
        st.markdown('<span class="pill">ADK Planner: entendiendo intención</span>', unsafe_allow_html=True)
        st.markdown('<span class="pill">Herramienta: BigQuery</span>', unsafe_allow_html=True)
        st.markdown('<span class="pill">Ejecución: SQL / Agregaciones</span>', unsafe_allow_html=True)
        st.markdown('<span class="pill">Visualización: Plotly</span>', unsafe_allow_html=True)
        st.markdown('<span class="pill">Insight Composer: KPIs & Recos</span>', unsafe_allow_html=True)

# Disparo del flujo
if run or user_msg:
    simulate_agent_run()

# ------------------------
# KPIs PRINCIPALES
# ------------------------
def compute_kpis(dfr):
    total_sales = dfr["sales"].sum()
    total_units = int(dfr["units"].sum())
    aov = total_sales / max(1, len(dfr))
    tti = np.random.randint(6, 12)     # Time To Insight (s) simulado
    adoption = np.random.randint(68, 88)  # % usuarios activos
    # Fill rate y OSA (operación) aproximados
    fill_rate = np.random.uniform(91, 97)
    osa = np.random.uniform(93, 98)
    return total_sales, total_units, aov, tti, adoption, fill_rate, osa

tot_sales, tot_units, aov, tti, adoption, fill_rate, osa = compute_kpis(df_recent)

# Encabezado con SQL + Log
cA, cB = st.columns([2, 3], gap="large")
with cA:
    st.markdown('<div class="section-title">Instrucción del usuario</div>', unsafe_allow_html=True)
    st.code(user_msg or "Muéstrame ventas del último trimestre por categoría y región", language="markdown")
    st.markdown('<div class="section-title">Consulta generada (ADK → BigQuery)</div>', unsafe_allow_html=True)
    st.code(sql_text, language="sql")
    with log_area:
        st.markdown('<div class="section-title">Bitácora de acciones del Agente</div>', unsafe_allow_html=True)
        st.markdown("- Intención: KPI ventas/categorías/regiones/canales")
        st.markdown("- Selección de herramienta: BigQuery")
        st.markdown("- Generación de SQL y agregaciones")
        st.markdown("- Visualizaciones y ensamblado de hallazgos")

with cB:
    st.markdown('<div class="section-title">KPIs (últimos {} días)</div>'.format(horizon_days), unsafe_allow_html=True)
    k1,k2,k3,k4,k5,k6 = st.columns([1,1,1,1,1,1], gap="small")
    k1.markdown(f'<div class="kpi-card"><div class="kpi-title">Ventas</div><div class="kpi-value">${tot_sales:,.0f}</div><div class="kpi-sub">MXN</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-card alt"><div class="kpi-title">Unidades</div><div class="kpi-value">{tot_units:,}</div><div class="kpi-sub">transacciones</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-card alt2"><div class="kpi-title">Ticket medio</div><div class="kpi-value">${aov:,.0f}</div><div class="kpi-sub">por transacción</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi-card"><div class="kpi-title">Tiempo a Insight</div><div class="kpi-value">{tti}s</div><div class="kpi-sub">desde prompt</div></div>', unsafe_allow_html=True)
    k5.markdown(f'<div class="kpi-card alt"><div class="kpi-title">Adopción</div><div class="kpi-value">{adoption}%</div><div class="kpi-sub">usuarios activos</div></div>', unsafe_allow_html=True)
    k6.markdown(f'<div class="kpi-card alt2"><div class="kpi-title">Fill Rate / OSA</div><div class="kpi-value">{fill_rate:.1f}% / {osa:.1f}%</div><div class="kpi-sub">operación</div></div>', unsafe_allow_html=True)

st.markdown('<hr class="section"/>', unsafe_allow_html=True)

# ------------------------
# TABS DE ANÁLISIS
# ------------------------
tab_over, tab_seg, tab_coh, tab_corr, tab_anom, tab_ops, tab_reco = st.tabs(
    ["Tendencias", "Segmentación", "Cohortes/Adopción", "Correlaciones", "Anomalías", "Operación", "Recomendaciones"]
)

# --- Tendencias ---
with tab_over:
    c1, c2, c3 = st.columns([2.2, 1.5, 1.3], gap="large")

    with c1:
        ts = df_recent.groupby("date", as_index=False)["sales"].sum()
        fig_ts = px.line(ts, x="date", y="sales", markers=True, color_discrete_sequence=PALETTE)
        fig_ts.update_traces(line=dict(width=3))
        fig_ts.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_ts, use_container_width=True)

    with c2:
        tmp = df_recent.copy()
        tmp["dow"] = tmp["date"].dt.day_name()
        tmp["week"] = tmp["date"].dt.isocalendar().week.astype(int)
        hm = tmp.groupby(["dow","week"], as_index=False)["sales"].sum()
        fig_hm = px.density_heatmap(hm, x="week", y="dow", z="sales", nbinsx=12, color_continuous_scale="Turbo")
        fig_hm.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_hm, use_container_width=True)

    with c3:
        by_cat = df_recent.groupby("category", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
        fig_cat = px.treemap(by_cat, path=["category"], values="sales", color="sales", color_continuous_scale="Magma")
        fig_cat.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_cat, use_container_width=True)

# --- Segmentación ---
with tab_seg:
    c4, c5 = st.columns([2,2], gap="large")
    with c4:
        seg = df_recent.groupby(["region","category"], as_index=False)["sales"].sum()
        fig_seg = px.bar(seg, x="region", y="sales", color="category", barmode="stack",
                         color_discrete_sequence=PALETTE)
        fig_seg.update_layout(height=380, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_seg, use_container_width=True)

    with c5:
        ch = df_recent.groupby(["channel"], as_index=False)["sales"].sum()
        ch["pct"] = 100*ch["sales"]/ch["sales"].sum()
        fig_ch = px.funnel(ch.sort_values("sales", ascending=False), y="channel", x="sales",
                           color="channel", color_discrete_sequence=PALETTE)
        fig_ch.update_layout(height=380, margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
        st.plotly_chart(fig_ch, use_container_width=True)

# --- Cohortes/Adopción ---
with tab_coh:
    c6, c7 = st.columns([2,2], gap="large")
    with c6:
        dfc = df_recent.merge(customers, on="customer_id", how="left")
        dfc["signup_month"] = dfc["signup_date"].dt.to_period("M").dt.to_timestamp()
        dfc["active_month"] = dfc["date"].dt.to_period("M").dt.to_timestamp()
        cohort = dfc.groupby(["signup_month","active_month"]).agg(users=("customer_id","nunique")).reset_index()
        pivot = cohort.pivot(index="signup_month", columns="active_month", values="users").fillna(0)
        fig_cohort = px.imshow(pivot, aspect="auto", color_continuous_scale="Viridis")
        fig_cohort.update_layout(height=400, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_cohort, use_container_width=True)
    with c7:
        adop = dfc.groupby(["date"]).agg(active_users=("customer_id","nunique")).reset_index()
        fig_adop = px.area(adop, x="date", y="active_users", color_discrete_sequence=PALETTE)
        fig_adop.update_layout(height=400, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_adop, use_container_width=True)

# --- Correlaciones ---
with tab_corr:
    c8, c9 = st.columns([2,2], gap="large")
    with c8:
        sample = df_recent.sample(min(4000, len(df_recent)), random_state=7).copy()
        sample["avg_price"] = sample["sales"] / np.maximum(1, sample["units"])
        fig_sc = px.scatter(sample, x="units", y="sales",
                            color="category", size="avg_price",
                            hover_data=["region","channel"],
                            color_discrete_sequence=PALETTE)
        fig_sc.update_layout(height=380, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_sc, use_container_width=True)
    with c9:
        corr_df = sample[["sales","units","avg_price"]].corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_df.values, x=corr_df.columns, y=corr_df.columns,
            colorscale="Plasma", zmin=-1, zmax=1, showscale=True))
        fig_corr.update_layout(height=380, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_corr, use_container_width=True)

# --- Anomalías ---
with tab_anom:
    tsd = df_recent.groupby("date", as_index=False)["sales"].sum()
    mu, sd = tsd["sales"].mean(), tsd["sales"].std()
    tsd["z"] = (tsd["sales"] - mu) / (sd if sd>0 else 1)
    tsd["anomaly"] = np.where(np.abs(tsd["z"])>2.4, "Anómalo", "Normal")
    fig_anom = px.scatter(tsd, x="date", y="sales", color="anomaly",
                          color_discrete_sequence=["#777", "#FF006E"], size=np.abs(tsd["z"])+4)
    fig_anom.add_trace(go.Scatter(x=tsd["date"], y=[mu]*len(tsd), mode="lines",
                                  line=dict(dash="dash"), name="Promedio"))
    fig_anom.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10), legend_title=None)
    st.plotly_chart(fig_anom, use_container_width=True)

# --- Operación (Fill Rate / OSA / Flujo de inventario)
with tab_ops:
    c10, c11 = st.columns([1.8, 2.2], gap="large")
    with c10:
        ops = pd.DataFrame({
            "Métrica":["Fill Rate","On-Shelf Availability (OSA)","Lead Time (días)","Rotación (veces/año)"],
            "Valor":[f"{np.random.uniform(91,97):.1f}%", f"{np.random.uniform(93,98):.1f}%", f"{np.random.uniform(2.5,4.0):.1f}", f"{np.random.uniform(8,12):.1f}"]
        })
        st.dataframe(ops, hide_index=True, use_container_width=True)
        st.caption("Indicadores operativos clave del agente ADK para logística y disponibilidad en anaquel.")

    with c11:
        # Sankey simple: Abasto → Centros → Tienda/Ecom
        sources = ["Proveedores","CD Sur","CD Norte","Tienda","Ecommerce"]
        node_labels = sources
        # indices
        idx = {n:i for i,n in enumerate(node_labels)}
        links = dict(
            source=[idx["Proveedores"], idx["Proveedores"], idx["CD Sur"], idx["CD Norte"]],
            target=[idx["CD Sur"], idx["CD Norte"], idx["Tienda"], idx["Ecommerce"]],
            value=[120, 140, 150, 110]
        )
        fig_sankey = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(pad=15, thickness=16, line=dict(color="black", width=0.4), label=node_labels),
            link=dict(source=links["source"], target=links["target"], value=links["value"])
        )])
        fig_sankey.update_layout(height=420, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_sankey, use_container_width=True)

# --- Recomendaciones (Next Best Actions)
with tab_reco:
    st.markdown("### Recomendaciones del Agente ADK")
    r1, r2, r3, r4 = st.columns([1,1,1,1], gap="large")
    r1.success("Incrementar inventario de **Electrónica** en Centro +12% esta semana (picos de demanda).")
    r2.info("Campaña **Ecommerce** en Moda (bundle + cross-sell) por 72h para elevar ticket medio.")
    r3.warning("Revisar anomalías días 14/28: validar promociones, precios y existencias vs POS.")
    r4.success("Piloto de recomendación de sustitutos en Perecederos para disminuir quiebres (OSA +1.5pp).")

# ------------------------
# EXPORTS / DESCARGAS
# ------------------------
st.markdown('<hr class="section"/>', unsafe_allow_html=True)
ex1, ex2, ex3 = st.columns([1,1,1], gap="large")
with ex1:
    st.download_button("📥 CSV: ventas (ventana actual)", data=df_recent.to_csv(index=False).encode("utf-8"),
                       file_name="ventas_ventana.csv")
with ex2:
    seg_csv = df_recent.groupby(["region","category","channel"], as_index=False)["sales"].sum()
    st.download_button("📥 CSV: segmentación", data=seg_csv.to_csv(index=False).encode("utf-8"),
                       file_name="segmentacion_rc.csv")
with ex3:
    ts_csv = df_recent.groupby("date", as_index=False)["sales"].sum()
    st.download_button("📥 CSV: serie temporal", data=ts_csv.to_csv(index=False).encode("utf-8"),
                       file_name="serie_temporal.csv")
