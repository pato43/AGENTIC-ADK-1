# app.py
# Agente ADK — Retail MX (estilo Walmart México)
# Interfaz moderna, tema oscuro, tablero horizontal y explicación automática de hallazgos.

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# =========================
# CONFIGURACIÓN Y ESTILO
# =========================
st.set_page_config(
    page_title="Agente ADK — Retail MX",
    layout="wide",
    page_icon="🛒"
)

# Paleta (inspirada en Walmart de México)
WALMART_BLUE = "#0071CE"
WALMART_YELLOW = "#FFC220"
BG_DARK = "#0E1117"
CARD_DARK = "#141922"
TEXT = "#F5F7FA"
MUTED = "#B8C2CC"

# Plotly: paleta vibrante + template oscuro
PALETTE = px.colors.qualitative.Prism
px.defaults.template = "plotly_dark"

# Tipografías y estilos globales
# Estilos principales (Walmart México)
custom_css = """
<style>
html, body, [class*="css"] {
  background-color: #0E1117;
  color: #F5F7FA;
  font-family: 'Inter', sans-serif;
}
.block-container {padding: 0.5rem 1rem 1rem 1rem;}
header {visibility: hidden;}
.brand {display:flex; align-items:center; gap:14px; padding:10px 0;}
.brand-title {font-weight:900; font-size:1.2rem; letter-spacing:.2px; color:#F5F7FA;}
.badge {
  font-size:.75rem; font-weight:800; padding:4px 10px; border-radius:8px; color:#0B1220;
  background: linear-gradient(135deg, #FFC220, #FFE180);
}
.pill {
  display:inline-block; padding:6px 12px; border-radius:999px; font-size:.8rem;
  background: #0071CE; color:#00142E; margin:0 8px 8px 0; font-weight:800;
  box-shadow: 0 4px 16px rgba(0,0,0,0.25);
}
.pill.muted { background: #21324A; color: #F5F7FA; }
.kpi-card {
  background: #141922; border: 1px solid #1F2633;
  border-radius: 16px; padding: 14px 16px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}
.kpi-title { font-size: .78rem; color:#B8C2CC; font-weight:600; letter-spacing:.2px; }
.kpi-value { font-size: 1.5rem; font-weight:900; color:#F5F7FA; }
.kpi-sub { font-size: .80rem; color:#B8C2CC; font-weight:600; }
.section-title { font-size: 1.05rem; font-weight:900; letter-spacing:.2px; margin: 6px 0 8px 0; }
hr.section {
  border: none; height: 1px;
  background: linear-gradient(90deg, #0071CE, #FFC220);
  margin: 8px 0 16px;
}
.card {
  background: #141922; border: 1px solid #1F2633;
  border-radius: 14px; padding: 12px 14px;
}
.stTextInput > div > div > input {
  background: #0F1420 !important; color: #F5F7FA !important;
  border: 1px solid #20283A !important; border-radius: 10px;
}
.stButton > button {
  background: #0071CE; color: #00142E;
  font-weight:900; border-radius: 10px; border: none;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# =========================
# DATOS SINTÉTICOS (Retail MX)
# =========================
@st.cache_data(show_spinner=False)
def make_data(seed=23, days=420):
    rng = np.random.default_rng(seed)
    end = datetime.today().date()
    dates = pd.date_range(end - timedelta(days=days-1), end, freq="D")

    regions = ["Centro", "Norte", "Occidente", "Sureste"]
    cats = ["Abarrotes", "Perecederos", "Hogar", "Electrónica", "Farmacia", "Moda"]
    channels = ["Tienda", "Ecommerce", "Pickup", "Mayorista"]

    rows = []
    for d in dates:
        for r in regions:
            for c in cats:
                base = 600 + 4.5*(d.toordinal() % 200)
                weekly = 150*np.sin((d.dayofyear/7)*2*np.pi)
                season = 180*np.sin((d.dayofyear/30)*2*np.pi)
                region_b = {"Centro":1.18, "Norte":1.04, "Occidente":0.97, "Sureste":0.92}[r]
                cat_b = {"Abarrotes":1.15,"Perecederos":1.05,"Hogar":1.0,"Electrónica":1.35,"Farmacia":0.95,"Moda":1.1}[c]
                noise = rng.normal(0, 90)
                sales = max(60, (base + weekly + season)*region_b*cat_b + noise)
                units = max(1, int(sales/28 + rng.normal(0, 5)))
                ch = rng.choice(channels, p=[0.48, 0.32, 0.12, 0.08])
                rows.append([d, r, c, ch, round(float(sales),2), units])

    df = pd.DataFrame(rows, columns=["date","region","category","channel","sales","units"])
    # Cohortes de clientes
    n_customers = 2200
    cust_ids = np.arange(1, n_customers+1)
    signup = pd.to_datetime(rng.choice(pd.date_range(dates.min()-pd.Timedelta(days=150), dates.min(), freq="D"),
                                       size=n_customers))
    customers = pd.DataFrame({"customer_id": cust_ids, "signup_date": signup})
    df["customer_id"] = rng.choice(cust_ids, size=len(df))
    return df, customers

df, customers = make_data()

# =========================
# ESTADO DE UI
# =========================
if "show_dashboard" not in st.session_state:
    st.session_state.show_dashboard = False
if "query" not in st.session_state:
    st.session_state.query = ""

# =========================
# PANTALLA INICIAL (vacía hasta que tecleas y enter)
# =========================
if not st.session_state.show_dashboard:
    head = st.container()
    with head:
        c0, c1, c2 = st.columns([2.2, 2, 1], gap="large")
        with c0:
            st.markdown(
                f'<div class="brand"><div class="brand-title">🧠 Agente ADK — Retail Omnicanal México</div>'
                f'<div class="badge">Ventas • Categorías • Regiones • Canales</div></div>',
                unsafe_allow_html=True
            )
            st.caption("Interfaz conversacional para inteligencia de negocio en retail de gran escala.")
        with c1:
            user_msg = st.text_input("Escribe tu consulta", value="", placeholder="Ej. Ventas del último trimestre por categoría y región")
        with c2:
            run = st.button("▶ Ejecutar", use_container_width=True, type="primary")

    # Cuando presionas Enter o el botón:
    if (user_msg and st.session_state.query != user_msg) or run:
        st.session_state.query = user_msg or st.session_state.query
        # Animación/píldoras de proceso
        st.markdown('<hr class="section"/>', unsafe_allow_html=True)
        st.markdown(f'<span class="pill">Planner: entendiendo la intención</span>', unsafe_allow_html=True); time.sleep(0.7)
        st.markdown(f'<span class="pill">Herramienta: BigQuery</span>', unsafe_allow_html=True); time.sleep(0.7)
        st.markdown(f'<span class="pill">SQL y agregaciones</span>', unsafe_allow_html=True); time.sleep(0.7)
        st.markdown(f'<span class="pill">Visualizaciones</span>', unsafe_allow_html=True); time.sleep(0.7)
        st.markdown(f'<span class="pill">Composición de insights</span>', unsafe_allow_html=True); time.sleep(0.6)
        st.session_state.show_dashboard = True
        st.rerun()

# =========================
# DASHBOARD (después de Enter)
# =========================
if st.session_state.show_dashboard:
    # Encabezado + filtros
    top = st.container()
    with top:
        cA, cB, cC = st.columns([2.5, 2, 1.2], gap="large")
        with cA:
            st.markdown(
                f'<div class="brand"><div class="brand-title">🧠 Agente ADK — Retail Omnicanal México</div>'
                f'<div class="badge">Consulta activa</div></div>',
                unsafe_allow_html=True
            )
            st.caption(st.session_state.query or "Consulta no especificada")
        with cB:
            region_sel = st.multiselect("Filtrar región", sorted(df["region"].unique()), default=None)
            cat_sel = st.multiselect("Filtrar categoría", sorted(df["category"].unique()), default=None)
        with cC:
            ch_sel = st.multiselect("Filtrar canal", sorted(df["channel"].unique()), default=None)
            horizon_days = st.slider("Ventana (días)", 60, 180, 120, 10)

    # Aplicar filtros
    df_viz = df.copy()
    if region_sel: df_viz = df_viz[df_viz["region"].isin(region_sel)]
    if cat_sel: df_viz = df_viz[df_viz["category"].isin(cat_sel)]
    if ch_sel: df_viz = df_viz[df_viz["channel"].isin(ch_sel)]

    cut_date = df_viz["date"].max() - pd.Timedelta(days=horizon_days-1)
    df_recent = df_viz[df_viz["date"] >= cut_date].copy()

    # KPIs
    def compute_kpis(dfr):
        total_sales = dfr["sales"].sum()
        total_units = int(dfr["units"].sum())
        aov = total_sales / max(1, total_units)
        tti = np.random.randint(6, 12)     # Time To Insight
        adoption = np.random.randint(68, 88)
        fill_rate = np.random.uniform(91, 97)
        osa = np.random.uniform(93, 98)
        return total_sales, total_units, aov, tti, adoption, fill_rate, osa

    tot_sales, tot_units, aov, tti, adoption, fill_rate, osa = compute_kpis(df_recent)

    c1, c2, c3, c4, c5, c6 = st.columns([1,1,1,1,1,1], gap="small")
    with c1:
        st.markdown('<div class="kpi-card"><div class="kpi-title">Ventas</div>'
                    f'<div class="kpi-value">${tot_sales:,.0f}</div>'
                    f'<div class="kpi-sub">MXN · {horizon_days}d</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="kpi-card"><div class="kpi-title">Unidades</div>'
                    f'<div class="kpi-value">{tot_units:,}</div>'
                    f'<div class="kpi-sub">Últimos {horizon_days}d</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="kpi-card"><div class="kpi-title">Ticket medio</div>'
                    f'<div class="kpi-value">${aov:,.2f}</div>'
                    f'<div class="kpi-sub">por unidad</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="kpi-card"><div class="kpi-title">Tiempo a Insight</div>'
                    f'<div class="kpi-value">{tti}s</div>'
                    f'<div class="kpi-sub">desde prompt</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="kpi-card"><div class="kpi-title">Adopción</div>'
                    f'<div class="kpi-value">{adoption}%</div>'
                    f'<div class="kpi-sub">usuarios activos</div></div>', unsafe_allow_html=True)
    with c6:
        st.markdown('<div class="kpi-card"><div class="kpi-title">Fill Rate / OSA</div>'
                    f'<div class="kpi-value">{fill_rate:.1f}% / {osa:.1f}%</div>'
                    f'<div class="kpi-sub">operación</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="section"/>', unsafe_allow_html=True)

    # TABS
    tab_over, tab_seg, tab_coh, tab_corr, tab_anom, tab_ops, tab_reco, tab_sql = st.tabs(
        ["Tendencias", "Segmentación", "Cohortes/Adopción", "Correlaciones", "Anomalías", "Operación", "Recomendaciones", "SQL"]
    )

    # --- Tendencias ---
    with tab_over:
        cA, cB, cC = st.columns([2.2, 1.5, 1.3], gap="large")
        with cA:
            ts = df_recent.groupby("date", as_index=False)["sales"].sum()
            fig_ts = px.line(ts, x="date", y="sales", markers=True, color_discrete_sequence=[WALMART_YELLOW])
            fig_ts.update_traces(line=dict(width=3))
            fig_ts.update_layout(height=380, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_ts, use_container_width=True)
        with cB:
            tmp = df_recent.copy()
            tmp["dow"] = tmp["date"].dt.day_name()
            tmp["week"] = tmp["date"].dt.isocalendar().week.astype(int)
            hm = tmp.groupby(["dow","week"], as_index=False)["sales"].sum()
            fig_hm = px.density_heatmap(hm, x="week", y="dow", z="sales", nbinsx=12, color_continuous_scale="Turbo")
            fig_hm.update_layout(height=380, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_hm, use_container_width=True)
        with cC:
            by_cat = df_recent.groupby("category", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
            fig_cat = px.treemap(by_cat, path=["category"], values="sales",
                                 color="sales", color_continuous_scale="Magma")
            fig_cat.update_layout(height=380, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_cat, use_container_width=True)

    # --- Segmentación ---
    with tab_seg:
        cD, cE = st.columns([2,2], gap="large")
        with cD:
            seg = df_recent.groupby(["region","category"], as_index=False)["sales"].sum()
            fig_seg = px.bar(seg, x="region", y="sales", color="category",
                             barmode="stack", color_discrete_sequence=PALETTE)
            fig_seg.update_layout(height=400, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_seg, use_container_width=True)
        with cE:
            ch = df_recent.groupby(["channel"], as_index=False)["sales"].sum()
            ch["pct"] = 100*ch["sales"]/ch["sales"].sum()
            fig_ch = px.funnel(ch.sort_values("sales", ascending=False), y="channel", x="sales",
                               color="channel", color_discrete_sequence=PALETTE)
            fig_ch.update_layout(height=400, margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
            st.plotly_chart(fig_ch, use_container_width=True)

    # --- Cohortes / Adopción ---
    with tab_coh:
        cF, cG = st.columns([2,2], gap="large")
        with cF:
            dfc = df_recent.merge(customers, on="customer_id", how="left")
            dfc["signup_month"] = dfc["signup_date"].dt.to_period("M").dt.to_timestamp()
            dfc["active_month"] = dfc["date"].dt.to_period("M").dt.to_timestamp()
            cohort = dfc.groupby(["signup_month","active_month"]).agg(users=("customer_id","nunique")).reset_index()
            pivot = cohort.pivot(index="signup_month", columns="active_month", values="users").fillna(0)
            fig_cohort = px.imshow(pivot, aspect="auto", color_continuous_scale="Viridis")
            fig_cohort.update_layout(height=420, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_cohort, use_container_width=True)
        with cG:
            adop = dfc.groupby(["date"]).agg(active_users=("customer_id","nunique")).reset_index()
            fig_adop = px.area(adop, x="date", y="active_users", color_discrete_sequence=[WALMART_BLUE])
            fig_adop.update_layout(height=420, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_adop, use_container_width=True)

    # --- Correlaciones ---
    with tab_corr:
        cH, cI = st.columns([2,2], gap="large")
        with cH:
            sample = df_recent.sample(min(4000, len(df_recent)), random_state=7).copy()
            sample["avg_price"] = sample["sales"] / np.maximum(1, sample["units"])
            fig_sc = px.scatter(sample, x="units", y="sales",
                                color="category", size="avg_price",
                                hover_data=["region","channel"],
                                color_discrete_sequence=PALETTE)
            fig_sc.update_layout(height=400, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_sc, use_container_width=True)
        with cI:
            corr_df = sample[["sales","units","avg_price"]].corr()
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_df.values, x=corr_df.columns, y=corr_df.columns,
                colorscale="Plasma", zmin=-1, zmax=1, showscale=True))
            fig_corr.update_layout(height=400, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_corr, use_container_width=True)

    # --- Anomalías ---
    with tab_anom:
        tsd = df_recent.groupby("date", as_index=False)["sales"].sum()
        mu, sd = tsd["sales"].mean(), tsd["sales"].std()
        tsd["z"] = (tsd["sales"] - mu) / (sd if sd>0 else 1)
        tsd["anomaly"] = np.where(np.abs(tsd["z"])>2.4, "Anómalo", "Normal")
        fig_anom = px.scatter(tsd, x="date", y="sales", color="anomaly",
                              color_discrete_sequence=["#6B7280", "#FF006E"], size=np.abs(tsd["z"])+4)
        fig_anom.add_trace(go.Scatter(x=tsd["date"], y=[mu]*len(tsd), mode="lines",
                                      line=dict(dash="dash", color="#94A3B8"), name="Promedio"))
        fig_anom.update_layout(height=380, margin=dict(l=10,r=10,t=10,b=10), legend_title=None)
        st.plotly_chart(fig_anom, use_container_width=True)

    # --- Operación ---
    with tab_ops:
        cJ, cK = st.columns([1.5, 2.5], gap="large")
        with cJ:
            ops = pd.DataFrame({
                "Métrica":["Fill Rate","On-Shelf Availability (OSA)","Lead Time (días)","Rotación (veces/año)"],
                "Valor":[f"{np.random.uniform(91,97):.1f}%", f"{np.random.uniform(93,98):.1f}%", f"{np.random.uniform(2.5,4.0):.1f}", f"{np.random.uniform(8,12):.1f}"]
            })
            st.dataframe(ops, hide_index=True, use_container_width=True)
            st.caption("Indicadores operativos clave para logística y disponibilidad.")
        with cK:
            sources = ["Proveedores","CD Sur","CD Norte","Tienda","Ecommerce"]
            idx = {n:i for i,n in enumerate(sources)}
            links = dict(
                source=[idx["Proveedores"], idx["Proveedores"], idx["CD Sur"], idx["CD Norte"]],
                target=[idx["CD Sur"], idx["CD Norte"], idx["Tienda"], idx["Ecommerce"]],
                value=[120, 140, 150, 110]
            )
            fig_sankey = go.Figure(data=[go.Sankey(
                arrangement="snap",
                node=dict(pad=15, thickness=18, line=dict(color="black", width=0.4), label=sources),
                link=dict(source=links["source"], target=links["target"], value=links["value"])
            )])
            fig_sankey.update_layout(height=440, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_sankey, use_container_width=True)

    # --- Recomendaciones ---
    with tab_reco:
        st.markdown("### Recomendaciones del Agente ADK")
        r1, r2, r3, r4 = st.columns([1,1,1,1], gap="large")
        r1.success("Incrementar inventario de **Electrónica** en Centro +12% esta semana (picos esperados).")
        r2.info("Campaña **Ecommerce** en Moda (bundle + cross-sell) por 72h para elevar ticket medio.")
        r3.warning("Revisar anomalías días 14/28: validar promociones, precios y existencias vs POS.")
        r4.success("Piloto de sustitutos en Perecederos para disminuir quiebres (OSA +1.5 pp).")

    # --- SQL (referencial para BI) ---
    with tab_sql:
        sql_text = """
-- NL→SQL • Retail MX (BigQuery)
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
        st.code(sql_text, language="sql")

    st.markdown('<hr class="section"/>', unsafe_allow_html=True)

    # =========================
    # EXPLICACIÓN AUTOMÁTICA DE HALLAZGOS (narrativa)
    # =========================
    # Datos para narrativa
    by_cat = df_recent.groupby("category", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    top_cat = by_cat.iloc[0]["category"] if len(by_cat) else "—"
    by_reg = df_recent.groupby("region", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    top_reg = by_reg.iloc[0]["region"] if len(by_reg) else "—"
    by_ch = df_recent.groupby("channel", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    top_ch = by_ch.iloc[0]["channel"] if len(by_ch) else "—"

    st.markdown('<div class="section-title">Explicación del Agente</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="card">
<p><strong>Tendencias:</strong> El comportamiento de ventas en la ventana analizada muestra una trayectoria estable con estacionalidad semanal marcada. 
Los picos recurrentes sugieren influencia de jornadas de alto tráfico y campañas periódicas.</p>

<p><strong>Segmentación:</strong> <em>{top_cat}</em> lidera en ingresos, lo que recomienda reforzar disponibilidad, 
pricing y promociones dirigidas a su audiencia. A nivel geográfico, <em>{top_reg}</em> concentra la mayor contribución; 
ajustar asignaciones de inventario y cobertura logística en esta región puede maximizar el efecto.</p>

<p><strong>Canales:</strong> El canal predominante es <em>{top_ch}</em>. Existe espacio para acelerar el crecimiento en los demás canales 
mediante estrategias de conversión (UX, surtido y precios dinámicos) y campañas con mensajes por segmento.</p>

<p><strong>Cohortes y adopción:</strong> La adopción estimada se ubica en <strong>{adoption}%</strong>. 
Las cohortes recientes mantienen tracción inicial; enfocar onboarding y activación durante las primeras 2–3 semanas 
mejora la retención y el valor de vida del cliente.</p>

<p><strong>Anomalías:</strong> Se detectaron puntos fuera de umbral (|z| &gt; 2.4). Verificar correlación con promociones, cambios de precio 
y quiebres de inventario. Estabilizar la disponibilidad (OSA) y el <em>fill rate</em> reduce la volatilidad.</p>

<p><strong>Operación:</strong> Métricas operativas actuales: <em>Fill Rate</em> ≈ <strong>{fill_rate:.1f}%</strong>, OSA ≈ <strong>{osa:.1f}%</strong>. 
Se recomienda priorizar reposición en categorías líderes y programar reabastos de alta frecuencia en tiendas con mayor rotación.</p>

<p><strong>Acciones sugeridas:</strong> (1) Refuerzo de inventario en {top_cat} para {top_reg} (+10–12% esta semana). 
(2) Campaña dirigida en canales no líderes con bundles y recomendación de complementarios. 
(3) Revisión puntual de eventos anómalos y validación POS/precios para evitar pérdidas por rotura o error.</p>
</div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # EXPORTACIONES
    # =========================
    st.markdown('<div class="section-title">Exportaciones</div>', unsafe_allow_html=True)
    ex1, ex2, ex3 = st.columns([1,1,1], gap="large")
    with ex1:
        st.download_button(
            "📥 CSV: ventas (ventana actual)",
            data=df_recent.to_csv(index=False).encode("utf-8"),
            file_name="ventas_ventana.csv"
        )
    with ex2:
        seg_csv = df_recent.groupby(["region","category","channel"], as_index=False)["sales"].sum()
        st.download_button(
            "📥 CSV: segmentación",
            data=seg_csv.to_csv(index=False).encode("utf-8"),
            file_name="segmentacion_rc.csv"
        )
    with ex3:
        ts_csv = df_recent.groupby("date", as_index=False)["sales"].sum()
        st.download_button(
            "📥 CSV: serie temporal",
            data=ts_csv.to_csv(index=False).encode("utf-8"),
            file_name="serie_temporal.csv"
        )
