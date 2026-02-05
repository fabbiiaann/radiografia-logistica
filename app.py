import streamlit as st
import pandas as pd
import numpy as np
import pgeocode
import pydeck as pdk
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import time
import math 

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Radiografía Logística PRO", layout="wide", page_icon="🚀")

# ==========================================
# ⚙️ ZONA DE CONFIGURACIÓN (IDs)
# ==========================================
ID_EXPEDICIONES = "12UrhIQqsFbxd-wM6kcI8_a7Hn6vxeoMJ" 
ID_FESTIVOS = "1_fJSLFsazlDMeI170QS9yPWyH7HR4mdr" 
COLOR_CORPORATIVO = "#1E3A8A"
# ==========================================

# --- ESTILOS CSS ---
st.markdown(f"""
    <style>
    .big-font {{ font-family: 'Trebuchet MS', sans-serif; font-size: 32px !important; font-weight: bold; color: {COLOR_CORPORATIVO}; }}
    .kpi-card {{ background-color: #f8f9fa; border-radius: 10px; padding: 15px; border-left: 5px solid {COLOR_CORPORATIVO}; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); text-align: center; }}
    .kpi-title {{ font-size: 14px; color: #666; text-transform: uppercase; margin-bottom: 5px; }}
    .kpi-value {{ font-size: 26px; font-weight: bold; color: {COLOR_CORPORATIVO}; }}
    .alert-box {{ background-color: #ffebee; border: 1px solid #ffcdd2; color: #b71c1c; padding: 10px; border-radius: 5px; margin-bottom: 10px; }}
    </style>
""", unsafe_allow_html=True)

# --- CARGA DE DATOS ---
@st.cache_data(show_spinner=False)
def cargar_desde_drive(file_id, tipo, cache_buster):
    if not file_id: return None
    try:
        url = ""
        if tipo == "sheet": 
            url = f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx&v={cache_buster}'
            return pd.read_excel(url)
        elif tipo == "csv": 
            url = f'https://drive.google.com/uc?id={file_id}&v={cache_buster}'
            return pd.read_csv(url, sep=None, engine='python') 
    except Exception as e:
        st.error(f"Error {tipo}: {e}")
        return None

# --- NORMALIZACIÓN ---
def limpiar_servicio(val):
    texto = str(val).upper().strip()
    if "10" in texto: return "10H"
    if "13" in texto: return "13H"
    if "19" in texto: return "19H"
    if "48" in texto: return "48H"
    if "24" in texto: return "24H"
    return texto

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830312.png", width=50)
    st.title("Control de Mando")
    
    if st.button("🔄 Actualizar Datos", type="primary"):
        st.cache_data.clear()
        st.session_state['last_update'] = time.time()
        st.rerun()
        
    if 'last_update' not in st.session_state:
        st.session_state['last_update'] = time.time()
        
    st.divider()
    st.header("1. Filtros de Análisis")
    date_container = st.container()
    service_container = st.container()
    st.divider()
    st.header("2. Simulador Económico")
    coste_penalizacion = st.number_input("Coste Penalización (€/día)", value=50, step=10)
    coste_gestion = st.number_input("Coste Gestión Incidencia (€)", value=25, step=5)
    coste_perdida = st.number_input("Pérdida por Cliente Crítico (€)", value=500, step=50)

# --- LÓGICA DE NEGOCIO ---
df = cargar_desde_drive(ID_EXPEDICIONES, "sheet", st.session_state['last_update'])
df_locales = cargar_desde_drive(ID_FESTIVOS, "csv", st.session_state['last_update'])

if df is None: st.stop()

# Festivos
dic_festivos = {}
dic_parciales = {}
if df_locales is not None:
    try:
        df_locales.columns = df_locales.columns.str.strip()
        c_fecha = next((c for c in df_locales.columns if 'fecha' in c.lower()), None)
        c_cp = next((c for c in df_locales.columns if 'cp' in c.lower()), None)
        if c_fecha and c_cp:
            df_locales[c_fecha] = pd.to_datetime(df_locales[c_fecha], dayfirst=True, errors='coerce')
            for _, row in df_locales.iterrows():
                if pd.isnull(row[c_fecha]): continue
                f_str = row[c_fecha].strftime('%Y-%m-%d')
                cp_str = str(row[c_cp]).split('.')[0].strip()
                if 'xx' in cp_str.lower(): dic_parciales[cp_str.lower().replace('xx','')] = [f_str]
                else: dic_festivos[cp_str.zfill(5)] = [f_str]
    except: pass

festivos_nac = ['2025-01-01', '2025-01-06', '2025-04-18', '2025-05-01', '2025-08-15', '2025-11-01', '2025-12-06', '2025-12-08', '2025-12-25']

# Limpieza inicial
df.columns = df.columns.str.strip()
cols_req = ['Fecha', 'Fecha Estado', 'CP Dest.', 'Artículo']
if not all(c in df.columns for c in cols_req):
    st.error(f"Faltan columnas: {cols_req}")
    st.stop()

df['Artículo'] = df['Artículo'].apply(limpiar_servicio)
df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
df['Fecha Estado'] = pd.to_datetime(df['Fecha Estado'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Fecha', 'Fecha Estado'])

with date_container:
    min_date, max_date = df['Fecha'].min(), df['Fecha'].max()
    f_inicio, f_fin = st.date_input("Rango de Fechas", [min_date, max_date])

with service_container:
    servicios = sorted(list(df['Artículo'].unique()))
    sel_servicios = st.multiselect("Servicios", servicios, default=servicios)

df_filtered = df[
    (df['Fecha'].dt.date >= f_inicio) & 
    (df['Fecha'].dt.date <= f_fin) &
    (df['Artículo'].isin(sel_servicios))
].copy()

if df_filtered.empty:
    st.warning("No hay datos con esos filtros.")
    st.stop()

# --- CÁLCULOS (CON MARGEN DE CORTESÍA) ---
def calcular_retraso(row):
    inicio = row['Fecha']
    fin = row['Fecha Estado']
    servicio = str(row['Artículo']).upper()
    cp = str(row['CP Dest.']).strip().split('.')[0].zfill(5)
    
    holidays = festivos_nac.copy()
    if cp in dic_festivos: holidays.extend(dic_festivos[cp])
    for p, fs in dic_parciales.items():
        if cp.startswith(p): holidays.extend(fs)
        
    try:
        dias = np.busday_count(inicio.date(), fin.date(), weekmask='1111100', holidays=np.array(holidays, dtype='datetime64[D]'))
    except: dias = 0
    
    plazo = 2 if ("24H" in servicio or "ESTANDAR" in servicio or "48H" in servicio) else 1
    retraso = max(0, dias - plazo)
    
    tipo_retraso = "A TIEMPO"
    if retraso > 0:
        if retraso > 9: tipo_retraso = "CRÍTICO (+9 días)"
        elif retraso >= 2: tipo_retraso = "GRAVE (+2 días)"
        else: tipo_retraso = "LEVE (+1 día)"
    elif dias == plazo:
        if "10H" in servicio:
            if fin.hour > 10 or (fin.hour == 10 and fin.minute > 5): tipo_retraso = "HORARIO"
        elif "13H" in servicio:
            if fin.hour > 13 or (fin.hour == 13 and fin.minute > 5): tipo_retraso = "HORARIO"
        
    return tipo_retraso, retraso

res = df_filtered.apply(calcular_retraso, axis=1)
df_filtered['Tipo Incidencia'] = [x[0] for x in res]
df_filtered['Días Retraso'] = [x[1] for x in res]

# --- CÁLCULO DE COSTE ---
def calc_coste(row):
    if row['Tipo Incidencia'] == "A TIEMPO": return 0
    c = coste_gestion
    c += row['Días Retraso'] * coste_penalizacion
    if "CRÍTICO" in row['Tipo Incidencia']: c += coste_perdida
    return c

df_filtered['Impacto €'] = df_filtered.apply(calc_coste, axis=1)

# SUBCONJUNTO SOLO INCIDENCIAS
df_inc = df_filtered[df_filtered['Tipo Incidencia'] != "A TIEMPO"]

# KPIs Globales
total_exp = len(df_filtered)
total_inc = len(df_inc)
otd_rate = ((total_exp - total_inc) / total_exp) * 100
total_coste = df_filtered['Impacto €'].sum()

# TIEMPO PROMEDIO
if not df_inc.empty:
    avg_delay = df_inc['Días Retraso'].mean()
else:
    avg_delay = 0

# --- PREPARACIÓN DE DATOS MENSUALES ---
df_monthly = df_filtered.copy()
df_monthly['Mes'] = df_monthly['Fecha'].dt.to_period('M')

monthly_metrics = df_monthly.groupby('Mes').agg({
    'Albarán': 'count', 
    'Tipo Incidencia': lambda x: (x != 'A TIEMPO').sum() 
}).reset_index()

monthly_metrics['OTD %'] = ((monthly_metrics['Albarán'] - monthly_metrics['Tipo Incidencia']) / monthly_metrics['Albarán']) * 100
monthly_metrics['Prev OTD'] = monthly_metrics['OTD %'].shift(1)
monthly_metrics['Variación'] = monthly_metrics['OTD %'] - monthly_metrics['Prev OTD']
monthly_metrics['Mes'] = monthly_metrics['Mes'].astype(str)

# --- INTERFAZ ---
st.markdown(f'<p class="big-font">Radiografía Logística: Informe Ejecutivo</p>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visión General", "🧠 Análisis Pareto & Causa", "💰 Impacto Económico", "📂 Datos Incidencias"])

# === TAB 1: VISIÓN GENERAL ===
with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f"""<div class="kpi-card"><div class="kpi-title">Envíos</div><div class="kpi-value">{total_exp}</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="kpi-card"><div class="kpi-title">Incidencias</div><div class="kpi-value" style="color: #d32f2f">{total_inc}</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="kpi-card"><div class="kpi-title">OTD (Calidad)</div><div class="kpi-value">{otd_rate:.1f}%</div></div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="kpi-card"><div class="kpi-title">Tiempo Medio Inc.</div><div class="kpi-value">{avg_delay:.1f} días</div></div>""", unsafe_allow_html=True)
    c5.markdown(f"""<div class="kpi-card" style="border-left: 5px solid #ff9800"><div class="kpi-title">Coste Estimado</div><div class="kpi-value" style="color: #ef6c00">{total_coste:,.0f}€</div></div>""", unsafe_allow_html=True)
    
    st.divider()
    col_izq, col_der = st.columns([1, 2])
    
    with col_izq:
        # === VELOCÍMETRO ===
        st.markdown("**Tasa Global de Incidencias**")
        tasa_inc = (total_inc/total_exp)*100 if total_exp > 0 else 0
        MAX_VAL = 24
        
        colors = ["#2E7D32", "#4CAF50", "#8BC34A", "#CDDC39", "#FFEB3B", "#FFC107", "#FF9800", "#FF5722", "#F44336", "#B71C1C"]
        steps_list = []
        step_size = MAX_VAL / len(colors)
        for i, color in enumerate(colors):
            steps_list.append({'range': [i*step_size, (i+1)*step_size], 'color': color})

        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge", value = tasa_inc,
            number = {'font': {'size': 1, 'color': "rgba(0,0,0,0)"}}, 
            gauge = {
                'axis': {'range': [0, MAX_VAL], 'tickwidth': 1, 'tickcolor': 'gray'}, 
                'bar': {'color': "rgba(0,0,0,0)"}, 
                'bgcolor': "white", 'borderwidth': 0, 'steps': steps_list,
            }
        ))
        
        angle = 180 - (tasa_inc / MAX_VAL) * 180
        theta = math.radians(angle)
        r = 0.45 
        x_head, y_head = 0.5 + r * math.cos(theta), 0.25 + r * math.sin(theta)
        width = 0.015 
        x_b1, y_b1 = 0.5 + width * math.cos(theta - math.pi/2), 0.25 + width * math.sin(theta - math.pi/2)
        x_b2, y_b2 = 0.5 + width * math.cos(theta + math.pi/2), 0.25 + width * math.sin(theta + math.pi/2)
        path = f"M {x_b1} {y_b1} L {x_head} {y_head} L {x_b2} {y_b2} Z"
        
        fig_gauge.add_shape(type="path", path=path, fillcolor="black", line_color="black")
        fig_gauge.add_shape(type="circle", x0=0.48, y0=0.23, x1=0.52, y1=0.27, fillcolor="#333", line_color="#333")
        fig_gauge.add_annotation(x=0.5, y=0.05, text=f"{tasa_inc:.2f}%", showarrow=False, font=dict(size=40, color=COLOR_CORPORATIVO, family="Arial Black"))
        fig_gauge.add_annotation(x=0.5, y=0.75, ax=0, ay=-40, text="Media Sector (12%)", showarrow=True, arrowhead=2, font=dict(size=12, color="black"))
        fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_der:
        st.subheader("📅 Comparativa Mensual")
        st.dataframe(
            monthly_metrics[['Mes', 'Albarán', 'Tipo Incidencia', 'OTD %', 'Variación']],
            use_container_width=True,
            column_config={
                "Mes": "Periodo",
                "Albarán": "📦 Expediciones",
                "Tipo Incidencia": "⚠️ Incidencias",
                "OTD %": st.column_config.NumberColumn("📊 Calidad (OTD)", format="%.1f%%"),
                "Variación": st.column_config.NumberColumn("Vs Mes Anterior", format="%.1f%%")
            },
            hide_index=True
        )
        st.markdown("**Tendencia Visual**")
        fig_line = px.line(monthly_metrics, x='Mes', y='OTD %', markers=True, color_discrete_sequence=[COLOR_CORPORATIVO])
        fig_line.update_layout(height=200, margin=dict(t=10, b=10))
        st.plotly_chart(fig_line, use_container_width=True)

# === TAB 2: ANÁLISIS ===
with tab2:
    if not df_inc.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 📊 Gravedad")
            df_tipo = df_inc['Tipo Incidencia'].value_counts().reset_index()
            df_tipo.columns = ['Tipo', 'Cantidad']
            fig_bar_h = px.bar(df_tipo, x='Cantidad', y='Tipo', orientation='h', text='Cantidad', color='Tipo', color_discrete_sequence=px.colors.sequential.Reds_r)
            st.plotly_chart(fig_bar_h, use_container_width=True)
        with c2:
            st.markdown("### 📈 Pareto (80/20)")
            # === CÓDIGO NUEVO PARETO ===
            # Agrupar por CP Destino en incidencias
            incidencias_cp = df_inc.groupby('CP Dest.').size().sort_values(ascending=False).head(10)
            porcentaje_acum = (incidencias_cp.cumsum() / incidencias_cp.sum() * 100)
            
            # Gráfico combinado
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=incidencias_cp.index.astype(str), y=incidencias_cp.values, name="Incidencias", marker_color=COLOR_CORPORATIVO), secondary_y=False)
            fig.add_trace(go.Scatter(x=incidencias_cp.index.astype(str), y=porcentaje_acum.values, name="% Acumulado", mode='lines+markers', marker_color="red"), secondary_y=True)
            fig.add_hline(y=80, line_dash="dash", line_color="red", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        try:
            st.markdown("### 🌍 Mapa de Incidencias")
            df_inc['CP_Clean'] = df_inc['CP Dest.'].astype(str).str.split('.').str[0].str.strip().str.zfill(5)
            nomi = pgeocode.Nominatim('es')
            geo = nomi.query_postal_code(df_inc['CP_Clean'].tolist())
            df_inc['lat'] = geo['latitude'].values
            df_inc['lon'] = geo['longitude'].values
            map_data = df_inc.dropna(subset=['lat', 'lon'])
            if not map_data.empty:
                layer = pdk.Layer("HeatmapLayer", data=map_data, get_position=["lon", "lat"], opacity=0.8, get_weight=1, radiusPixels=40, intensity=1.5)
                view = pdk.ViewState(latitude=40.4167, longitude=-3.7033, zoom=5)
                st.pydeck_chart(pdk.Deck(map_style='light', initial_view_state=view, layers=[layer]))
        except: pass

# === TAB 3: ECONÓMICO ===
with tab3:
    st.markdown("### 💸 Proyección Económica")
    coste_proyectado_anual = total_inc * avg_delay * coste_penalizacion
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Tiempo Medio Retraso", f"{avg_delay:.1f} días")
    c2.metric("Coste Real (Calculado)", f"{total_coste:,.0f}€")
    c3.metric("Proyección Anual", f"{coste_proyectado_anual:,.0f}€")
    
    st.divider()
    coste_por_tipo = df_inc.groupby('Tipo Incidencia')['Impacto €'].sum().reset_index()
    fig_pie = px.pie(coste_por_tipo, values='Impacto €', names='Tipo Incidencia', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_pie, use_container_width=True)

# === TAB 4: DATOS (SOLO INCIDENCIAS) ===
with tab4:
    st.markdown("### 🔍 Datos Detallados (Solo Incidencias)")
    busqueda = st.text_input("Buscar:", placeholder="Albarán, CP...")
    
    if busqueda:
        mask = df_inc.astype(str).apply(lambda x: x.str.contains(busqueda, case=False)).any(axis=1)
        df_display = df_inc[mask]
    else: 
        df_display = df_inc
        
    st.dataframe(df_display, use_container_width=True)
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_display.to_excel(writer, sheet_name='Incidencias', index=False)
    st.download_button("📥 Descargar Excel (Solo Incidencias)", buffer, "informe_incidencias.xlsx", "application/vnd.ms-excel", type="primary")
