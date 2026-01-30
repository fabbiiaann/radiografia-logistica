import streamlit as st
import pandas as pd
import numpy as np
import pgeocode
import pydeck as pdk
import datetime

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Radiografía Logística (Drive)", layout="wide", page_icon="☁️")

# ==========================================
# ⚙️ ZONA DE CONFIGURACIÓN (TUS IDs)
# ==========================================
ID_EXPEDICIONES = "12UrhIQqsFbxd-wM6kcI8_a7Hn6vxeoMJ" 
ID_FESTIVOS = "1_fJSLFsazlDMeI170QS9yPWyH7HR4mdr" 
# ==========================================

# --- ESTÉTICA ---
st.markdown("""
    <style>
    .big-font { font-family: 'Trebuchet MS', sans-serif; font-size: 45px !important; font-weight: bold; color: #1E3A8A; text-align: center; }
    .subtitle { font-family: 'Helvetica', sans-serif; font-size: 18px; color: #4B5563; text-align: center; }
    .authors { font-family: 'Helvetica', sans-serif; font-size: 14px; color: #6B7280; text-align: center; font-style: italic; margin-bottom: 20px; }
    hr { margin-top: 1rem; margin-bottom: 2rem; border: 0; border-top: 1px solid rgba(0,0,0,.1); }
    </style>
    
    <p class="big-font">Radiografía Logística: Análisis Drive</p>
    <p class="subtitle">Conexión en tiempo real con Google Drive</p>
    <div class="authors">Desarrollado por: <b>Fabián Cruz y Nerea Mallo</b></div>
    <hr>
""", unsafe_allow_html=True)

# --- CARGA MULTIFORMATO ---
@st.cache_data(ttl=60) 
def cargar_desde_drive(file_id, tipo):
    if not file_id: return None
    try:
        url = ""
        if tipo == "sheet":
            url = f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx'
            return pd.read_excel(url)
        elif tipo == "csv":
            url = f'https://drive.google.com/uc?id={file_id}'
            return pd.read_csv(url, sep=None, engine='python') 
    except Exception as e:
        st.error(f"Error cargando {tipo} (ID: {file_id}): {e}")
        return None

# --- BOTÓN DE ACTUALIZAR ---
col_btn, col_txt = st.columns([1, 5])
with col_btn:
    if st.button("🔄 Actualizar Datos"):
        st.cache_data.clear()
        st.rerun()
with col_txt:
    st.caption("Pulsa para refrescar los datos desde la nube.")

# --- CARGA DE DATOS ---
with st.spinner('Conectando con la nube...'):
    df = cargar_desde_drive(ID_EXPEDICIONES, "sheet")
    df_locales = cargar_desde_drive(ID_FESTIVOS, "csv")

if df_locales is None:
    st.warning("⚠️ No se ha cargado el archivo de Festivos.")
else:
    st.success(f"✅ Festivos cargados: {len(df_locales)} registros.")

if df is None:
    st.error("❌ Error de conexión con Expediciones.")
    st.stop()

# --- PROCESAMIENTO ---
diccionario_festivos_locales = {}
diccionario_festivos_parciales = {}

if df_locales is not None:
    try:
        df_locales.columns = df_locales.columns.str.strip()
        col_fecha = next((c for c in df_locales.columns if 'fecha' in c.lower()), None)
        col_cp = next((c for c in df_locales.columns if 'cp' in c.lower() or 'postal' in c.lower()), None)

        if col_fecha and col_cp:
            df_locales[col_fecha] = pd.to_datetime(df_locales[col_fecha], dayfirst=True, errors='coerce')
            for index, row in df_locales.iterrows():
                if pd.isnull(row[col_fecha]): continue
                fecha_str = row[col_fecha].strftime('%Y-%m-%d')
                cp_raw = str(row[col_cp]).split('.')[0].strip()
                
                if 'xx' in cp_raw.lower():
                    prefix = cp_raw.lower().replace('xx', '')
                    if prefix not in diccionario_festivos_parciales: diccionario_festivos_parciales[prefix] = []
                    diccionario_festivos_parciales[prefix].append(fecha_str)
                else:
                    cp_full = cp_raw.zfill(5)
                    if cp_full not in diccionario_festivos_locales: diccionario_festivos_locales[cp_full] = []
                    diccionario_festivos_locales[cp_full].append(fecha_str)
    except: pass

festivos_nacionales = ['2025-01-01', '2025-01-06', '2025-04-18', '2025-05-01', '2025-08-15', '2025-11-01', '2025-12-06', '2025-12-08', '2025-12-25']

df.columns = df.columns.str.strip()
cols_necesarias = ['Fecha', 'Fecha Estado', 'CP Dest.', 'Artículo']

if all(col in df.columns for col in cols_necesarias):
    df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
    df['Fecha Estado'] = pd.to_datetime(df['Fecha Estado'], dayfirst=True, errors='coerce')
    df_clean = df.dropna(subset=['Fecha', 'Fecha Estado']).copy()
    
    def analizar(row):
        inicio = row['Fecha']
        fin = row['Fecha Estado']
        servicio = str(row['Artículo']).upper()
        
        festivos_fila = festivos_nacionales.copy()
        raw_cp = str(row['CP Dest.']).strip().split('.')[0].zfill(5)
        
        if raw_cp in diccionario_festivos_locales: festivos_fila.extend(diccionario_festivos_locales[raw_cp])
        for prefix, fechas in diccionario_festivos_parciales.items():
            if raw_cp.startswith(prefix): festivos_fila.extend(fechas)

        try:
            dias = np.busday_count(inicio.date(), fin.date(), weekmask='1111100', holidays=np.array(festivos_fila, dtype='datetime64[D]'))
        except: dias = 0

        res = "A Tiempo"
        if "10H" in servicio:
            if dias > 1: res = f"RETRASO (+{dias} días)"
            elif dias == 1 and fin.hour >= 10: res = "RETRASO HORARIO (>10:00)"
        elif "13H" in servicio:
            if dias > 1: res = f"RETRASO (+{dias} días)"
            elif dias == 1 and fin.hour >= 13: res = "RETRASO HORARIO (>13:00)"
        elif "19H" in servicio:
             if dias > 1: res = f"RETRASO (+{dias-1} días)"
        else: 
            if dias > 2: res = f"RETRASO (+{dias-2} días)"
        return res

    df_clean['Resultado Auditoría'] = df_clean.apply(analizar, axis=1)
    
    st.divider()
    df_inc = df_clean[df_clean['Resultado Auditoría'].str.contains("RETRASO")]
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Expediciones", len(df_clean))
    k2.metric("Incidencias", len(df_inc), delta_color="inverse")
    if len(df_clean) > 0:
        perc = (len(df_inc)/len(df_clean))*100
        k3.metric("% Error Global", f"{perc:.1f}%")
    
    st.subheader("🔎 Detalle de Incidencias")
    st.dataframe(df_inc[['Fecha', 'Albarán', 'CP Dest.', 'Artículo', 'Resultado Auditoría']], use_container_width=True)
    
    # --- MAPA ARREGLADO (V30) ---
    if not df_inc.empty:
        st.divider()
        st.subheader("🔥 Mapa de Calor de Incidencias (Destino)")
        
        try:
            # 1. Limpieza y Geolocalización (USANDO CP DESTINO)
            df_inc['CP_Clean'] = df_inc['CP Dest.'].astype(str).str.split('.').str[0].str.strip().str.zfill(5)
            
            nomi = pgeocode.Nominatim('es')
            geo = nomi.query_postal_code(df_inc['CP_Clean'].tolist())
            
            df_inc['lat'] = geo['latitude'].values
            df_inc['lon'] = geo['longitude'].values
            
            map_data = df_inc.dropna(subset=['lat', 'lon'])
            
            if not map_data.empty:
                # 2. Configuración Visual
                layer = pdk.Layer(
                    "HeatmapLayer",
                    data=map_data,
                    get_position=["lon", "lat"],
                    opacity=0.8,
                    get_weight=1,
                    radiusPixels=50
                )
                
                # Vista centrada en España
                view_state = pdk.ViewState(
                    latitude=40.4167, 
                    longitude=-3.7033, 
                    zoom=5, 
                    pitch=0
                )
                
                # 3. CAMBIO CLAVE: Estilo 'light' para ver nombres y carreteras
                st.pydeck_chart(pdk.Deck(
                    map_style='light',  # <--- ESTO ARREGLA EL FONDO BLANCO
                    initial_view_state=view_state,
                    layers=[layer],
                    tooltip={"text": "Zona con incidencias"}
                ))
                st.caption(f"Visualizando {len(map_data)} incidencias en destino.")
            else:
                st.warning("No se pudieron geolocalizar los códigos postales.")
        except Exception as e:
            st.warning(f"Error mapa: {e}")

else:
    st.error(f"Faltan columnas en el Excel. Necesarias: {cols_necesarias}")