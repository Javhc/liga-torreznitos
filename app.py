import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px

# Configuración de la página
st.set_page_config(page_title="Liga Torreznitos", layout="wide")

st.title("🏀 Estadísticas Liga Torreznitos")
st.markdown("Análisis y simulación de Playoffs (50.000 iteraciones) basados en rendimiento histórico.")

# --- CONSTANTES FIJAS DE LA LIGA ---
NUM_SIMULACIONES = 50000
PUESTOS_PLAYOFF = 8
MARGEN_EMPATE = 1.5
JORNADAS_JUGADAS = 26

# 1. ESTADO ACTUAL DE LA CLASIFICACIÓN (Jornada 26)
clasificacion_actual = {
    "Strava Palencia": {"V": 20, "PTS": 2740},
    "Foster's Rivas Sureste": {"V": 18, "PTS": 2274},
    "CB Perales Nuit": {"V": 17, "PTS": 2259},
    "Mahle Baltanás": {"V": 17, "PTS": 2252},
    "Ron Negrita Badalona": {"V": 16, "PTS": 2352},
    "Chupa-Chups Magaluf": {"V": 15, "PTS": 2324},
    "Disney Burgos": {"V": 15, "PTS": 2185},
    "CUPRA Lantadilla": {"V": 15, "PTS": 2165},
    "San Miguel Carabanchel": {"V": 15, "PTS": 2100},
    "Banco Sinentender": {"V": 14, "PTS": 2109},
    "Oliva Virgen Extra CB Andujar": {"V": 12, "PTS": 2103},
    "INDITEX Vallekas": {"V": 12, "PTS": 1871},
    "La Rosa Nostra": {"V": 10, "PTS": 1996},
    "HSNVillamuriel": {"V": 9, "PTS": 2101},
    "Marlboro Dueñas": {"V": 9, "PTS": 1935},
    "Soria Natural": {"V": 8, "PTS": 1770},
    "Multiópticas Salgado": {"V": 6, "PTS": 1788},
    "Mercadona Carnoedo BC": {"V": 6, "PTS": 1785}
}

# 2. BASE DE DATOS REAL (J1 a J26)
@st.cache_data
def obtener_datos_reales():
    historico = {
        "Strava Palencia": [(97,96), (64,54), (61,68), (58,100), (106,88), (107,105), (84,106), (141,55), (64,54), (102,74), (122,88), (132,93), (167,51), (119,99), (120,61), (95,60), (160,123), (123,54), (77,116), (130,75), (74,81), (78,77), (118,81), (108,58), (94,116), (108,69)],
        "Foster's Rivas Sureste": [(84,36), (84,77), (84,50), (122,58), (52,69), (78,40), (31,71), (72,92), (108,99), (74,102), (109,59), (71,51), (69,91), (111,60), (83,63), (110,58), (118,91), (53,110), (116,87), (64,62), (94,50), (127,119), (116,102), (85,67), (116,94), (59,70)],
        "CB Perales Nuit": [(99,85), (112,78), (91,115), (100,58), (133,62), (124,86), (70,67), (92,31), (107,99), (62,63), (88,49), (75,74), (54,102), (79,81), (58,57), (81,68), (91,118), (81,52), (71,44), (62,113), (68,74), (53,65), (86,103), (126,69), (85,55), (111,71)],
        "Mahle Baltanás": [(52,62), (80,103), (64,94), (103,69), (96,48), (47,67), (89,70), (104,51), (120,64), (27,107), (49,88), (109,52), (91,69), (74,67), (101,108), (71,67), (123,160), (65,59), (85,76), (92,75), (119,85), (85,60), (70,46), (86,151), (130,96), (120,109)],
        "Ron Negrita Badalona": [(142,43), (86,71), (112,58), (121,64), (85,90), (118,83), (106,84), (87,46), (112,52), (81,111), (74,123), (51,71), (102,54), (67,74), (89,130), (94,71), (52,66), (78,48), (116,77), (77,96), (74,89), (78,56), (69,80), (80,87), (79,68), (92,79)],
        "Chupa-Chups Magaluf": [(82,62), (39,82), (94,64), (64,121), (88,106), (92,45), (67,70), (61,87), (48,100), (105,64), (59,109), (98,97), (143,42), (98,92), (93,92), (83,44), (98,67), (54,123), (103,125), (87,36), (89,74), (119,127), (105,34), (151,86), (110,82), (94,137)],
        "Disney Burgos": [(94,55), (85,88), (98,89), (86,58), (60,40), (109,110), (96,108), (46,87), (54,64), (107,27), (103,124), (74,75), (87,32), (110,71), (63,83), (44,83), (68,41), (124,63), (117,53), (75,130), (135,80), (65,53), (74,61), (104,94), (48,50), (59,52)],
        "CUPRA Lantadilla": [(100,95), (71,62), (85,106), (102,93), (104,73), (47,95), (108,57), (51,104), (68,60), (64,105), (124,103), (68,60), (78,53), (91,97), (130,89), (68,81), (103,89), (59,65), (51,65), (36,87), (121,94), (103,36), (81,118), (87,80), (55,85), (70,59)],
        "San Miguel Carabanchel": [(96,97), (85,83), (62,28), (60,68), (100,79), (67,47), (108,96), (92,72), (100,48), (76,69), (82,62), (83,93), (69,53), (81,79), (68,48), (71,94), (89,103), (95,72), (76,85), (59,44), (81,113), (66,67), (71,107), (69,126), (115,73), (79,92)],
        "Banco Sinentender": [(95,100), (83,85), (68,61), (69,103), (62,133), (110,109), (94,80), (94,84), (80,43), (81,111), (105,72), (77,56), (124,72), (60,111), (89,76), (95,81), (67,98), (52,81), (53,117), (96,77), (94,121), (95,94), (46,70), (68,97), (79,67), (73,70)],
        "Oliva Virgen Extra CB Andujar": [(62,52), (88,85), (50,84), (68,60), (73,104), (103,66), (31,92), (107,71), (50,71), (90,77), (123,74), (97,98), (60,96), (66,54), (61,120), (81,95), (80,115), (110,53), (82,71), (113,62), (80,135), (56,78), (107,71), (58,108), (96,130), (137,94)],
        "INDITEX Vallekas": [(72,66), (103,80), (106,85), (101,76), (79,100), (83,118), (71,31), (87,61), (71,50), (63,62), (72,73), (56,77), (32,87), (99,119), (35,74), (79,74), (93,81), (72,95), (71,82), (46,82), (35,94), (77,78), (80,69), (67,85), (50,48), (71,111)],
        "La Rosa Nostra": [(36,84), (61,115), (115,91), (70,74), (90,85), (78,106), (61,41), (116,101), (79,71), (69,76), (73,72), (93,132), (72,124), (71,110), (92,93), (67,71), (115,80), (48,78), (99,56), (82,46), (50,94), (36,103), (81,102), (67,86), (105,80), (69,108)],
        "HSNVillamuriel": [(62,82), (71,86), (50,88), (58,122), (40,60), (106,78), (80,94), (54,72), (99,107), (82,57), (88,122), (93,83), (96,60), (97,91), (108,101), (81,83), (81,93), (76,48), (87,116), (44,59), (81,74), (94,95), (102,81), (94,104), (68,79), (109,120)],
        "Marlboro Dueñas": [(66,72), (78,112), (89,98), (93,102), (43,89), (40,78), (84,81), (101,116), (43,80), (127,101), (90,59), (52,109), (51,167), (92,98), (48,68), (83,81), (66,52), (84,72), (56,99), (75,92), (94,35), (135,88), (34,105), (86,59), (73,115), (52,59)],
        "Soria Natural": [(43,142), (82,39), (88,50), (74,70), (69,52), (86,124), (57,108), (55,141), (64,120), (77,90), (59,90), (47,85), (53,69), (82,66), (76,89), (74,79), (41,68), (48,76), (65,51), (47,93), (74,68), (88,135), (102,116), (86,67), (67,79), (66,72)],
        "Multiópticas Salgado": [(55,94), (107,105), (28,62), (76,101), (48,96), (45,92), (41,61), (72,54), (52,112), (101,127), (72,105), (85,47), (53,78), (54,66), (57,58), (58,110), (41,68), (63,124), (125,103), (93,47), (85,119), (67,66), (103,86), (59,86), (80,105), (70,73)],
        "Mercadona Carnoedo BC": [(85,99), (52,77), (58,112), (89,43), (58,86), (66,103), (70,89), (84,94), (71,79), (57,82), (62,82), (60,68), (42,143), (66,82), (74,35), (60,95), (68,41), (72,84), (44,71), (62,64), (113,81), (60,85), (61,74), (97,68), (82,110), (72,66)]
    }
    
    lista_df = []
    for eq, jornadas in historico.items():
        for i, (a, r) in enumerate(jornadas, 1):
            lista_df.append({
                "Jornada": i, "Equipo": eq, "Anotados": a, "Recibidos": r, "Victoria": 1 if a > r else 0
            })
    return pd.DataFrame(lista_df)

df_historico = obtener_datos_reales()

# 3. CALCULAR EVOLUCIÓN
@st.cache_data
def calcular_evolucion_real(_df):
    evolucion = []
    equipos = _df["Equipo"].unique()
    acumulado = {eq: {"V": 0, "PTS": 0} for eq in equipos}
    
    for j in range(1, JORNADAS_JUGADAS + 1):
        df_j = _df[_df["Jornada"] == j]
        for _, row in df_j.iterrows():
            acumulado[row["Equipo"]]["V"] += row["Victoria"]
            acumulado[row["Equipo"]]["PTS"] += row["Anotados"]
        
        ranking = sorted(acumulado.keys(), key=lambda x: (acumulado[x]["V"], acumulado[x]["PTS"]), reverse=True)
        for pos, eq in enumerate(ranking, 1):
            evolucion.append({"Jornada": j, "Equipo": eq, "Posición": pos})
    return pd.DataFrame(evolucion)

df_evolucion = calcular_evolucion_real(df_historico)

# 4. CALENDARIO RESTANTE (J27 - J34)
partidos_restantes = [
    ("La Rosa Nostra", "Mahle Baltanás"), ("Oliva Virgen Extra CB Andujar", "CUPRA Lantadilla"),
    ("Disney Burgos", "Ron Negrita Badalona"), ("Marlboro Dueñas", "Banco Sinentender"),
    ("San Miguel Carabanchel", "Strava Palencia"), ("Foster's Rivas Sureste", "CB Perales Nuit"),
    ("Soria Natural", "Chupa-Chups Magaluf"), ("INDITEX Vallekas", "Mercadona Carnoedo BC"),
    ("Multiópticas Salgado", "HSNVillamuriel"),
    ("Mahle Baltanás", "Soria Natural"), ("HSNVillamuriel", "Marlboro Dueñas"),
    ("Strava Palencia", "Banco Sinentender"), ("Ron Negrita Badalona", "Foster's Rivas Sureste"),
    ("INDITEX Vallekas", "Multiópticas Salgado"), ("La Rosa Nostra", "San Miguel Carabanchel"),
    ("Mercadona Carnoedo BC", "Oliva Virgen Extra CB Andujar"), ("CUPRA Lantadilla", "Disney Burgos"),
    ("CB Perales Nuit", "Chupa-Chups Magaluf"),
    ("Foster's Rivas Sureste", "Mahle Baltanás"), ("Oliva Virgen Extra CB Andujar", "La Rosa Nostra"),
    ("Disney Burgos", "Soria Natural"), ("Marlboro Dueñas", "Strava Palencia"),
    ("Banco Sinentender", "San Miguel Carabanchel"), ("CUPRA Lantadilla", "HSNVillamuriel"),
    ("Multiópticas Salgado", "Mercadona Carnoedo BC"), ("CB Perales Nuit", "Ron Negrita Badalona"),
    ("Chupa-Chups Magaluf", "INDITEX Vallekas"),
    ("Mahle Baltanás", "INDITEX Vallekas"), ("HSNVillamuriel", "CB Perales Nuit"),
    ("Strava Palencia", "Multiópticas Salgado"), ("Marlboro Dueñas", "Ron Negrita Badalona"),
    ("San Miguel Carabanchel", "CUPRA Lantadilla"), ("Banco Sinentender", "Foster's Rivas Sureste"),
    ("Soria Natural", "Oliva Virgen Extra CB Andujar"), ("Mercadona Carnoedo BC", "La Rosa Nostra"),
    ("Chupa-Chups Magaluf", "Disney Burgos"),
    ("Disney Burgos", "Mahle Baltanás"), ("Oliva Virgen Extra CB Andujar", "Marlboro Dueñas"),
    ("Ron Negrita Badalona", "Soria Natural"), ("Foster's Rivas Sureste", "San Miguel Carabanchel"),
    ("INDITEX Vallekas", "Banco Sinentender"), ("La Rosa Nostra", "Chupa-Chups Magaluf"),
    ("Mercadona Carnoedo BC", "HSNVillamuriel"), ("Multiópticas Salgado", "CUPRA Lantadilla"),
    ("CB Perales Nuit", "Strava Palencia"),
    ("Strava Palencia", "Mahle Baltanás"), ("HSNVillamuriel", "Oliva Virgen Extra CB Andujar"),
    ("Ron Negrita Badalona", "Multiópticas Salgado"), ("Marlboro Dueñas", "Foster's Rivas Sureste"),
    ("San Miguel Carabanchel", "Disney Burgos"), ("Banco Sinentender", "Chupa-Chups Magaluf"),
    ("Soria Natural", "INDITEX Vallekas"), ("CUPRA Lantadilla", "Mercadona Carnoedo BC"),
    ("CB Perales Nuit", "La Rosa Nostra"),
    ("Mahle Baltanás", "Ron Negrita Badalona"), ("Oliva Virgen Extra CB Andujar", "Multiópticas Salgado"),
    ("Marlboro Dueñas", "CB Perales Nuit"), ("Foster's Rivas Sureste", "Disney Burgos"),
    ("Soria Natural", "San Miguel Carabanchel"), ("INDITEX Vallekas", "CUPRA Lantadilla"),
    ("La Rosa Nostra", "Banco Sinentender"), ("Mercadona Carnoedo BC", "Strava Palencia"),
    ("Chupa-Chups Magaluf", "HSNVillamuriel"),
    ("CB Perales Nuit", "Mahle Baltanás"), ("Strava Palencia", "Soria Natural"),
    ("Disney Burgos", "La Rosa Nostra"), ("Ron Negrita Badalona", "Mercadona Carnoedo BC"),
    ("San Miguel Carabanchel", "Chupa-Chups Magaluf"), ("Banco Sinentender", "Oliva Virgen Extra CB Andujar"),
    ("CUPRA Lantadilla", "Marlboro Dueñas"), ("Multiópticas Salgado", "Foster's Rivas Sureste"),
    ("HSNVillamuriel", "INDITEX Vallekas")
]

# INTERFAZ POR PESTAÑAS
tab1, tab2, tab3 = st.tabs(["Simulación Playoffs", "Evolución Clasificación", "Puntos por Jornada"])

# --- TAB 1: SIMULACIÓN DE MONTECARLO PREDICITVO ---
with tab1:
    st.sidebar.header("Ajustes de Simulación")
    
    st.sidebar.markdown(f"**Iteraciones:** {NUM_SIMULACIONES:,}")
    st.sidebar.markdown(f"**Puestos de Playoff:** Top {PUESTOS_PLAYOFF}")
    st.sidebar.markdown(f"**Margen de Empate:** {MARGEN_EMPATE} pts")
    
    local_gana_empate = st.sidebar.checkbox("En caso de empate, gana el Local", value=True)

    stats_sim = df_historico.groupby("Equipo").agg({"Anotados": ["mean", "std"]})
    stats_sim.columns = ["media", "desv"]

    if st.button("Iniciar Simulación", type="primary"):
        prog_bar = st.progress(0)
        status = st.empty()
        
        stats_finales = {eq: {"pos": [], "vic": [], "pts": []} for eq in clasificacion_actual}
        
        # En una liga virtual no hay bonus local (todos juegan en campo neutral)
        bonus_local = 0.0 
        
        for i in range(NUM_SIMULACIONES):
            sim = {eq: d.copy() for eq, d in clasificacion_actual.items()}
            
            for loc, vis in partidos_restantes:
                media_loc = stats_sim.loc[loc, "media"]
                std_loc = stats_sim.loc[loc, "desv"]
                media_vis = stats_sim.loc[vis, "media"]
                std_vis = stats_sim.loc[vis, "desv"]

                # Simulación de anotación sin ventaja de campo
                pts_loc = random.gauss(media_loc + bonus_local, std_loc)
                pts_vis = random.gauss(media_vis, std_vis)

                diferencia = abs(pts_loc - pts_vis)
                
                # Resolución del partido
                if diferencia <= MARGEN_EMPATE:
                    if local_gana_empate:
                        sim[loc]["V"] += 1
                    else:
                        ganador = random.choice([loc, vis])
                        sim[ganador]["V"] += 1
                elif pts_loc > pts_vis:
                    sim[loc]["V"] += 1
                else:
                    sim[vis]["V"] += 1

                sim[loc]["PTS"] += pts_loc
                sim[vis]["PTS"] += pts_vis
                
            df_sim = pd.DataFrame.from_dict(sim, orient='index')
            df_sim = df_sim.sort_values(by=["V", "PTS"], ascending=[False, False])
            
            for pos, (eq, row) in enumerate(df_sim.iterrows(), start=1):
                stats_finales[eq]["pos"].append(pos)
                stats_finales[eq]["vic"].append(row["V"])
                stats_finales[eq]["pts"].append(row["PTS"])
                
            if i % (NUM_SIMULACIONES // 10) == 0:
                prog = int((i / NUM_SIMULACIONES) * 100)
                prog_bar.progress(prog)
                status.text(f"Calculando probabilidades... {prog}%")

        prog_bar.empty()
        status.empty()

        res = []
        for eq, s in stats_finales.items():
            pos = pd.Series(s["pos"])
            res.append({
                "Equipo": eq,
                "V. Actuales": clasificacion_actual[eq]["V"],
                "Cab. Serie (1-4)": (pos <= 4).mean(),
                f"Playoff (5-{PUESTOS_PLAYOFF})": ((pos >= 5) & (pos <= PUESTOS_PLAYOFF)).mean(),
                "Descenso (17-18)": (pos >= 17).mean(),
                "Pos. Media": pos.mean(),
                "Proyección (V)": pd.Series(s["vic"]).mean(),
                "Proyección PTS": pd.Series(s["pts"]).mean()
            })

        df_res = pd.DataFrame(res).sort_values(by="Pos. Media").reset_index(drop=True)
        df_res.index += 1

        st.dataframe(
            df_res.style.format({
                "V. Actuales": "{:.0f}", "Cab. Serie (1-4)": "{:.1%}", 
                f"Playoff (5-{PUESTOS_PLAYOFF})": "{:.1%}", "Descenso (17-18)": "{:.1%}", 
                "Pos. Media": "{:.1f}", "Proyección (V)": "{:.1f}", "Proyección PTS": "{:.0f}"
            }).background_gradient(cmap="Greens", subset=["Cab. Serie (1-4)", f"Playoff (5-{PUESTOS_PLAYOFF})"])
              .background_gradient(cmap="Reds", subset=["Descenso (17-18)"])
              .background_gradient(cmap="Blues", subset=["Proyección PTS"]),
            use_container_width=True, height=650
        )

# --- TAB 2: EVOLUCIÓN CLASIFICACIÓN ---
with tab2:
    st.subheader("Trayectoria en la Clasificación")
    st.markdown("Observa las rachas y caídas de cada equipo a lo largo de las 26 jornadas ya jugadas.")
    
    equipos_seleccionados = st.multiselect("Filtrar equipos:", list(clasificacion_actual.keys()), default=["Strava Palencia", "Foster's Rivas Sureste", "CB Perales Nuit", "Mahle Baltanás"])
    
    df_filtrado = df_evolucion[df_evolucion["Equipo"].isin(equipos_seleccionados)]
    
    if not df_filtrado.empty:
        fig_evo = px.line(df_filtrado, x="Jornada", y="Posición", color="Equipo", markers=True,
                          title="Posición al final de cada Jornada")
        fig_evo.update_yaxes(autorange="reversed", tickmode="linear", tick0=1, dtick=1)
        fig_evo.update_xaxes(tickmode="linear", tick0=1, dtick=1)
        st.plotly_chart(fig_evo, use_container_width=True)

# --- TAB 3: PUNTOS POR JORNADA ---
with tab3:
    st.subheader("Balance ofensivo y defensivo")
    st.markdown("Compara los puntos anotados y recibidos de cada equipo.")
    
    equipo_a_ver = st.selectbox("Selecciona un equipo:", list(clasificacion_actual.keys()))
    
    df_equipo = df_historico[df_historico["Equipo"] == equipo_a_ver].copy()
    
    media_anotados = df_equipo["Anotados"].mean()
    media_recibidos = df_equipo["Recibidos"].mean()
    diferencia_media = media_anotados - media_recibidos
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Media Anotada", f"{media_anotados:.1f} pts")
    col2.metric("Media Recibida", f"{media_recibidos:.1f} pts")
    col3.metric("Balance Promedio", f"{diferencia_media:+.1f} pts")
    
    st.markdown("---")
    
    df_plot = df_equipo.melt(id_vars=["Jornada"], value_vars=["Anotados", "Recibidos"], 
                             var_name="Tipo", value_name="Puntos")
    
    fig_pts = px.bar(df_plot, x="Jornada", y="Puntos", color="Tipo", barmode="group",
                     text="Puntos", color_discrete_map={"Anotados": "#1f77b4", "Recibidos": "#d62728"})
    
    fig_pts.add_hline(y=media_anotados, line_dash="dot", line_color="#1f77b4")
    fig_pts.add_hline(y=media_recibidos, line_dash="dot", line_color="#d62728")
    
    fig_pts.update_traces(textposition='outside')
    fig_pts.update_xaxes(tickmode="linear", tick0=1, dtick=1)
    
    max_pts = df_equipo[["Anotados", "Recibidos"]].max().max()
    fig_pts.update_yaxes(range=[0, max_pts * 1.15])
    
    st.plotly_chart(fig_pts, use_container_width=True)
