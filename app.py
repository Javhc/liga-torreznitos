import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px

# Configuración de la página
st.set_page_config(page_title="Liga Torreznitos", layout="wide")

st.title("🏀 Liga Torreznitos")
st.markdown("Análisis probabilístico basado en inercia de rendimiento anotador (EWMA) y azar puro.")

# --- CONSTANTES FIJAS ---
NUM_SIMULACIONES = 10000
PUESTOS_PLAYOFF = 8
MARGEN_EMPATE = 1.5
JORNADA_MAX_REAL = 28 

# 1. BASE DE DATOS REAL (J1 a J28)
@st.cache_data
def obtener_datos_reales():
    historico = {
        "Strava Palencia": [(97,96), (107,105), (61,68), (58,100), (106,88), (95,47), (84,106), (141,55), (64,54), (102,74), (122,88), (132,93), (167,51), (119,99), (120,61), (95,60), (160,123), (123,54), (77,116), (130,75), (74,81), (78,77), (118,81), (108,58), (94,116), (108,69), (83,65), (116,71)],
        "Foster's Rivas Sureste": [(84,36), (77,52), (84,50), (122,58), (52,69), (78,40), (31,71), (72,92), (99,108), (74,102), (109,59), (71,51), (69,91), (111,60), (83,63), (110,58), (118,91), (53,110), (116,87), (64,62), (94,50), (127,119), (116,102), (85,67), (116,94), (59,70), (119,89), (122,77)],
        "CB Perales Nuit": [(99,85), (112,78), (91,115), (100,58), (133,62), (124,86), (70,67), (92,31), (107,99), (62,63), (88,49), (75,74), (54,102), (79,81), (58,57), (81,68), (91,118), (81,52), (71,44), (62,113), (68,74), (53,65), (86,103), (126,69), (85,55), (111,71), (89,119), (134,86)],
        "Mahle Baltanás": [(52,62), (80,103), (64,94), (103,69), (96,48), (47,67), (89,70), (104,51), (120,64), (27,107), (49,88), (109,52), (91,69), (74,67), (101,108), (71,67), (123,160), (65,59), (85,76), (92,75), (119,85), (85,60), (70,46), (86,151), (130,96), (120,109), (85,99), (96,61)],
        "Ron Negrita Badalona": [(142,43), (86,71), (112,58), (121,64), (85,90), (118,83), (106,84), (87,46), (112,52), (111,81), (74,123), (51,71), (102,54), (67,74), (89,130), (94,71), (52,66), (78,48), (116,77), (77,96), (74,89), (78,56), (69,80), (80,87), (79,68), (92,79), (75,82), (77,122)],
        "Chupa-Chups Magaluf": [(82,62), (39,82), (94,64), (64,121), (88,106), (92,45), (67,70), (61,87), (48,100), (105,64), (59,109), (98,97), (143,42), (98,92), (93,92), (83,44), (98,67), (54,123), (103,125), (87,36), (89,74), (119,127), (105,34), (151,86), (110,82), (94,137), (124,85), (86,134)],
        "Disney Burgos": [(94,55), (85,88), (98,89), (86,58), (60,40), (109,110), (96,108), (46,87), (54,64), (107,27), (103,124), (74,75), (87,32), (110,71), (63,83), (44,83), (68,41), (124,63), (117,53), (75,130), (135,80), (65,53), (74,61), (104,94), (48,50), (59,52), (82,75), (47,87)],
        "CUPRA Lantadilla": [(100,95), (71,62), (85,106), (102,93), (104,73), (47,95), (108,57), (51,104), (108,99), (64,105), (124,103), (68,60), (78,53), (91,97), (130,89), (68,81), (103,89), (59,65), (51,65), (36,87), (121,94), (103,36), (81,118), (87,80), (55,85), (70,59), (119,95), (87,47)],
        "San Miguel Carabanchel": [(96,97), (85,83), (62,28), (60,68), (100,79), (67,47), (108,96), (92,72), (100,48), (76,69), (82,62), (83,93), (69,53), (81,79), (68,48), (71,94), (89,103), (95,72), (76,85), (59,44), (81,113), (66,67), (71,107), (69,126), (115,73), (79,92), (65,83), (89,98)],
        "Banco Sinentender": [(95,100), (83,85), (68,61), (69,103), (62,133), (110,109), (94,80), (94,84), (80,43), (81,111), (105,72), (77,56), (124,72), (60,111), (89,76), (95,81), (67,98), (52,81), (53,117), (96,77), (94,121), (95,94), (46,70), (68,97), (79,67), (73,70), (69,65), (71,116)],
        "Oliva Virgen Extra CB Andujar": [(62,52), (88,85), (50,84), (68,60), (73,104), (103,66), (81,84), (31,92), (50,71), (90,77), (123,74), (97,98), (60,96), (66,54), (61,120), (81,95), (80,115), (110,53), (82,71), (113,62), (80,135), (56,78), (107,71), (58,108), (96,130), (137,94), (95,119), (94,67)],
        "INDITEX Vallekas": [(72,66), (103,80), (106,85), (101,76), (79,100), (83,118), (71,31), (87,61), (71,50), (63,62), (72,73), (56,77), (32,87), (99,119), (35,74), (79,74), (93,81), (72,95), (71,82), (46,82), (35,94), (77,78), (80,69), (67,85), (50,48), (71,111), (41,71), (75,66)],
        "La Rosa Nostra": [(36,84), (62,71), (115,91), (70,74), (90,85), (78,106), (61,41), (116,101), (79,71), (69,76), (73,72), (93,132), (72,124), (71,110), (92,93), (67,71), (115,80), (48,78), (99,56), (82,46), (50,94), (36,103), (81,102), (67,86), (105,80), (69,108), (99,85), (98,89)],
        "HSNVillamuriel": [(62,82), (71,86), (50,88), (58,122), (40,60), (106,78), (80,94), (54,72), (99,107), (82,57), (88,122), (93,83), (96,60), (97,91), (108,101), (81,83), (81,93), (76,48), (87,116), (44,59), (81,74), (94,95), (102,81), (94,104), (68,79), (109,120), (66,57), (62,116)],
        "Marlboro Dueñas": [(66,72), (78,112), (89,98), (93,102), (43,89), (40,78), (84,81), (101,116), (43,80), (127,101), (90,59), (52,109), (51,167), (92,98), (48,68), (83,81), (66,52), (84,72), (56,99), (75,92), (94,35), (135,88), (34,105), (86,59), (73,115), (52,59), (65,69), (116,62)],
        "Soria Natural": [(43,142), (82,39), (88,50), (74,70), (69,52), (86,124), (57,108), (55,141), (64,120), (77,90), (59,90), (47,85), (53,69), (82,66), (76,89), (74,79), (41,68), (48,76), (65,51), (47,93), (74,68), (88,135), (102,116), (86,67), (67,79), (66,72), (85,124), (61,96)],
        "Multiópticas Salgado": [(55,94), (105,107), (28,62), (76,101), (48,96), (45,92), (41,61), (72,54), (52,112), (101,127), (72,105), (85,47), (53,78), (54,66), (57,58), (58,110), (41,68), (63,124), (125,103), (93,47), (85,119), (67,66), (103,86), (59,86), (80,105), (70,73), (57,66), (66,75)],
        "Mercadona Carnoedo BC": [(85,99), (52,77), (58,112), (58,86), (89,43), (66,103), (70,89), (84,94), (71,79), (57,82), (62,82), (60,68), (42,143), (66,82), (74,35), (60,95), (68,41), (72,84), (44,71), (62,64), (113,81), (60,85), (61,74), (97,68), (82,110), (72,66), (71,41), (67,94)]
    }
    
    lista_df = []
    for eq, jornadas in historico.items():
        for i, (a, r) in enumerate(jornadas, 1):
            lista_df.append({
                "Jornada": i, "Equipo": eq, "Anotados": a, "Recibidos": r, 
                "Victoria": 1 if a > r else 0
            })
    return pd.DataFrame(lista_df), historico

df_historico, dict_historico = obtener_datos_reales()

# --- CÁLCULO DINÁMICO DEL EXPONENTE PITAGÓRICO ---
@st.cache_data
def calcular_exponente_optimo(_df_hist):
    equipos = _df_hist["Equipo"].unique()
    datos_equipos = []
    for eq in equipos:
        df_t = _df_hist[_df_hist["Equipo"] == eq]
        datos_equipos.append((
            df_t["Victoria"].sum(), df_t["Anotados"].sum(), df_t["Recibidos"].sum()
        ))
        
    mejor_exponente, menor_error = 1.0, float('inf')
    for exp in np.arange(1.0, 20.0, 0.1):
        error_actual = sum(abs(v - ((pf**exp / (pf**exp + pc**exp)) * JORNADA_MAX_REAL)) for v, pf, pc in datos_equipos if (pf+pc)>0)
        if error_actual < menor_error: 
            menor_error, mejor_exponente = error_actual, exp
            
    return mejor_exponente

EXPONENTE_SUERTE = calcular_exponente_optimo(df_historico)

# 2. RECONSTRUCCIÓN DEL CALENDARIO PASADO
@st.cache_data
def reconstruir_calendario_pasado(dict_hist):
    calendario = []
    for j in range(JORNADA_MAX_REAL):
        equipos_procesados = set()
        for eq, resultados in dict_hist.items():
            if eq in equipos_procesados: continue
            pts_f, pts_c = resultados[j]
            for rival, r_resultados in dict_hist.items():
                if rival == eq or rival in equipos_procesados: continue
                r_pts_f, r_pts_c = r_resultados[j]
                if pts_f == r_pts_c and pts_c == r_pts_f:
                    calendario.append((j + 1, eq, rival))
                    equipos_procesados.update([eq, rival])
                    break
    return calendario

calendario_pasado = reconstruir_calendario_pasado(dict_historico)

# 3. CALENDARIO FUTURO (J29 - J34)
calendario_futuro = [
    (29, "Foster's Rivas Sureste", "Mahle Baltanás"), (29, "Oliva Virgen Extra CB Andujar", "La Rosa Nostra"), (29, "Disney Burgos", "Soria Natural"), (29, "Marlboro Dueñas", "Strava Palencia"), (29, "Banco Sinentender", "San Miguel Carabanchel"), (29, "CUPRA Lantadilla", "HSNVillamuriel"), (29, "Multiópticas Salgado", "Mercadona Carnoedo BC"), (29, "CB Perales Nuit", "Ron Negrita Badalona"), (29, "Chupa-Chups Magaluf", "INDITEX Vallekas"),
    (30, "Mahle Baltanás", "INDITEX Vallekas"), (30, "HSNVillamuriel", "CB Perales Nuit"), (30, "Strava Palencia", "Multiópticas Salgado"), (30, "Marlboro Dueñas", "Ron Negrita Badalona"), (30, "San Miguel Carabanchel", "CUPRA Lantadilla"), (30, "Banco Sinentender", "Foster's Rivas Sureste"), (30, "Soria Natural", "Oliva Virgen Extra CB Andujar"), (30, "Mercadona Carnoedo BC", "La Rosa Nostra"), (30, "Chupa-Chups Magaluf", "Disney Burgos"),
    (31, "Disney Burgos", "Mahle Baltanás"), (31, "Oliva Virgen Extra CB Andujar", "Marlboro Dueñas"), (31, "Ron Negrita Badalona", "Soria Natural"), (31, "Foster's Rivas Sureste", "San Miguel Carabanchel"), (31, "INDITEX Vallekas", "Banco Sinentender"), (31, "La Rosa Nostra", "Chupa-Chups Magaluf"), (31, "Mercadona Carnoedo BC", "HSNVillamuriel"), (31, "Multiópticas Salgado", "CUPRA Lantadilla"), (31, "CB Perales Nuit", "Strava Palencia"),
    (32, "Strava Palencia", "Mahle Baltanás"), (32, "HSNVillamuriel", "Oliva Virgen Extra CB Andujar"), (32, "Ron Negrita Badalona", "Multiópticas Salgado"), (32, "Marlboro Dueñas", "Foster's Rivas Sureste"), (32, "San Miguel Carabanchel", "Disney Burgos"), (32, "Banco Sinentender", "Chupa-Chups Magaluf"), (32, "Soria Natural", "INDITEX Vallekas"), (32, "CUPRA Lantadilla", "Mercadona Carnoedo BC"), (32, "CB Perales Nuit", "La Rosa Nostra"),
    (33, "Mahle Baltanás", "Ron Negrita Badalona"), (33, "Oliva Virgen Extra CB Andujar", "Multiópticas Salgado"), (33, "Marlboro Dueñas", "CB Perales Nuit"), (33, "Foster's Rivas Sureste", "Disney Burgos"), (33, "Soria Natural", "San Miguel Carabanchel"), (33, "INDITEX Vallekas", "CUPRA Lantadilla"), (33, "La Rosa Nostra", "Banco Sinentender"), (33, "Mercadona Carnoedo BC", "Strava Palencia"), (33, "Chupa-Chups Magaluf", "HSNVillamuriel"),
    (34, "CB Perales Nuit", "Mahle Baltanás"), (34, "Strava Palencia", "Soria Natural"), (34, "Disney Burgos", "La Rosa Nostra"), (34, "Ron Negrita Badalona", "Mercadona Carnoedo BC"), (34, "San Miguel Carabanchel", "Chupa-Chups Magaluf"), (34, "Banco Sinentender", "Oliva Virgen Extra CB Andujar"), (34, "CUPRA Lantadilla", "Marlboro Dueñas"), (34, "Multiópticas Salgado", "Foster's Rivas Sureste"), (34, "HSNVillamuriel", "INDITEX Vallekas")
]

calendario_total = calendario_pasado + calendario_futuro

# 4. EVOLUCIÓN REAL
@st.cache_data
def calcular_evolucion_real(_df):
    evolucion = []
    equipos = _df["Equipo"].unique()
    acumulado = {eq: {"V": 0, "PTS": 0} for eq in equipos}
    for j in range(1, JORNADA_MAX_REAL + 1):
        df_j = _df[_df["Jornada"] == j]
        for _, row in df_j.iterrows():
            acumulado[row["Equipo"]]["V"] += row["Victoria"]
            acumulado[row["Equipo"]]["PTS"] += row["Anotados"]
            
        ranking = sorted(acumulado.keys(), key=lambda x: (acumulado[x]["V"], acumulado[x]["PTS"]), reverse=True)
        for pos, eq in enumerate(ranking, 1): 
            evolucion.append({"Jornada": j, "Equipo": eq, "Posición": pos})
            
    return pd.DataFrame(evolucion)

df_evolucion = calcular_evolucion_real(df_historico)
mapa_colores = {eq: px.colors.qualitative.Alphabet[i % len(px.colors.qualitative.Alphabet)] for i, eq in enumerate(dict_historico.keys())}

# --- INTERFAZ POR PESTAÑAS ---
tab1, tab2, tab3 = st.tabs(["Simulaciones", "Evolución Histórica", "Estadísticas y Clasificación"])

with tab1:
    st.sidebar.header("Selecciona Jornada")
    jornada_referencia = st.sidebar.slider("Ver clasificación simulada desde:", 1, JORNADA_MAX_REAL, JORNADA_MAX_REAL)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Ajustes de Simulación")
    
    modelo_elegido = st.sidebar.radio(
        "Selecciona el modelo predictivo:", 
        ["Montecarlo Puro (Cara o Cruz 50%)", "Estado de Forma (EWMA - Solo Ataque)"]
    )
    
    n_jornadas_forma = st.sidebar.slider(
        "Ventana de análisis (últimas J para EWMA):", 
        min_value=3, max_value=15, value=10, 
        help="Cuántos partidos pasados afectan a la inercia anotadora del equipo."
    )
    
    modo_ejecucion = st.sidebar.radio(
        "Modo de ejecución:", 
        ["Simular SOLO la Jornada seleccionada", "Simular TODAS las jornadas"]
    )
    
    local_gana_empate = st.sidebar.checkbox("En caso de empate matemático (EWMA), gana el Local", value=True)

    def calcular_medias_ataque(tipo, j_ref, n_form):
        medias = {}
        for eq in dict_historico.keys():
            historial = dict_historico[eq][:j_ref]
            
            if len(historial) == 0: 
                medias[eq] = {"media": 0, "std": 10}
                continue

            # SOLO NOS IMPORTAN LOS PUNTOS A FAVOR EN FANTASY
            pts_a_favor = [x[0] for x in historial]

            if tipo.startswith("Montecarlo"):
                # No se usan para simular en Montecarlo puro, rellenamos por estructura
                medias[eq] = {"media": 0, "std": 0}
                
            else: # EWMA
                n_total = min(n_form, len(historial))
                u_ataque = pts_a_favor[-n_total:]
                
                alpha = 2 / (n_total + 1)
                pesos = [(1 - alpha)**(n_total - 1 - i) for i in range(n_total)]
                
                medias[eq] = {
                    "media": np.average(u_ataque, weights=pesos),
                    "std": np.std(pts_a_favor) if len(pts_a_favor) > 1 else 10
                }
        return medias

    if st.button("Iniciar Simulación", type="primary"):
        prog_bar = st.progress(0)
        status = st.empty()
        datos_evolucion_probs = []
        rango_jornadas = [jornada_referencia] if modo_ejecucion.startswith("Simular SOLO") else range(1, JORNADA_MAX_REAL + 1)
        
        for idx, j_actual in enumerate(rango_jornadas):
            status.text(f"Calculando escenarios... J{j_actual}")
            stats_finales = {eq: {"pos": [], "vic": [], "pts": []} for eq in dict_historico.keys()}
            
            estado_j = {
                eq: {
                    "V": sum(1 for a, r in dict_historico[eq][:j_actual] if a > r), 
                    "PTS": sum(a for a, r in dict_historico[eq][:j_actual])
                } for eq in dict_historico.keys()
            }
            
            partidos_a_simular = [p for p in calendario_total if p[0] > j_actual]
            medias_modelo = calcular_medias_ataque(modelo_elegido, j_actual, n_jornadas_forma)
            
            for i in range(NUM_SIMULACIONES):
                sim = {eq: d.copy() for eq, d in estado_j.items()}
                
                for _, loc, vis in partidos_a_simular:
                    
                    if modelo_elegido.startswith("Montecarlo"):
                        # AZAR PURO AL 50%
                        ganador = loc if random.choice([True, False]) else vis
                        p_loc = 85 if ganador == loc else 80
                        p_vis = 85 if ganador == vis else 80
                        
                    else:
                        # EWMA: El equipo tira contra su propia media anotadora, sin importar la defensa rival
                        p_loc = random.gauss(medias_modelo[loc]["media"], medias_modelo[loc]["std"])
                        p_vis = random.gauss(medias_modelo[vis]["media"], medias_modelo[vis]["std"])
                        
                        if abs(p_loc - p_vis) <= MARGEN_EMPATE:
                            ganador = loc if local_gana_empate else random.choice([loc, vis])
                        else:
                            ganador = loc if p_loc > p_vis else vis
                            
                    sim[ganador]["V"] += 1
                    sim[loc]["PTS"] += p_loc
                    sim[vis]["PTS"] += p_vis
                
                ranking = sorted(sim.items(), key=lambda x: (x[1]["V"], x[1]["PTS"]), reverse=True)
                for pos, (eq, d) in enumerate(ranking, 1): 
                    stats_finales[eq]["pos"].append(pos)
                    stats_finales[eq]["vic"].append(d["V"])
                    stats_finales[eq]["pts"].append(d["PTS"])
            
            res_jornada = []
            for eq, s in stats_finales.items():
                pos = pd.Series(s["pos"])
                prob_playoff = (pos <= PUESTOS_PLAYOFF).mean()
                prob_cab_serie = (pos <= 4).mean()
                prob_fuera = (pos > PUESTOS_PLAYOFF).mean()
                
                datos_evolucion_probs.append({
                    "Jornada": j_actual, "Equipo": eq, 
                    "Prob. Playoff": prob_playoff, 
                    "Prob. Cab. Serie": prob_cab_serie, 
                    "Prob. Fuera Playoff": prob_fuera
                })
                
                if j_actual == jornada_referencia:
                    res_jornada.append({
                        "Equipo": eq, 
                        "V. en J"+str(jornada_referencia): estado_j[eq]["V"], 
                        "Cab. Serie (1-4)": prob_cab_serie, 
                        f"Playoff (5-{PUESTOS_PLAYOFF})": ((pos >= 5) & (pos <= PUESTOS_PLAYOFF)).mean(), 
                        "Fuera Playoff (>8)": prob_fuera, 
                        "Descenso (17-18)": (pos >= 17).mean(), 
                        "Pos. Media": pos.mean(), 
                        "Proyección (V)": pd.Series(s["vic"]).mean()
                    })
                    
            if j_actual == jornada_referencia: 
                st.session_state["tabla_principal"] = pd.DataFrame(res_jornada).sort_values(by="Pos. Media").reset_index(drop=True)
                st.session_state["tabla_principal"].index += 1
                st.session_state["jornada_simulada"] = jornada_referencia
                
            prog_bar.progress(int(((idx + 1) / len(rango_jornadas)) * 100))
            
        if not modo_ejecucion.startswith("Simular SOLO"): 
            st.session_state["df_probs_historia"] = pd.DataFrame(datos_evolucion_probs)
            
        prog_bar.empty()
        status.empty()

    if "tabla_principal" in st.session_state:
        st.subheader(f"Proyecciones desde la Jornada {st.session_state['jornada_simulada']}")
        st.dataframe(
            st.session_state["tabla_principal"].style.format({
                "Cab. Serie (1-4)": "{:.1%}", 
                f"Playoff (5-{PUESTOS_PLAYOFF})": "{:.1%}", 
                "Fuera Playoff (>8)": "{:.1%}", 
                "Descenso (17-18)": "{:.1%}", 
                "Pos. Media": "{:.1f}", 
                "Proyección (V)": "{:.1f}"
            }).background_gradient(cmap="Greens", subset=["Cab. Serie (1-4)", f"Playoff (5-{PUESTOS_PLAYOFF})"])
              .background_gradient(cmap="Oranges", subset=["Fuera Playoff (>8)"])
              .background_gradient(cmap="Reds", subset=["Descenso (17-18)"]), 
            use_container_width=True, height=650
        )

# --- TAB 2: EVOLUCIÓN HISTÓRICA ---
with tab2:
    st.subheader("Análisis de trayectoria y probabilidades")
    eqs_sel = st.multiselect("Filtrar equipos:", list(dict_historico.keys()), default=["Strava Palencia", "CB Perales Nuit", "Soria Natural"])
    
    df_f = df_evolucion[df_evolucion["Equipo"].isin(eqs_sel)]
    if not df_f.empty: 
        fig_evo = px.line(df_f, x="Jornada", y="Posición", color="Equipo", markers=True, title="1. Posición Real a lo largo de la temporada", color_discrete_map=mapa_colores)
        fig_evo.update_yaxes(autorange="reversed", dtick=1)
        st.plotly_chart(fig_evo, use_container_width=True)
        
    st.markdown("---")
        
    if "df_probs_historia" in st.session_state:
        df_p = st.session_state["df_probs_historia"]
        df_pf = df_p[df_p["Equipo"].isin(eqs_sel)]
        
        fig_play = px.line(df_pf, x="Jornada", y="Prob. Playoff", color="Equipo", markers=True, title="2. Opciones de entrar en Playoff (Top 8)", color_discrete_map=mapa_colores)
        fig_play.update_layout(yaxis=dict(tickformat=".0%", range=[-0.05, 1.05]))
        st.plotly_chart(fig_play, use_container_width=True)
        
        fig_cab = px.line(df_pf, x="Jornada", y="Prob. Cab. Serie", color="Equipo", markers=True, title="3. Opciones de ser Cabeza de Serie (Top 4)", color_discrete_map=mapa_colores)
        fig_cab.update_layout(yaxis=dict(tickformat=".0%", range=[-0.05, 1.05]))
        st.plotly_chart(fig_cab, use_container_width=True)
        
        fig_fuera = px.line(df_pf, x="Jornada", y="Prob. Fuera Playoff", color="Equipo", markers=True, title="4. Opciones de quedar Fuera de Playoff (>8)", color_discrete_map=mapa_colores)
        fig_fuera.update_layout(yaxis=dict(tickformat=".0%", range=[-0.05, 1.05]))
        st.plotly_chart(fig_fuera, use_container_width=True)
    else:
        st.warning("Para ver las gráficas de probabilidad, selecciona 'Simular TODAS las jornadas' en la pestaña 1 y pulsa iniciar.")

# --- TAB 3: ESTADÍSTICAS AVANZADAS ---
with tab3:
    st.subheader(f"🏆 Clasificación General (Jornada {JORNADA_MAX_REAL})")
    
    metricas = []
    for eq in dict_historico.keys():
        df_t = df_historico[df_historico["Equipo"] == eq].copy()
        pf, pc, v = df_t["Anotados"].sum(), df_t["Recibidos"].sum(), df_t["Victoria"].sum()
        suerte = v - ((pf**EXPONENTE_SUERTE / (pf**EXPONENTE_SUERTE + pc**EXPONENTE_SUERTE)) * JORNADA_MAX_REAL)
        
        racha_actual = 0
        tipo_racha = None
        for vic in reversed(df_t["Victoria"].tolist()):
            if tipo_racha is None:
                tipo_racha = vic
                racha_actual = 1
            elif vic == tipo_racha:
                racha_actual += 1
            else:
                break
        racha_str = f"{'W' if tipo_racha == 1 else 'L'}{racha_actual}"
        
        metricas.append({
            "Equipo": eq, "V": v, "D": JORNADA_MAX_REAL-v, 
            "Favor": int(pf), "Contra": int(pc), "Diferencial": int(pf-pc), 
            "Media Anotada": df_t["Anotados"].mean(), "Media Recibida": df_t["Recibidos"].mean(), 
            "Racha": racha_str, "Factor Suerte": suerte
        })
        
    df_real = pd.DataFrame(metricas).sort_values(by=["V", "Diferencial"], ascending=False).reset_index(drop=True)
    df_real.index += 1
    
    st.dataframe(
        df_real.style.format({
            "Media Anotada": "{:.1f}", 
            "Media Recibida": "{:.1f}", 
            "Factor Suerte": "{:+.1f}"
        }).background_gradient(cmap="Blues", subset=["V", "Favor", "Media Anotada"])
          .background_gradient(cmap="Reds", subset=["D", "Contra", "Media Recibida"])
          .background_gradient(cmap="RdYlGn", subset=["Diferencial"])
          .background_gradient(cmap="PRGn", subset=["Factor Suerte"]), 
        use_container_width=True, height=650
    )
    
    st.markdown("---")
    st.subheader("Métricas individuales e historial")
    
    eq_ver = st.selectbox("Selecciona un equipo para el gráfico:", list(dict_historico.keys()))
    df_eq = df_historico[df_historico["Equipo"] == eq_ver].copy()
    
    m_a, m_r, std_a = df_eq["Anotados"].mean(), df_eq["Recibidos"].mean(), df_eq["Anotados"].std()
    df_eq["Margen"] = df_eq["Anotados"] - df_eq["Recibidos"]
    max_jornada = df_historico.groupby("Jornada")["Anotados"].max()
    min_jornada = df_historico.groupby("Jornada")["Anotados"].min()
    
    v_poco = len(df_eq[(df_eq["Margen"] > 0) & (df_eq["Margen"] < 5)])
    d_poco = len(df_eq[(df_eq["Margen"] < 0) & (df_eq["Margen"] > -5)])
    veces_mejor = sum(df_eq.apply(lambda row: row["Anotados"] == max_jornada[row["Jornada"]], axis=1))
    veces_peor = sum(df_eq.apply(lambda row: row["Anotados"] == min_jornada[row["Jornada"]], axis=1))
    
    st.markdown(f"#### Balance General: {eq_ver}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Media Anotada", f"{m_a:.1f} pts")
    c2.metric("Media Recibida", f"{m_r:.1f} pts")
    c3.metric("Diferencial", f"{m_a - m_r:+.1f} pts")
    c4.metric("Desviación Estándar", f"{std_a:.1f} pts")
    
    st.markdown("#### Datos de clutch y MVP")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Mejor de la Jornada", f"{veces_mejor} veces")
    c6.metric("Peor de la Jornada", f"{veces_peor} veces")
    c7.metric("Victorias <5 pts", f"{v_poco} partidos")
    c8.metric("Derrotas <5 pts", f"{d_poco} partidos")
    
    st.markdown("---")
    df_plot = df_eq.melt(id_vars=["Jornada"], value_vars=["Anotados", "Recibidos"], var_name="Tipo", value_name="Puntos")
    fig_pts = px.bar(df_plot, x="Jornada", y="Puntos", color="Tipo", barmode="group", text="Puntos", color_discrete_map={"Anotados": "#1f77b4", "Recibidos": "#d62728"})
    fig_pts.add_hline(y=m_a, line_dash="dot", line_color="#1f77b4")
    fig_pts.add_hline(y=m_r, line_dash="dot", line_color="#d62728")
    fig_pts.update_traces(textposition='outside')
    fig_pts.update_yaxes(range=[0, df_eq[["Anotados", "Recibidos"]].max().max() * 1.15])
    st.plotly_chart(fig_pts, use_container_width=True)
