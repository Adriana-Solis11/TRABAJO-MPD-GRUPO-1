import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title='Predicción de depósitos bancarios', page_icon='🏦', layout='wide')


@st.cache_resource(show_spinner='Entrenando modelos predictivos...')
def entrenar():
    from pathlib import Path
    ruta = Path(__file__).resolve().parent
    df = pd.read_csv(ruta / 'bank.csv')
    X = df[['age', 'balance', 'duration']]
    y = df['deposit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    modelos = {
        'Árbol de Decisiones': DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=500, random_state=42),
        'SVM': Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))]),
        'Regresión Logística L1': Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', LogisticRegression(
                penalty='l1',
                solver='liblinear',
                C=1.0,
                class_weight='balanced',
                random_state=42
            ))
        ]),
    }
    metricas = []
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        metricas.append({'Modelo': nombre, 'Accuracy': accuracy_score(y_test, pred), 'Precisión (sí)': precision_score(y_test, pred, pos_label='yes'), 'Recall (sí)': recall_score(y_test, pred, pos_label='yes'), 'F1 (sí)': f1_score(y_test, pred, pos_label='yes')})

    # Parámetros para la simulación: matriz de confusión del SVM en el conjunto de prueba
    pred_svm = modelos['SVM'].predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_svm, labels=['no', 'yes']).ravel()
    param_svm = {
        'prop_A': (tp + fp) / len(y_test),        # clientes que el SVM clasifica como "sí" (Grupo A)
        'p_A': tp / (tp + fp),                    # de esos, cuántos contrataron realmente
        'p_B': fn / (fn + tn),                    # clientes clasificados "no" que sí contrataron
        'matriz': (int(tn), int(fp), int(fn), int(tp)),
        'n_test': int(len(y_test)),
    }
    return df, modelos, pd.DataFrame(metricas), param_svm


# ---------------------------------------------------------------------------
# Simulación de eventos discretos del call center (réplica del modelo en Arena)
# ---------------------------------------------------------------------------
def simular_jornada(seed, prioridad, n_clientes, agentes, minutos, prop_A, p_A, p_B,
                    distribucion, lista_completa):
    """Una réplica. Devuelve llamadas, contratos A, contratos B y espera media (min).
    prioridad=False -> FIFO; True -> Lowest Attribute Value (GRUPO), el Grupo A primero."""
    rng = np.random.default_rng(seed)
    grupo = np.where(rng.random(n_clientes) < prop_A, 1, 2)
    u = rng.random(n_clientes)
    if distribucion.startswith('Lognormal'):
        media, desv = 6.59, 8.18                  # LOGN(6.59, 8.18) del Input Analyzer
        s2 = np.log(1 + (desv / media) ** 2)
        servicio = rng.lognormal(np.log(media) - s2 / 2, np.sqrt(s2), n_clientes)
    else:
        servicio = rng.exponential(6.2, n_clientes)  # EXPO(6.2)
    llegada = np.zeros(n_clientes) if lista_completa else np.cumsum(rng.exponential(1.0, n_clientes))

    pendientes = list(np.argsort(llegada, kind='stable'))
    cola, libre = [], [0.0] * agentes
    llamadas = cA = cB = 0
    esperas = []
    while True:
        a = int(np.argmin(libre))
        t = libre[a]
        if t >= minutos:
            break
        while pendientes and llegada[pendientes[0]] <= t:
            cola.append(pendientes.pop(0))
        if not cola:
            if not pendientes:
                break
            libre[a] = llegada[pendientes[0]]
            continue
        if prioridad:
            cola.sort(key=lambda k: (grupo[k], llegada[k]))
        k = cola.pop(0)
        fin = t + servicio[k]
        if fin > minutos:                          # la llamada no termina dentro de la jornada
            libre[a] = np.inf
            continue
        libre[a] = fin
        llamadas += 1
        esperas.append(t - llegada[k])
        if u[k] <= (p_A if grupo[k] == 1 else p_B):
            if grupo[k] == 1:
                cA += 1
            else:
                cB += 1
    return llamadas, cA, cB, (np.mean(esperas) if esperas else 0.0)


@st.cache_data(show_spinner='Simulando réplicas...')
def correr_escenario(replicas, prioridad, **kw):
    res = np.array([simular_jornada(s, prioridad, **kw) for s in range(replicas)], dtype=float)
    return pd.DataFrame(res, columns=['Llamadas', 'Contratos A', 'Contratos B', 'Espera media (min)'])


def media_ic(x):
    from scipy import stats
    n = len(x)
    hw = stats.t.ppf(0.975, n - 1) * np.std(x, ddof=1) / np.sqrt(n)
    return float(np.mean(x)), float(hw)


df, modelos, metricas, param_svm = entrenar()
st.title('🏦 Predicción de contratación de depósitos a plazo')
st.write('Aplicación de Machine Learning basada en el estudio de clientes bancarios.')
pestana1, pestana2, pestana3 = st.tabs(['🔮 Predicción individual', '📊 Evaluación de modelos', '📞 Simulación del call center'])

with pestana1:
    st.subheader('Datos del cliente')
    st.caption('Ingresa los valores del cliente para estimar si contratará un depósito a plazo.')
    c1, c2, c3 = st.columns(3)
    with c1: edad = st.number_input('Edad (años)', min_value=18, max_value=100, value=41)
    with c2: saldo = st.number_input('Saldo bancario (unidades monetarias del dataset)', value=1500, step=100)
    with c3: duracion = st.number_input('Duración del contacto (segundos)', min_value=0, value=370)
    nombre = st.selectbox('Modelo de predicción', list(modelos), index=2)
    if st.button('Predecir contratación', type='primary'):
        entrada = pd.DataFrame([{'age': edad, 'balance': saldo, 'duration': duracion}])
        modelo = modelos[nombre]
        pred = modelo.predict(entrada)[0]
        indice_si = list(modelo.classes_).index('yes')
        probabilidad = float(modelo.predict_proba(entrada)[0][indice_si])
        st.success('Predicción: SÍ contrataría el depósito' if pred == 'yes' else 'Predicción: NO contrataría el depósito')
        st.metric('Probabilidad estimada de contratación', f'{probabilidad:.1%}')
        st.progress(probabilidad)
        st.caption('Probabilidad estimada por el modelo, no una garantía. La duración del contacto se conoce durante o después de la llamada; no es adecuada para priorizar llamadas antes de realizarlas.')

with pestana2:
    st.subheader('Comparación en el conjunto de prueba (20%)')
    st.dataframe(metricas.style.format({c: '{:.2%}' for c in metricas.columns if c != 'Modelo'}), hide_index=True, use_container_width=True)
    grafico = metricas.set_index('Modelo')[['Accuracy', 'F1 (sí)']]
    try:
        st.bar_chart(grafico, stack=False)       # barras lado a lado (no sumar métricas)
    except TypeError:                            # versiones antiguas de Streamlit
        st.bar_chart(grafico.T)
    st.caption('Misma partición estratificada 80/20 (random_state=42). Las métricas se calculan al ejecutar la aplicación.')

with pestana3:
    st.subheader('¿Cuánto vale el modelo en la operación del call center?')
    st.write('Réplica en Python del modelo de **Arena**: el SVM clasifica a cada cliente en **Grupo A** (probable comprador) '
             'o **Grupo B**; los agentes atienden las llamadas y se compara atender **por orden de llegada (FIFO)** '
             'frente a **priorizar al Grupo A**.')

    p = param_svm
    k1, k2, k3 = st.columns(3)
    k1.metric('Clientes en Grupo A', f"{p['prop_A']:.1%}")
    k2.metric('P(contratar) Grupo A', f"{p['p_A']:.2f}")
    k3.metric('P(contratar) Grupo B', f"{p['p_B']:.2f}")
    tn, fp, fn, tp = p['matriz']
    st.caption(f"Calculado de la matriz de confusión del SVM en {p['n_test']} clientes de prueba: "
               f"TP={tp}, FP={fp}, FN={fn}, TN={tn}. Grupo A = predichos 'sí'.")

    st.markdown('**Parámetros del escenario**')
    c1, c2, c3 = st.columns(3)
    with c1:
        n_clientes = st.slider('Clientes en la lista', 50, 500, 140, 10)
        agentes = st.slider('Agentes de telemarketing', 1, 10, 2)
    with c2:
        horas = st.slider('Duración de la jornada (horas)', 1.0, 8.0, 4.0, 0.5)
        replicas = st.slider('Réplicas', 10, 50, 30, 5)
    with c3:
        distribucion = st.radio('Tiempo de atención', ['Lognormal LOGN(6.59, 8.18) — ajuste Input Analyzer', 'Exponencial EXPO(6.2)'])
        lista_completa = st.checkbox('Lista completa disponible al inicio (campaña saliente)', value=True)

    capacidad = agentes * horas * 60 / 6.2
    if capacidad >= n_clientes:
        st.info(f'La jornada alcanza para unas {capacidad:.0f} llamadas: se puede llamar a toda la lista, '
                'así que priorizar casi no cambiará el resultado.')
    else:
        st.warning(f'La jornada alcanza para unas {capacidad:.0f} llamadas de {n_clientes} clientes: '
                   'la lista supera la capacidad, que es el caso real de la campaña (11 162 clientes).')

    kw = dict(n_clientes=n_clientes, agentes=agentes, minutos=horas * 60, prop_A=p['prop_A'], p_A=p['p_A'],
              p_B=p['p_B'], distribucion=distribucion, lista_completa=lista_completa)
    fifo = correr_escenario(replicas, False, **kw)
    prio = correr_escenario(replicas, True, **kw)
    for r in (fifo, prio):
        r['Contratos'] = r['Contratos A'] + r['Contratos B']
        r['Conversión'] = r['Contratos'] / r['Llamadas'].where(r['Llamadas'] > 0)

    filas = []
    for nombre_esc, r in [('Sin modelo (FIFO)', fifo), ('Con modelo (prioridad Grupo A)', prio)]:
        fila = {'Escenario': nombre_esc}
        for col in ['Llamadas', 'Contratos', 'Contratos A', 'Contratos B', 'Conversión', 'Espera media (min)']:
            m, hw = media_ic(r[col].fillna(0))
            fila[col] = f'{m:.1%} ± {hw:.1%}' if col == 'Conversión' else f'{m:.1f} ± {hw:.1f}'
        filas.append(fila)

    dif = prio['Contratos'] - fifo['Contratos']    # mismas semillas -> comparación pareada
    m_dif, hw_dif = media_ic(dif)
    m_fifo = fifo['Contratos'].mean()
    r1, r2, r3 = st.columns(3)
    r1.metric('Contratos sin modelo', f'{m_fifo:.1f}')
    r2.metric('Contratos con modelo', f"{prio['Contratos'].mean():.1f}",
              delta=f'{m_dif:+.1f} ({m_dif / m_fifo:+.0%})' if m_fifo > 0 else None)
    r3.metric('Conversión con modelo', f"{prio['Conversión'].mean():.1%}",
              delta=f"{(prio['Conversión'].mean() - fifo['Conversión'].mean()) * 100:+.1f} pp")

    st.bar_chart(pd.DataFrame({'Contratos por jornada': [m_fifo, prio['Contratos'].mean()]},
                              index=['Sin modelo (FIFO)', 'Con modelo (prioridad A)']))
    st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)

    if hw_dif < abs(m_dif):
        st.success(f'Diferencia pareada: {m_dif:+.1f} ± {hw_dif:.1f} contratos (IC 95 %). '
                   'El intervalo no incluye el 0: la mejora es estadísticamente significativa.')
    else:
        st.info(f'Diferencia pareada: {m_dif:+.1f} ± {hw_dif:.1f} contratos (IC 95 %). '
                'El intervalo incluye el 0: con estos parámetros la diferencia no es significativa.')

    with st.expander('Resultados obtenidos en Arena (30 réplicas, EXPO(6.2))'):
        st.dataframe(pd.DataFrame({
            'Escenario': ['A: 150 clientes · 2 agentes · 8 h', 'B: 140 clientes · 2 agentes · 4 h'],
            'Contratos FIFO': ['68,8 ± 2,5', '37,2 ± 2,1'],
            'Contratos prioridad A': ['69,5 ± 2,2', '51,9 ± 1,6'],
            'Mejora': ['+1 %', '+40 %'],
        }), hide_index=True, use_container_width=True)
        st.caption('La simulación de esta pestaña reproduce la lógica del modelo de Arena; los valores pueden variar '
                   'ligeramente por los números aleatorios. Limitación: el SVM usa la duración de la llamada, que solo '
                   'se conoce después de llamar; el modelo pre-llamada de la dimensión Power BI aborda este punto.')

if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx(suppress_warning=True) is None:
        import subprocess
        import sys
        from pathlib import Path

        subprocess.Popen([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).resolve())
        ])
