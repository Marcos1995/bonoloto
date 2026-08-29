"""Interfaz web del Lotto Optimizer V3 (Streamlit)."""
import streamlit as st

from JamCat61 import Configuracion, LottoOptimizerV3, Velocidad

st.set_page_config(page_title="Bonoloto Optimizer", page_icon="🎰", layout="centered")
st.title("🎰 Lotto Optimizer V3")
st.caption("Bonoloto / covering — corre en la nube desde el navegador.")

perfiles = [v for v in Velocidad if v.nombre != "USUARIO"]

with st.form("optimizar"):
    c1, c2, c3, c4 = st.columns(4)
    v = c1.number_input("v (números)", min_value=2, max_value=90, value=49)
    k = c2.number_input("k (por apuesta)", min_value=1, max_value=20, value=6)
    t = c3.number_input("t (garantía)", min_value=1, max_value=10, value=3)
    m = c4.number_input("m (sorteo)", min_value=1, max_value=20, value=6)

    n_apuestas = st.slider("Apuestas iniciales", 5, 200, 30)
    segundos = st.slider("Segundos de optimización", 5, 60, 15)
    universo = st.select_slider("Sorteos simulados", options=[2000, 5000, 10000, 20000], value=5000)
    perfil = st.selectbox("Velocidad", [p.nombre for p in perfiles], index=1)
    lanzar = st.form_submit_button("Optimizar", type="primary")

if not lanzar:
    st.info("Pulsa **Optimizar**. En la nube el arranque puede tardar un minuto.")
    st.stop()

if t > k or t > m or k > v or m > v:
    st.error("Parámetros inválidos: hace falta t ≤ k, t ≤ m, k ≤ v y m ≤ v.")
    st.stop()

velocidad = next(p for p in perfiles if p.nombre == perfil)

with st.spinner("Generando universo y optimizando…"):
    config = Configuracion(v=int(v), k=int(k), t=int(t), m=int(m), universo_size=int(universo))
    opt = LottoOptimizerV3(config)
    opt.aplicar_velocidad(velocidad)
    opt.generar_aleatorias(int(n_apuestas))
    cob_ini = opt.cobertura
    opt.optimizar(max_segundos=float(segundos), interactivo=False)

apuestas = opt.apuestas_como_listas()
texto = "\n".join(" ".join(f"{n:02d}" for n in fila) for fila in apuestas)

m1, c2, c3 = st.columns(3)
m1.metric("Cobertura", f"{opt.cobertura:.2f}%", f"{opt.cobertura - cob_ini:+.2f}")
c2.metric("Apuestas", f"{opt.num_apuestas}")
c3.metric("Ciclos", f"{opt.ciclos:,}")

st.download_button("Descargar apuestas", texto + "\n", file_name="apuestas_bonoloto.txt")
st.text_area("Apuestas", texto, height=280)
