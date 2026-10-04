# ============================================================
# MOTOR INTELIGENTE DE FORMULACIÓN AGROINDUSTRIAL - WEB
# ============================================================

import io
import os
import smtplib
import tempfile
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from scipy.optimize import linprog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image as RLImage, PageBreak
)


# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Motor de Formulación Agroindustrial",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #8EA1B8;
        font-size: 16px;
        margin-top: 5px;
    }
    .card {
        background: #17243A;
        border: 1px solid #243653;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }
    .badge {
        display: inline-block;
        background: #1B2A44;
        color: #FACC15;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 13px;
        margin: 4px 0;
    }
    .emojis {
        font-size: 24px;
        letter-spacing: 6px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #38BDF8, #8B5CF6);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 28px;
        border-radius: 10px;
        font-size: 15px;
    }
    .stButton > button:hover {
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATOS BASE
# ============================================================

@st.cache_data
def cargar_datos_base():
    df = pd.DataFrame({
        "Materia_Prima": [
            "Avena en hojuelas", "Mantequilla de maní", "Miel",
            "Semillas de chía", "Almendras molidas"
        ],
        "Proteina_%": [13.2, 25.0, 0.3, 16.5, 21.2],
        "Grasa_%": [6.9, 50.0, 0.0, 30.7, 49.9],
        "Fibra_%": [10.6, 6.0, 0.2, 34.4, 12.5],
        "Cenizas_%": [1.8, 2.5, 0.2, 4.8, 3.0],
        "Precio_Soles_kg": [4.80, 18.50, 22.00, 28.00, 38.00],
        "Disponible_kg": [400, 200, 150, 100, 120]
    })
    df["ELN_%"] = 100 - df["Proteina_%"] - df["Grasa_%"] - df["Fibra_%"] - df["Cenizas_%"]
    df["Energia_kcal_kg"] = 10 * (
        3.5 * df["Proteina_%"] + 8.5 * df["Grasa_%"] + 3.5 * df["ELN_%"]
    )
    return df


PROTEINA_MIN = 12.0
GRASA_MIN = 18.0
GRASA_MAX = 28.0
FIBRA_MIN = 6.0
ENERGIA_MIN = 3800
ENERGIA_MAX = 4800

LIMITES_BASE = [
    (0.25, 0.55), (0.15, 0.35), (0.05, 0.20),
    (0.05, 0.18), (0.03, 0.15)
]


# ============================================================
# GAUSS-JORDAN
# ============================================================

def gauss_jordan(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    matriz = np.hstack((A, b))
    n = matriz.shape[0]
    for i in range(n):
        pivote = i + np.argmax(np.abs(matriz[i:, i]))
        matriz[[i, pivote]] = matriz[[pivote, i]]
        if abs(matriz[i, i]) < 1e-12:
            raise ValueError("Sistema sin solución única.")
        matriz[i] = matriz[i] / matriz[i, i]
        for j in range(n):
            if j != i:
                matriz[j] = matriz[j] - matriz[j, i] * matriz[i]
    return matriz[:, -1]


# ============================================================
# RANDOM FOREST
# ============================================================

@st.cache_resource
def entrenar_random_forest():
    np.random.seed(42)
    X, y = [], []

    def evaluar(p, g, f, e):
        return int(
            p >= PROTEINA_MIN and GRASA_MIN <= g <= GRASA_MAX
            and f >= FIBRA_MIN and ENERGIA_MIN <= e <= ENERGIA_MAX
        )

    for _ in range(20000):
        p = np.random.uniform(8, 30)
        g = np.random.uniform(10, 35)
        f = np.random.uniform(2, 20)
        e = np.random.uniform(3000, 5500)
        X.append([p, g, f, e])
        y.append(evaluar(p, g, f, e))

    casos_limite = [
        [12.0, 18.0, 6.0, 3800], [12.0, 28.0, 6.0, 4800],
        [16.0, 24.0, 9.0, 4500], [11.9, 18.0, 6.0, 4000],
        [12.0, 17.9, 6.0, 4000], [12.0, 18.0, 5.9, 4000],
        [12.0, 18.0, 6.0, 3799], [12.0, 28.1, 6.0, 4000]
    ]
    for c in casos_limite:
        X.append(c)
        y.append(evaluar(*c))

    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    modelo = RandomForestClassifier(
        n_estimators=300, max_depth=15,
        random_state=42, class_weight="balanced", n_jobs=-1
    )
    modelo.fit(X_train, y_train)
    precision = accuracy_score(y_test, modelo.predict(X_test))
    return modelo, precision


# ============================================================
# CORREO
# ============================================================

def obtener_credenciales_correo():
    """Lee credenciales desde st.secrets o variables de entorno."""
    try:
        correo = st.secrets["correo"]["email"]
        password = st.secrets["correo"]["app_password"]
    except Exception:
        correo = os.environ.get("NUTRIBAR_EMAIL", "")
        password = os.environ.get("NUTRIBAR_APP_PASSWORD", "")
    return correo, password


def enviar_reporte_correo(destinatario, pdf_buffer):
    """Envía el PDF por correo usando Gmail SMTP con SSL (puerto 465)."""
    correo_empresa, password_app = obtener_credenciales_correo()

    if not correo_empresa or not password_app:
        return False, (
            "El administrador no ha configurado las credenciales de correo.\n"
            "Contacta al responsable del sistema."
        )

    try:
        mensaje = MIMEMultipart()
        mensaje["Subject"] = "🍫 Tu reporte – Motor de Formulación Agroindustrial"
        mensaje["From"] = correo_empresa
        mensaje["To"] = destinatario

        cuerpo = f"""
Hola,

Adjunto encontrarás el reporte de tu formulación de barras energéticas
generado por el Motor Inteligente de Formulación Agroindustrial.

El informe incluye:
• Formulación optimizada
• Costos de producción
• Información nutricional
• Evaluación con Random Forest
• Gráficos de análisis

Motor de Formulación Agroindustrial
Optimización de Costos y Calidad
"""
        mensaje.attach(MIMEText(cuerpo, "plain", "utf-8"))

        pdf_buffer.seek(0)
        adjunto = MIMEApplication(pdf_buffer.read(), _subtype="pdf")
        adjunto.add_header(
            "Content-Disposition",
            "attachment",
            filename="Reporte_Motor_Agroindustrial.pdf"
        )
        mensaje.attach(adjunto)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as servidor:
            servidor.login(correo_empresa, password_app)
            servidor.send_message(mensaje)

        return True, "✓ Reporte enviado correctamente."

    except smtplib.SMTPAuthenticationError:
        return False, (
            "Error de autenticación.\n"
            "Verifica que la contraseña sea una contraseña de aplicación de Gmail."
        )
    except Exception as e:
        return False, f"Error al enviar: {e}"


# ============================================================
# GRÁFICOS
# ============================================================

def crear_graficos(r, datos):
    nombres = datos["Materia_Prima"].values
    cantidades = r["cantidades_kg"]
    precios = datos["Precio_Soles_kg"].values
    costos = cantidades * precios
    x = np.arange(len(nombres))

    figs = {}

    fig1 = plt.Figure(figsize=(8, 5), dpi=100)
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.bar3d(x, np.zeros(len(x)), np.zeros(len(x)), 0.6, 0.6, cantidades)
    ax1.set_xticks(x + 0.3)
    ax1.set_xticklabels(nombres, rotation=15, ha="right", fontsize=8)
    ax1.set_zlabel("Cantidad (kg)")
    ax1.set_title("Cantidad de materias primas", fontweight="bold")
    figs["Cantidad"] = fig1

    fig2 = plt.Figure(figsize=(8, 5), dpi=100)
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.bar3d(x, np.zeros(len(x)), np.zeros(len(x)), 0.6, 0.6, costos)
    ax2.set_xticks(x + 0.3)
    ax2.set_xticklabels(nombres, rotation=15, ha="right", fontsize=8)
    ax2.set_zlabel("Costo (S/)")
    ax2.set_title("Costo por materia prima", fontweight="bold")
    figs["Costos"] = fig2

    fig3 = plt.Figure(figsize=(8, 5), dpi=100)
    ax3 = fig3.add_subplot(111, projection="3d")
    np.random.seed(42)
    for _ in range(40):
        p = np.random.uniform(8, 30)
        g = np.random.uniform(10, 35)
        e = np.random.uniform(300, 550)
        ax3.scatter(p, g, e, alpha=0.35, s=20)
    ax3.scatter(r["proteina"], r["grasa"], r["energia"]/10,
                s=180, color="red", label="Formulación")
    ax3.set_xlabel("Proteína (%)")
    ax3.set_ylabel("Grasa (%)")
    ax3.set_zlabel("Energía (kcal/100g)")
    ax3.set_title("Espacio nutricional 3D", fontweight="bold")
    ax3.legend()
    figs["Nutricion 3D"] = fig3

    categorias = ["Proteína", "Grasa", "Fibra", "Energía"]
    valores = [
        r["proteina"]/25, r["grasa"]/30,
        r["fibra"]/15, (r["energia"]/10)/500
    ]
    valores += valores[:1]
    angulos = np.linspace(0, 2*np.pi, len(categorias), endpoint=False).tolist()
    angulos += angulos[:1]

    fig4 = plt.Figure(figsize=(7, 6), dpi=100)
    ax4 = fig4.add_subplot(111, polar=True)
    ax4.plot(angulos, valores, "o-", linewidth=2)
    ax4.fill(angulos, valores, alpha=0.2)
    ax4.set_xticks(angulos[:-1])
    ax4.set_xticklabels(categorias)
    ax4.set_title("Perfil nutricional", fontweight="bold")
    figs["Perfil nutricional"] = fig4

    fig5 = plt.Figure(figsize=(8, 6), dpi=100)
    ax5 = fig5.add_subplot(111)
    ax5.pie(r["proporcion"]*100, labels=nombres,
            autopct="%1.1f%%", startangle=90)
    ax5.set_title("Composición de la formulación", fontweight="bold")
    figs["Composición"] = fig5

    return figs


# ============================================================
# PDF
# ============================================================

def generar_pdf_bytes(r, formulacion, figs):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30
    )
    styles = getSampleStyleSheet()
    titulo = ParagraphStyle("T", parent=styles["Heading1"], fontSize=19,
        textColor=colors.HexColor("#1A365D"), alignment=TA_CENTER, spaceAfter=15)
    sub = ParagraphStyle("S", parent=styles["Heading2"], fontSize=12,
        textColor=colors.HexColor("#2563EB"), spaceBefore=10, spaceAfter=8)
    normal = ParagraphStyle("N", parent=styles["Normal"], fontSize=9)

    story = []
    story.append(Paragraph("🍫 MOTOR DE FORMULACIÓN AGROINDUSTRIAL", titulo))
    story.append(Paragraph("REPORTE DE FORMULACIÓN DE BARRAS ENERGÉTICAS", titulo))
    story.append(Paragraph("Optimización de Costos y Calidad", normal))
    story.append(Spacer(1, 15))

    story.append(Paragraph("📋 Resumen del lote", sub))
    resumen = [
        ["Cantidad", f"{r['cantidad']:.2f} kg", "Peso barra", f"{r['peso_barrita_g']:.1f} g"],
        ["Barras", f"{r['num_barritas']:,}", "Costo total", f"S/ {r['costo_total']:,.2f}"],
        ["Costo/kg", f"S/ {r['costo_kg']:.2f}", "Costo/barra", f"S/ {r['costo_barrita']:.2f}"]
    ]
    t = Table(resumen, colWidths=[120, 120, 120, 120])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#EDF2F7")),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTSIZE", (0,0), (-1,-1), 9)
    ]))
    story.append(t)

    story.append(Paragraph("🍫 Formulación optimizada", sub))
    filas = [["Materia prima", "Prop. (%)", "Cantidad (kg)", "Precio", "Costo"]]
    for _, row in formulacion.iterrows():
        filas.append([
            str(row["Materia Prima"]), f"{row['Proporcion (%)']:.2f}%",
            f"{row['Cantidad (kg)']:.2f}",
            f"S/ {row['Precio (S/kg)']:.2f}",
            f"S/ {row['Costo (S/)']:.2f}"
        ])
    t2 = Table(filas, colWidths=[155, 80, 90, 85, 90])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2563EB")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ("ALIGN", (1,0), (-1,-1), "CENTER"),
        ("FONTSIZE", (0,0), (-1,-1), 8)
    ]))
    story.append(t2)

    story.append(Paragraph("🥗 Características nutricionales", sub))
    nut = [
        ["Proteína", f"{r['proteina']:.2f} %"],
        ["Grasa", f"{r['grasa']:.2f} %"],
        ["Fibra", f"{r['fibra']:.2f} %"],
        ["Energía", f"{r['energia']/10:.1f} kcal/100 g"],
        ["Proteína/barra", f"{r['proteina_barrita']:.2f} g"],
        ["Grasa/barra", f"{r['grasa_barrita']:.2f} g"],
        ["Fibra/barra", f"{r['fibra_barrita']:.2f} g"],
        ["Energía/barra", f"{r['energia_barrita']:.1f} kcal"]
    ]
    t3 = Table(nut, colWidths=[200, 250])
    t3.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ("FONTSIZE", (0,0), (-1,-1), 9)
    ]))
    story.append(t3)

    story.append(Paragraph("🤖 Evaluación e inteligencia artificial", sub))
    story.append(Paragraph(f"Evaluación: <b>{r['resultado_final']}</b>", normal))
    story.append(Paragraph(f"Random Forest: <b>{r['resultado_rf']}</b>", normal))
    story.append(Paragraph(f"Confianza: {r['confianza']:.2f} %", normal))
    story.append(Paragraph(f"Precisión: {r['precision_rf']*100:.2f} %", normal))

    story.append(PageBreak())
    story.append(Paragraph("📊 ANÁLISIS GRÁFICO", titulo))

    for nombre, fig in figs.items():
        tmp = os.path.join(tempfile.gettempdir(), f"agro_{nombre.replace(' ','_')}.png")
        fig.savefig(tmp, dpi=180, bbox_inches="tight")
        story.append(Paragraph(nombre, sub))
        story.append(RLImage(tmp, width=430, height=280))
        story.append(Spacer(1, 8))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ============================================================
# ESTADO DE SESIÓN
# ============================================================

if "datos" not in st.session_state:
    st.session_state.datos = cargar_datos_base()
if "resultado" not in st.session_state:
    st.session_state.resultado = None
if "formulacion" not in st.session_state:
    st.session_state.formulacion = None
if "figuras" not in st.session_state:
    st.session_state.figuras = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;">
        <div style="font-size:50px;">🍫</div>
        <div class="main-title" style="font-size:22px;">Motor</div>
        <div style="color:#38BDF8; font-weight:bold; font-size:16px;">Agroindustrial</div>
        <div style="color:#8EA1B8; font-size:11px; margin-top:6px;">
            Optimización de<br>Costos y Calidad
        </div>
        <div class="emojis" style="font-size:16px; margin-top:12px;">
            🍫 🍪 🥜 🌾 🍯
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    opcion = st.radio(
        "Menú",
        ["🏠 Inicio", "⚙ Formulación", "🌾 Materias primas", "📊 Resultados", "💾 Descargar app"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#8EA1B8; font-size:11px;'>"
        "Métodos Numéricos • 2026<br>"
        "LP · Random Forest · Gauss-Jordan"
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# VISTA: INICIO
# ============================================================

if opcion == "🏠 Inicio":
    st.markdown('<div class="main-title">🍫 Motor de Formulación Agroindustrial</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Optimización de Costos y Calidad en barras energéticas mediante métodos numéricos e inteligencia artificial</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🍫 🍪 🥜 🌾 🍯 🌰")
        st.markdown("## Formula. Optimiza. Decide.")
        st.markdown(
            "Motor inteligente diseñado para encontrar la formulación "
            "óptima de barras energéticas que cumpla criterios nutricionales "
            "y minimice el costo de producción."
        )
        if st.button("⚡ COMENZAR FORMULACIÓN"):
            st.session_state.menu_forzado = "formulacion"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:50px;">🍫</div>
            <div style="font-size:28px;">🥜 🍯 🌰</div>
            <div style="font-size:28px;">🌾 🥣 🍪</div>
            <div style="margin-top:15px; font-weight:bold; color:#38BDF8;">
                BARRAS ENERGÉTICAS
            </div>
            <div class="badge">LP · AI · G-J</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🍫 ¿Cómo funciona el motor?")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="card">
            <div style="color:#8B5CF6; font-weight:bold;">01 📥 ENTRADA</div>
            <div style="color:#8EA1B8; font-size:13px; margin-top:8px;">
                Se define el lote, peso de barra y disponibilidad.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
            <div style="color:#38BDF8; font-weight:bold;">02 ∑ MATEMÁTICA</div>
            <div style="color:#8EA1B8; font-size:13px; margin-top:8px;">
                Gauss-Jordan resuelve el sistema base.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card">
            <div style="color:#22C55E; font-weight:bold;">03 ⚙ OPTIMIZACIÓN</div>
            <div style="color:#8EA1B8; font-size:13px; margin-top:8px;">
                Programación lineal minimiza el costo.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="card">
            <div style="color:#F59E0B; font-weight:bold;">04 🤖 IA</div>
            <div style="color:#8EA1B8; font-size:13px; margin-top:8px;">
                Random Forest analiza la calidad nutricional.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# VISTA: FORMULACIÓN
# ============================================================

elif opcion == "⚙ Formulación":
    st.markdown('<div class="main-title">⚙ Nueva formulación</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Configure el lote de barras energéticas que desea optimizar</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🍫 Parámetros del lote")
        cantidad = st.number_input("🍪 Cantidad del lote (kg)", 1.0, 10000.0, 100.0, 10.0)
        peso_barra = st.number_input("⚖ Peso de cada barra (g)", 10.0, 500.0, 50.0, 5.0)

        if st.button("⚡ EJECUTAR MOTOR", use_container_width=True):
            with st.spinner("Optimizando formulación..."):
                try:
                    datos = st.session_state.datos
                    peso_kg = peso_barra / 1000
                    num_barras = int(cantidad / peso_kg)

                    if num_barras <= 0:
                        st.error("El lote no alcanza para producir una barra.")
                        st.stop()

                    precios = datos["Precio_Soles_kg"].values
                    proteina = datos["Proteina_%"].values
                    grasa = datos["Grasa_%"].values
                    fibra = datos["Fibra_%"].values
                    energia = datos["Energia_kcal_kg"].values
                    disp = datos["Disponible_kg"].values
                    n = len(datos)

                    A_gj = np.array([[13.2, 25.0, 0.3], [6.9, 50.0, 0.0], [1.0, 1.0, 1.0]])
                    b_gj = np.array([15.0, 22.0, 1.0])
                    try:
                        sol_gj = gauss_jordan(A_gj, b_gj)
                    except Exception:
                        sol_gj = np.zeros(3)

                    A_ub = np.array([-proteina, -grasa, grasa, -fibra, -energia, energia])
                    b_ub = np.array([-PROTEINA_MIN, -GRASA_MIN, GRASA_MAX,
                                     -FIBRA_MIN, -ENERGIA_MIN, ENERGIA_MAX])

                    bounds = []
                    for i in range(n):
                        mn = LIMITES_BASE[i][0]
                        mx = min(LIMITES_BASE[i][1], disp[i] / cantidad)
                        if mn > mx:
                            st.error(f"Restricción imposible en {datos.loc[i, 'Materia_Prima']}")
                            st.stop()
                        bounds.append((mn, mx))

                    resultado = linprog(
                        c=precios, A_ub=A_ub, b_ub=b_ub,
                        A_eq=np.ones((1, n)), b_eq=[1],
                        bounds=bounds, method="highs"
                    )

                    if not resultado.success:
                        st.error("No se encontró formulación factible con esas restricciones.")
                        st.stop()

                    prop = resultado.x
                    cant_kg = prop * cantidad
                    costo_total = np.sum(cant_kg * precios)
                    costo_kg = costo_total / cantidad
                    costo_barra = costo_kg * peso_kg

                    p_fin = np.sum(prop * proteina)
                    g_fin = np.sum(prop * grasa)
                    f_fin = np.sum(prop * fibra)
                    e_fin = np.sum(prop * energia)

                    factor = peso_barra / 100
                    p_b = p_fin * factor
                    g_b = g_fin * factor
                    f_b = f_fin * factor
                    e_b = (e_fin / 10) * factor

                    calidad = (
                        p_fin >= PROTEINA_MIN and GRASA_MIN <= g_fin <= GRASA_MAX
                        and f_fin >= FIBRA_MIN and ENERGIA_MIN <= e_fin <= ENERGIA_MAX
                    )
                    resultado_final = "ADECUADA" if calidad else "NO ADECUADA"

                    formulacion = pd.DataFrame({
                        "Materia Prima": datos["Materia_Prima"],
                        "Proporcion (%)": prop * 100,
                        "Cantidad (kg)": cant_kg,
                        "Precio (S/kg)": precios,
                        "Costo (S/)": cant_kg * precios
                    })

                    modelo, precision = entrenar_random_forest()
                    datos_form = np.array([[p_fin, g_fin, f_fin, e_fin]])
                    pred = modelo.predict(datos_form)[0]
                    proba = modelo.predict_proba(datos_form)[0]
                    confianza = max(proba) * 100
                    resultado_rf = "ADECUADA" if pred == 1 else "NO ADECUADA"

                    st.session_state.resultado = {
                        "cantidad": cantidad, "peso_barrita_g": peso_barra,
                        "num_barritas": num_barras, "proporcion": prop,
                        "cantidades_kg": cant_kg, "costo_total": costo_total,
                        "costo_kg": costo_kg, "costo_barrita": costo_barra,
                        "proteina": p_fin, "grasa": g_fin,
                        "fibra": f_fin, "energia": e_fin,
                        "proteina_barrita": p_b, "grasa_barrita": g_b,
                        "fibra_barrita": f_b, "energia_barrita": e_b,
                        "resultado_final": resultado_final,
                        "resultado_rf": resultado_rf,
                        "confianza": confianza, "precision_rf": precision,
                        "gauss_jordan": sol_gj
                    }
                    st.session_state.formulacion = formulacion
                    st.session_state.figuras = crear_graficos(
                        st.session_state.resultado, datos
                    )
                    st.success("✓ Formulación optimizada. Ve a 'Resultados'.")
                    st.balloons()

                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        st.markdown("### 🧠 Motor de optimización")
        st.markdown("""
        <div class="card">
            <div style="color:#8B5CF6; font-weight:bold;">01 ∑ Gauss-Jordan</div>
            <div style="color:#8EA1B8; font-size:13px;">Sistema matemático de referencia</div>
        </div>
        <div class="card">
            <div style="color:#38BDF8; font-weight:bold;">02 ⚙ Programación Lineal</div>
            <div style="color:#8EA1B8; font-size:13px;">Minimización del costo</div>
        </div>
        <div class="card">
            <div style="color:#22C55E; font-weight:bold;">03 🤖 Random Forest</div>
            <div style="color:#8EA1B8; font-size:13px;">Clasificación nutricional</div>
        </div>
        <div class="card">
            <div style="color:#F59E0B; font-weight:bold;">04 📊 Visualización</div>
            <div style="color:#8EA1B8; font-size:13px;">Análisis gráfico</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='text-align:center; font-size:22px; margin-top:20px;'>🍫 🥜 🌾 🍯 🌰 🍪</div>",
                    unsafe_allow_html=True)


# ============================================================
# VISTA: MATERIAS PRIMAS
# ============================================================

elif opcion == "🌾 Materias primas":
    st.markdown('<div class="main-title">🌾 Materias primas</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Configura los ingredientes de tus barras energéticas</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    datos = st.session_state.datos

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🥣 Ingredientes", len(datos))
    c2.metric("✎ Editables", "Precio + Stock")
    c3.metric("📦 Stock total", f"{datos['Disponible_kg'].sum():,.0f} kg")
    c4.metric("⚙ Optimización", "LP + IA")

    st.markdown("### 🍫 Tabla editable de materias primas")
    st.caption("Puedes editar el precio y la disponibilidad directamente en la tabla.")

    datos_editados = st.data_editor(
        datos[["Materia_Prima", "Proteina_%", "Grasa_%", "Fibra_%",
               "Precio_Soles_kg", "Disponible_kg"]],
        column_config={
            "Materia_Prima": st.column_config.TextColumn("🍫 Materia Prima", disabled=True),
            "Proteina_%": st.column_config.NumberColumn("💪 Proteína %", disabled=True),
            "Grasa_%": st.column_config.NumberColumn("🥑 Grasa %", disabled=True),
            "Fibra_%": st.column_config.NumberColumn("🌾 Fibra %", disabled=True),
            "Precio_Soles_kg": st.column_config.NumberColumn("💰 Precio S/kg", min_value=0.1, step=0.1),
            "Disponible_kg": st.column_config.NumberColumn("📦 Disponible kg", min_value=0.0, step=1.0)
        },
        use_container_width=True,
        hide_index=True,
        key="editor_materias"
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("💾 GUARDAR CAMBIOS", use_container_width=True):
            st.session_state.datos = datos_editados.copy()
            st.success("Cambios guardados ✓")
    with col_b:
        if st.button("↻ RESTAURAR DATOS", use_container_width=True):
            st.session_state.datos = cargar_datos_base()
            st.success("Datos restaurados ✓")
            st.rerun()


# ============================================================
# VISTA: RESULTADOS
# ============================================================

elif opcion == "📊 Resultados":
    if st.session_state.resultado is None:
        st.warning("⚠ Primero ejecuta una formulación desde 'Formulación'.")
        st.stop()

    st.markdown('<div class="main-title">📊 Resultados de la formulación</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Resumen técnico de la solución optimizada</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    r = st.session_state.resultado

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Costo total", f"S/ {r['costo_total']:,.2f}")
    c2.metric("🍫 Costo por barra", f"S/ {r['costo_barrita']:.2f}")
    c3.metric("📦 Número de barras", f"{r['num_barritas']:,}")
    c4.metric("✓ Evaluación", r["resultado_final"])

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🥗 Perfil nutricional")
        st.markdown(f"""
        <div class="card">
            <div>💪 <b>Proteína:</b> {r['proteina']:.2f} %</div>
            <div>🥑 <b>Grasa:</b> {r['grasa']:.2f} %</div>
            <div>🌾 <b>Fibra:</b> {r['fibra']:.2f} %</div>
            <div>⚡ <b>Energía:</b> {r['energia']/10:.1f} kcal/100g</div>
            <hr style="border-color:#243653;">
            <div>🍫 <b>Proteína/barra:</b> {r['proteina_barrita']:.2f} g</div>
            <div>🍫 <b>Grasa/barra:</b> {r['grasa_barrita']:.2f} g</div>
            <div>🍫 <b>Fibra/barra:</b> {r['fibra_barrita']:.2f} g</div>
            <div>🍫 <b>Energía/barra:</b> {r['energia_barrita']:.1f} kcal</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🤖 Inteligencia artificial")
        color = "#22C55E" if r["resultado_rf"] == "ADECUADA" else "#EF4444"
        st.markdown(f"""
        <div class="card">
            <div style="font-size:15px; color:#8EA1B8;">🌲 Random Forest</div>
            <div style="font-size:28px; font-weight:bold; color:{color}; margin:10px 0;">
                {r['resultado_rf']}
            </div>
            <div>🎯 Confianza: {r['confianza']:.2f} %</div>
            <div>📈 Precisión del modelo: {r['precision_rf']*100:.2f} %</div>
            <hr style="border-color:#243653;">
            <div>∑ Gauss-Jordan (referencia):</div>
            <div style="font-size:12px; color:#8EA1B8;">
                {', '.join([f'{v:.3f}' for v in r['gauss_jordan']])}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🍫 Formulación detallada")
    st.dataframe(
        st.session_state.formulacion.style.format({
            "Proporcion (%)": "{:.2f}%",
            "Cantidad (kg)": "{:.2f}",
            "Precio (S/kg)": "S/ {:.2f}",
            "Costo (S/)": "S/ {:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.markdown("### 📈 Análisis gráfico")

    figs = st.session_state.figuras
    if figs:
        tabs = st.tabs(list(figs.keys()))
        for tab, (nombre, fig) in zip(tabs, figs.items()):
            with tab:
                st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 📄 Exportar reporte")

    pdf_buffer = generar_pdf_bytes(r, st.session_state.formulacion, figs)

    col_a, col_b = st.columns(2)

    with col_a:
        st.download_button(
            label="📄 DESCARGAR REPORTE PDF",
            data=pdf_buffer,
            file_name="Reporte_Motor_Agroindustrial.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    with col_b:
        with st.expander("📧 ENVIAR POR CORREO", expanded=False):
            correo_destino = st.text_input(
                "Tu correo electrónico",
                placeholder="tucorreo@gmail.com",
                key="correo_destino"
            )

            if st.button("✉ ENVIAR REPORTE", use_container_width=True):
                if "@" not in correo_destino:
                    st.error("Ingresa un correo válido.")
                else:
                    with st.spinner("Enviando correo..."):
                        exito, msg = enviar_reporte_correo(
                            correo_destino, pdf_buffer
                        )
                    if exito:
                        st.success(msg)
                        st.balloons()
                    else:
                        st.error(msg)



# ============================================================
# VISTA: DESCARGAR APP
# ============================================================

elif opcion == "💾 Descargar app":
    st.markdown('<div class="main-title">💾 Descarga la aplicación</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Lleva el Motor Agroindustrial a tu PC para trabajar sin internet</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    URL_EXE = "https://github.com/motorparaoptimizarcostosycalidad/motor_agro_web/releases/download/v1.1/MotorAgroindustrial.exe"

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 💻 Versión de escritorio (Windows)")
        st.markdown("""
        <div class="card">
            <div style="font-size:50px; text-align:center;">🍫</div>
            <h3 style="color:#38BDF8; margin-top:15px; text-align:center;">
                Motor Agroindustrial v1.1
            </h3>
            <p>✅ Motor completo: Gauss-Jordan + Programación Lineal + Random Forest</p>
            <p>✅ 5 gráficos interactivos</p>
            <p>✅ Generación de reportes PDF</p>
            <p>✅ Envío por correo electrónico</p>
            <p>✅ Funciona sin internet</p>
            <hr style="border-color:#243653;">
            <p style="color:#8EA1B8; font-size:13px;">
                📦 Tamaño: ~299 MB<br>
                🖥️ Requisitos: Windows 10 o superior<br>
                💾 Espacio libre: 300 MB
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.link_button(
            "⬇ DESCARGAR .EXE",
            URL_EXE,
            use_container_width=True
        )

    with col2:
        st.markdown("### 📱 Escanea para descargar")
        st.markdown("""
        <div class="card" style="text-align:center;">
            <p style="color:#8EA1B8; font-size:13px;">
                Apunta la cámara de tu celular al código QR
                para descargar directamente en tu dispositivo.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Generar el QR dinámicamente
        import qrcode
        from io import BytesIO

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=2,
        )
        qr.add_data(URL_EXE)
        qr.make(fit=True)

        img_qr = qr.make_image(
            fill_color="#0B1220",
            back_color="white"
        ).convert("RGB")

        buf = BytesIO()
        img_qr.save(buf, format="PNG")
        buf.seek(0)

        col_qr_a, col_qr_b, col_qr_c = st.columns([1, 2, 1])
        with col_qr_b:
            st.image(buf, use_container_width=True)

    st.markdown("---")

    # Instrucciones de instalación
    st.markdown("### 📖 ¿Cómo instalar el .exe?")
    st.markdown("""
    <div class="card">
        <p><b>1.</b> Descarga el archivo <code>MotorAgroindustrial.exe</code></p>
        <p><b>2.</b> Guárdalo en una carpeta de tu preferencia (ej: Escritorio)</p>
        <p><b>3.</b> Haz doble clic para ejecutarlo</p>
        <p><b>4.</b> La primera vez, Windows mostrará una advertencia azul:</p>
        <ul style="font-size:13px;">
            <li>Clic en <b>"Más información"</b></li>
            <li>Clic en <b>"Ejecutar de todas formas"</b></li>
        </ul>
        <p style="color:#F59E0B; font-size:13px; margin-top:12px;">
            ⚠️ La advertencia es normal porque el programa
            no tiene firma digital. No contiene virus.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; font-size:22px;'>🍫 🍪 🥜 🌾 🍯 🌰</div>",
        unsafe_allow_html=True
    )