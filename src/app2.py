from pathlib import Path
import streamlit as st
from video_detection_4 import detect_logo_in_video

# Ruta base del proyecto
PROJECT_DIR = Path(__file__).resolve().parents[1]  # Subir dos niveles desde src
WEIGHTS_PATH = PROJECT_DIR / "models" / "logo_detection" / "weights" / "best.pt"

# Mostrar información de depuración
st.write(f"Directorio base del proyecto: {PROJECT_DIR}")
st.write(f"Ruta completa al archivo de pesos: {WEIGHTS_PATH}")

# Verificar la existencia del archivo de pesos
if not WEIGHTS_PATH.exists():
    st.error(f"El archivo de pesos no existe en la ruta: {WEIGHTS_PATH}")
    st.stop()

# Interfaz de usuario
st.title("Detección de Logos en Videos")
video_file = st.file_uploader("Sube un video", type=["mp4", "avi", "mov"])

if video_file is not None:
    # Crear carpeta para videos subidos
    uploaded_videos_dir = PROJECT_DIR / "uploaded_videos"
    uploaded_videos_dir.mkdir(exist_ok=True)

    # Guardar video temporalmente
    video_path = uploaded_videos_dir / video_file.name
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    st.write("Procesando video...")

    try:
        results = detect_logo_in_video(video_path, WEIGHTS_PATH)
        st.success("Procesamiento completado.")
        st.video(str(results['output_path']))
    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
