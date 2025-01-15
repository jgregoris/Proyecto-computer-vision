import streamlit as st
import torch
import cv2
import numpy as np
from pathlib import Path
import tempfile
import time
from PIL import Image
from video_detection_3 import detect_logo_in_video  # Actualizado el nombre del archivo

# Definir la ruta base del proyecto
PROJECT_DIR = Path(__file__).resolve().parents[1]  # Subir dos niveles desde src
WEIGHTS_PATH = PROJECT_DIR / "models" / "logo_detection" / "weights" / "best.pt"

def load_model(weights_path):
    """Carga el modelo YOLOv5 con los pesos especificados."""
    device = torch.device('cpu')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(weights_path), device=device)
    model.conf = 0.65  # Umbral de confianza
    return model

def process_image(image, model):
    """Procesa una imagen y detecta el logo de Adidas."""
    results = model(image)
    
    # Convertir resultados a formato numpy para visualización
    img_with_boxes = results.render()[0]
    
    detections = []
    for det in results.xyxy[0]:  # xyxy format: x1, y1, x2, y2, confidence, class
        x1, y1, x2, y2, conf, cls = det.tolist()
        detections.append({
            'bbox': (int(x1), int(y1), int(x2), int(y2)),
            'confidence': conf
        })
    
    return img_with_boxes, detections

def main():
    st.set_page_config(page_title="Adidas Logo Detector", layout="wide")
    
    st.title("🎯 Adidas Logo Detector")
    
    # Cargar modelo
    try:
        
        model = load_model(WEIGHTS_PATH)
        st.success("✅ Modelo cargado correctamente")
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return

    # Crear tabs para imagen y video
    tab1, tab2 = st.tabs(["📷 Detección en Imagen", "🎥 Detección en Video"])
    
    # Tab de Imagen
    with tab1:
        st.header("Detección de Logo en Imagen")
        uploaded_image = st.file_uploader("Sube una imagen", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_image:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Imagen Original")
                image = Image.open(uploaded_image)
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("Imagen Procesada")
                # Procesar imagen
                try:
                    processed_image, detections = process_image(image, model)
                    st.image(processed_image, use_container_width=True)
                    
                    # Mostrar resultados
                    if detections:
                        st.success(f"✅ Se detectaron {len(detections)} logos de Adidas")
                        for i, det in enumerate(detections, 1):
                            st.info(f"Detección {i}: Confianza {det['confidence']:.2%}")
                    else:
                        st.warning("⚠️ No se detectaron logos de Adidas en la imagen")
                except Exception as e:
                    st.error(f"❌ Error al procesar la imagen: {str(e)}")
    
    # Tab de Video
    with tab2:
        st.header("Detección de Logo en Video")
        uploaded_video = st.file_uploader("Sube un video", type=['mp4', 'mov', 'avi'])
        
        if uploaded_video:
            # Guardar el video temporalmente
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name
            
            try:
                with st.spinner("Procesando video... Esto puede tomar varios minutos."):
                    # Procesar video
                    results = detect_logo_in_video(video_path, WEIGHTS_PATH)
                    
                    # Mostrar resultados en un formato amigable
                    st.success("✅ Video procesado correctamente")
                    
                    # Métricas principales
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Tiempo Total", f"{results['total_time']:.2f}s")
                    with col2:
                        st.metric("Tiempo con Logo", f"{results['logo_time']:.2f}s")
                    with col3:
                        st.metric("Porcentaje de Aparición", f"{results['percentage']:.2f}%")
                    
                    # Estadísticas detalladas
                    st.subheader("📊 Estadísticas Detalladas")
                    stats_col1, stats_col2 = st.columns(2)
                    
                    with stats_col1:
                        st.info(f"Total de frames: {results['total_frames']}")
                        st.info(f"Frames con logo: {results['frames_with_logo']}")
                    
                    with stats_col2:
                        st.info(f"Total de detecciones: {results['total_detections']}")
                        st.info(f"Confianza promedio: {results['average_confidence']:.2%}")
                    
                    # Mostrar video procesado
                    results_dir = PROJECT_DIR / "results"
                    video_file = results_dir / f"detected_{Path(video_path).stem}.mp4"
                    if video_file.exists():
                        st.subheader("🎬 Video Procesado")
                        st.video(str(video_file))
                
            except Exception as e:
                st.error(f"❌ Error al procesar el video: {str(e)}")
            finally:
                # Limpiar archivos temporales
                Path(video_path).unlink(missing_ok=True)

if __name__ == "__main__":
    main()
