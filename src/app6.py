import streamlit as st
import torch
import cv2
import numpy as np
from pathlib import Path
import tempfile
import time
import webbrowser
import os
import shutil
from PIL import Image
from video_detection_8 import detect_logo_in_video

# Definir la ruta base del proyecto
PROJECT_DIR = Path(__file__).resolve().parents[1]  # Subir dos niveles desde src
WEIGHTS_PATH = PROJECT_DIR / "models" / "logo_detection" / "weights" / "best.pt"
DOWNLOADS_DIR = Path("downloads").resolve()

def load_model(weights_path):
    """Carga el modelo YOLOv5 con los pesos especificados."""
    device = torch.device('cpu')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(weights_path), device=device)
    model.conf = 0.5  # Umbral de confianza
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

def save_video_for_download(video_path, original_filename):
    """Guarda el video procesado en el directorio de descargas."""
    # Crear directorio de descargas si no existe
    DOWNLOADS_DIR.mkdir(exist_ok=True, parents=True)
    
    # Generar nombre único para el archivo
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_filename = f"processed_{timestamp}_{original_filename}"
    output_path = DOWNLOADS_DIR / output_filename
    
    # Copiar el video al directorio de descargas
    shutil.copy2(video_path, output_path)
    
    return output_path

def open_file_explorer(path):
    """Abre el explorador de archivos en la ubicación del archivo."""
    if os.name == 'nt':  # Windows
        os.startfile(os.path.dirname(path))
    elif os.name == 'posix':  # Linux/Mac
        webbrowser.open(f'file://{os.path.dirname(path)}')

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
        uploaded_image = st.file_uploader("Sube una imagen", type=['jpg', 'jpeg', 'png'], key="image_uploader")
        
        if uploaded_image:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Imagen Original")
                image = Image.open(uploaded_image)
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("Imagen Procesada")
                try:
                    processed_image, detections = process_image(image, model)
                    st.image(processed_image, use_container_width=True)
                    
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
        uploaded_video = st.file_uploader("Sube un video", type=['mp4', 'mov', 'avi'], key="video_uploader")
        
        if uploaded_video:
            # Crear un archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.getvalue())
                temp_video_path = Path(tmp_file.name)
            
            try:
                with st.spinner("Procesando video... Esto puede tomar varios minutos."):
                    results = detect_logo_in_video(str(temp_video_path), WEIGHTS_PATH, conf_threshold=0.5)
                    
                    if results and 'error' not in results:
                        st.success("✅ Video procesado correctamente")
                        
                        # Métricas principales
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Tiempo Total", f"{results.get('total_time', 0):.2f}s")
                        with col2:
                            st.metric("Tiempo con Logo", f"{results.get('logo_time', 0):.2f}s")
                        with col3:
                            st.metric("Porcentaje de Aparición", f"{results.get('percentage', 0):.2f}%")
                        
                        # Estadísticas detalladas
                        st.subheader("📊 Estadísticas Detalladas")
                        stats_col1, stats_col2 = st.columns(2)
                        
                        with stats_col1:
                            st.info(f"Total de frames: {results.get('total_frames', 0)}")
                            st.info(f"Frames con logo: {results.get('frames_with_logo', 0)}")
                        
                        with stats_col2:
                            st.info(f"Total de detecciones: {results.get('total_detections', 0)}")
                            st.info(f"Confianza promedio: {results.get('average_confidence', 0):.2%}")
                        
                        # Guardar y mostrar el video procesado
                        if 'output_path' in results and Path(results['output_path']).exists():
                            st.subheader("🎬 Video Procesado")
                            
                            # Guardar el video en el directorio de descargas
                            saved_video_path = save_video_for_download(
                                results['output_path'], 
                                uploaded_video.name
                            )
                            
                            
                            # Mostrar la ruta donde se guardó el video
                            st.info(f"📍 Video guardado en: {saved_video_path}")
                            
                        else:
                            st.warning("⚠️ El video procesado no está disponible")
                    else:
                        st.error("❌ Error en el procesamiento del video")
                
            except Exception as e:
                st.error(f"❌ Error al procesar el video: {str(e)}")
            finally:
                # Limpiar archivo temporal de entrada
                if temp_video_path.exists():
                    temp_video_path.unlink()

if __name__ == "__main__":
    main()