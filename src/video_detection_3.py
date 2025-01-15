import cv2
import torch
import time
from pathlib import Path
import os
import warnings
import platform

# Ignorar las advertencias de FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)

def detect_logo_in_video(
    video_path, 
    weights_path, 
    conf_threshold=0.90,  # Aumentado para mayor precisión
    iou_threshold=0.45,  # Añadido umbral IOU para NMS
    min_area_ratio=0.003  # Área mínima relativa que debe ocupar el logo
):
    """
    Detect logos in a video using YOLOv5 model with high precision settings
    
    Args:
        video_path (str): Path to the input video
        weights_path (str): Path to the YOLOv5 weights
        conf_threshold (float): Confidence threshold for detections
        iou_threshold (float): IOU threshold for NMS
        min_area_ratio (float): Minimum area ratio the logo should occupy in the frame
    """
    # Verificar que el archivo de video existe
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"El archivo de video no existe en la ruta: {video_path}")
    
    # Configurar dispositivo
    device = torch.device('cpu')
    
    # Load YOLOv5 model with custom settings
    print(f"Cargando modelo desde: {weights_path}")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, device=device)
    model.conf = conf_threshold
    model.iou = iou_threshold  # Umbral IOU para Non-Maximum Suppression
    
    # Optimize model for inference
    model.eval()
    if hasattr(model, 'model'):
        model.model.float()
    
    # Open video file
    print(f"Intentando abrir video desde: {video_path}")
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise ValueError(f"No se pudo abrir el video en la ruta: {video_path}")
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError(f"FPS inválidos detectados: {fps}")
        
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_area = frame_width * frame_height
    min_logo_area = frame_area * min_area_ratio
    
    print(f"\nConfiguración de detección:")
    print(f"- Umbral de confianza: {conf_threshold}")
    print(f"- Umbral IOU: {iou_threshold}")
    print(f"- Área mínima del logo: {min_logo_area:.0f} pixels")
    
    print(f"\nPropiedades del video:")
    print(f"- FPS: {fps}")
    print(f"- Resolución: {frame_width}x{frame_height}")
    print(f"- Frames totales: {total_frames}")
    print(f"- Procesando todos los frames para máxima precisión")
    
    # Create output directory if it doesn't exist
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Configurar el códec según el sistema operativo
    if platform.system() == 'Darwin':
        output_path = str(output_dir / f"detected_{Path(video_path).stem}.mov")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    else:
        output_path = str(output_dir / f"detected_{Path(video_path).stem}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        raise ValueError("No se pudo crear el archivo de salida. Verifica que el códec esté disponible.")
    
    # Initialize counters
    frames_with_logo = 0
    frame_count = 0
    total_detections = 0
    confidence_sum = 0
    processed_frames = 0
    start_time = time.time()
    
    print("\nIniciando procesamiento del video...")
    
    try:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            frame_count += 1
            processed_frames += 1
            
            # Perform detection
            results = model(frame)
            
            frame_has_logo = False
            frame_detections = []
            
            # Process detections
            for det in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = det.tolist()
                
                # Calcular área del logo detectado
                logo_area = (x2 - x1) * (y2 - y1)
                
                # Verificar área mínima y confianza
                if conf >= conf_threshold and logo_area >= min_logo_area:
                    frame_has_logo = True
                    confidence_sum += conf
                    total_detections += 1
                    
                    # Draw bounding box
                    cv2.rectangle(frame, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                (0, 255, 0), 2)
                    
                    # Add label with confidence and area
                    area_percentage = (logo_area / frame_area) * 100
                    label = f'Adidas {conf:.2f} ({area_percentage:.1f}%)'
                    cv2.putText(frame, label, 
                              (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 2)
                    
                    frame_detections.append({
                        'confidence': conf,
                        'area': logo_area,
                        'area_percentage': area_percentage
                    })
            
            if frame_has_logo:
                frames_with_logo += 1
            
            # Write frame to output video
            out.write(frame)
            
            # Print progress every 5 seconds
            if time.time() - start_time > 5:
                progress = (frame_count / total_frames) * 100
                elapsed_time = time.time() - start_time
                fps_processing = processed_frames / elapsed_time
                print(f"Progreso: {progress:.1f}% - FPS: {fps_processing:.1f}")
                start_time = time.time()
                processed_frames = 0
        
        # Calculate statistics
        total_time = total_frames / fps
        logo_time = frames_with_logo / fps
        percentage = (frames_with_logo / total_frames) * 100
        avg_confidence = confidence_sum / total_detections if total_detections > 0 else 0
        
    finally:
        # Clean up
        video.release()
        out.release()
        cv2.destroyAllWindows()
    
    print("\n¡Procesamiento completado!")
    print(f"\nEstadísticas de detección:")
    print(f"- Duración total del video: {total_time:.2f} segundos")
    print(f"- Tiempo de aparición del logo: {logo_time:.2f} segundos")
    print(f"- Porcentaje de aparición del logo: {percentage:.2f}%")
    print(f"- Total de detecciones: {total_detections}")
    print(f"- Confianza promedio: {avg_confidence:.2f}")
    print(f"- Frames con logo detectado: {frames_with_logo}")
    print(f"\nVideo procesado guardado en: {output_path}")
    
    return {
        'total_time': total_time,
        'logo_time': logo_time,
        'percentage': percentage,
        'frames_with_logo': frames_with_logo,
        'total_frames': total_frames,
        'total_detections': total_detections,
        'average_confidence': avg_confidence
    }

if __name__ == "__main__":
    # Obtener la ruta absoluta del directorio del proyecto
    project_dir = Path(__file__).resolve().parents[1]
    
    # Configurar rutas
    VIDEO_PATH = str(project_dir / "data" / "videos" / "video_adidas.mp4")
    WEIGHTS_PATH = str(project_dir / "yolov5s.pt")
    
    print(f"Ruta del video: {VIDEO_PATH}")
    print(f"Ruta de los pesos: {WEIGHTS_PATH}")
    
    try:
        results = detect_logo_in_video(VIDEO_PATH, WEIGHTS_PATH)
    except KeyboardInterrupt:
        print("\nProceso interrumpido por el usuario")
    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")