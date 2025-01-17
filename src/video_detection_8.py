import cv2
import torch
from pathlib import Path
import warnings
import time

warnings.filterwarnings('ignore', category=FutureWarning)

def detect_logo_in_video(
    video_path, 
    weights_path, 
    conf_threshold=0.5,
    iou_threshold=0.45
):
    # Resolviendo rutas absolutas
    video_path = Path(video_path).resolve()
    weights_path = Path(weights_path).resolve()

    # Verificación de archivos
    if not video_path.exists():
        raise FileNotFoundError(f"El archivo de video no existe: {video_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"El archivo de pesos no existe: {weights_path}")

    # Inicializar variables para estadísticas
    start_time = time.time()
    total_frames = 0
    frames_with_logo = 0
    total_detections = 0
    confidence_sum = 0
    time_with_logo = 0
    frame_duration = 0

    device = torch.device('cpu')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(weights_path), device=device)
    model.conf = conf_threshold
    model.iou = iou_threshold

    # Abrir el video de entrada
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"No se pudo abrir el video en: {video_path}")

    # Obtener propiedades del video
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_duration = 1 / fps if fps > 0 else 0

    # Crear directorio de resultados si no existe
    output_dir = Path("results").resolve()
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Archivo de salida
    output_path = output_dir / f"detected_{video_path.stem}.mp4"

    # Configurar el escritor de video usando mp4v
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frame_width, frame_height)
    )

    try:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            total_frames += 1
            frame_has_logo = False
            
            # Realizar la detección
            results = model(frame)
            detections_in_frame = len(results.xyxy[0])
            
            if detections_in_frame > 0:
                frames_with_logo += 1
                frame_has_logo = True
                total_detections += detections_in_frame
                
                # Sumar las confidencias
                for det in results.xyxy[0]:
                    confidence_sum += det[4].item()
            
            # Dibujar las detecciones
            for det in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = det.tolist()
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{conf:.2f}', (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if frame_has_logo:
                time_with_logo += frame_duration
                
            # Escribir el frame procesado
            out.write(frame)

    finally:
        # Liberar recursos
        video.release()
        out.release()

    # Calcular estadísticas finales
    total_time = time.time() - start_time
    average_confidence = (confidence_sum / total_detections) if total_detections > 0 else 0
    percentage = (time_with_logo / (total_frames / fps)) * 100 if total_frames > 0 else 0

    return {
        'output_path': str(output_path),
        'total_time': total_time,
        'logo_time': time_with_logo,
        'percentage': percentage,
        'total_frames': total_frames,
        'frames_with_logo': frames_with_logo,
        'total_detections': total_detections,
        'average_confidence': average_confidence
    }
