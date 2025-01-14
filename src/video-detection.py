import cv2
import torch
from pathlib import Path
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class LogoDetector:
    def __init__(self, weights_path, device='cpu'):
        """
        Inicializa el detector de logos
        
        Args:
            weights_path: Ruta al archivo de pesos del modelo entrenado
            device: Dispositivo para inferencia ('cpu' o 'cuda')
        """
        self.device = device
        # Cargar modelo YOLOv5
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        self.model.to(device)
        logging.info(f"Modelo cargado desde {weights_path}")
        
    def process_video(self, video_path, output_path, conf_thres=0.25):
        """
        Procesa un video y guarda el resultado con las detecciones
        
        Args:
            video_path: Ruta al video de entrada
            output_path: Ruta donde guardar el video procesado
            conf_thres: Umbral de confianza para las detecciones
        """
        try:
            # Abrir video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"No se pudo abrir el video: {video_path}")
            
            # Obtener información del video
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Configurar escritor de video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
            
            frame_count = 0
            detections_count = 0
            
            logging.info(f"Iniciando procesamiento de video: {video_path}")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % 30 == 0:  # Log cada 30 frames
                    logging.info(f"Procesando frame {frame_count}/{total_frames}")
                
                # Realizar detección
                results = self.model(frame)
                detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
                
                # Dibujar detecciones
                for det in detections:
                    if det[4] >= conf_thres:  # Filtrar por confianza
                        detections_count += 1
                        x1, y1, x2, y2, conf, cls = det
                        
                        # Dibujar bbox
                        cv2.rectangle(frame, 
                                    (int(x1), int(y1)), 
                                    (int(x2), int(y2)), 
                                    (0, 255, 0), 2)
                        
                        # Añadir etiqueta con confianza
                        label = f"adidas {conf:.2f}"
                        y1 = max(y1, 30)  # Asegurar que la etiqueta es visible
                        cv2.putText(frame, label, 
                                  (int(x1), int(y1) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                  (0, 255, 0), 2)
                
                # Guardar frame
                out.write(frame)
            
            # Liberar recursos
            cap.release()
            out.release()
            
            logging.info(f"Procesamiento completado: {detections_count} detecciones en {frame_count} frames")
            return {
                'total_frames': frame_count,
                'total_detections': detections_count,
                'fps': fps,
                'duration_seconds': frame_count / fps
            }
            
        except Exception as e:
            logging.error(f"Error procesando video: {e}")
            raise
        finally:
            cv2.destroyAllWindows()

def main():
    # Configurar rutas
    project_path = Path(__file__).parent.parent
    weights_path = project_path / 'models' / 'logo_detection' / 'weights' / 'best.pt'
    
    # Verificar disponibilidad de GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Inicializar detector
        detector = LogoDetector(weights_path, device)
        
        # Procesar video
        video_path = project_path / 'data' / 'test_video.mp4'  # Ajusta según tu estructura
        output_path = project_path / 'results' / 'processed_video.mp4'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Procesar video y obtener estadísticas
        stats = detector.process_video(video_path, output_path)
        
        # Mostrar resultados
        logging.info("Estadísticas del procesamiento:")
        logging.info(f"Frames totales: {stats['total_frames']}")
        logging.info(f"Detecciones totales: {stats['total_detections']}")
        logging.info(f"Duración: {stats['duration_seconds']:.2f} segundos")
        logging.info(f"FPS: {stats['fps']}")
        
    except Exception as e:
        logging.error(f"Error en la ejecución: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()