import cv2
import torch
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

def detect_logo_in_video(
    video_path, 
    weights_path, 
    conf_threshold=0.90,
    iou_threshold=0.45
):
    # Resolviendo rutas absolutas
    video_path = Path(video_path).resolve()
    weights_path = Path(weights_path).resolve()

    # Depuración de rutas
    print(f"Ruta del video: {video_path}")
    print(f"Ruta del modelo: {weights_path}")

    if not video_path.exists():
        raise FileNotFoundError(f"El archivo de video no existe: {video_path}")

    if not weights_path.exists():
        raise FileNotFoundError(f"El archivo de pesos no existe: {weights_path}")

    device = torch.device('cpu')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(weights_path), device=device)
    model.conf = conf_threshold
    model.iou = iou_threshold

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"No se pudo abrir el video en: {video_path}")

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    output_dir = Path("results").resolve()
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"detected_{video_path.stem}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        results = model(frame)
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det.tolist()
            if conf >= conf_threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        out.write(frame)

    video.release()
    out.release()

    return {'output_path': output_path}
