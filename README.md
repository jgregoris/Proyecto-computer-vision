# 🎯 Proyecto de Detección de Logo Adidas

Un proyecto de visión por computador que detecta y rastrea logos de Adidas en imágenes y videos utilizando YOLOv5.

## 📝 Descripción

Este proyecto implementa un sistema de detección de logos basado en aprendizaje profundo, específicamente entrenado para identificar logos de Adidas tanto en imágenes estáticas como en contenido de video. El sistema utiliza la arquitectura YOLOv5 y proporciona análisis detallados sobre las apariciones del logo.

## 🚀 Características

- 📷 Detección de logos en tiempo real en imágenes
- 🎥 Procesamiento de videos con análisis detallado
- 📊 Estadísticas y métricas de detección
- 💻 Interfaz de usuario amigable con Streamlit
- 🔄 Capacidades de procesamiento por lotes
- 📈 Análisis de rendimiento y visualización

## 🏗️ Estructura del Proyecto

```
logo-detection-project/
├── datasets/
│   └── OpenLogo-Dataset/
│       ├── Annotations/
│       ├── ImageSets/
│       └── JPEGImages/
├── data/
├── models/
│   └── logo_detection/
│       └── weights/
├── results/
├── src/
│   ├── app.py               # Interfaz web Streamlit
│   ├── preprocessing.py      # Utilidades de preprocesamiento
│   ├── train.py             # Script de entrenamiento
│   └── video_detection.py  # Módulo de procesamiento de video
├── uploaded_videos/
└── yolov5/                   # Submódulo YOLOv5
```

## 🔧 Requisitos

- Python 3.12.4+
- PyTorch 2.5.1+
- OpenCV
- Streamlit
- YOLOv5
- CUDA (opcional, para aceleración GPU)

## 📦 Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tuusuario/logo-detection-project.git
cd logo-detection-project
```

2. Crear y activar el entorno virtual:
```bash
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
myenv\Scripts\activate     # Windows
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Clonar el repositorio YOLOv5:
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

## 🎮 Uso

### Entrenando el Modelo

```bash
python src/train.py --epochs 100 --batch-size 16 --img-size 640
```

### Ejecutando la Interfaz Web

```bash
streamlit run src/app.py
```

### Procesando un Video

```python
from video_detection import detect_logo_in_video

resultados = detect_logo_in_video(
    video_path="ruta/al/video.mp4",
    weights_path="models/logo_detection/weights/best.pt",
    conf_threshold=0.5
)
```

## 📊 Resultados

El sistema proporciona métricas completas de detección incluyendo:
- Tiempo total de detección
- Duración de aparición del logo
- Porcentaje de presencia del logo
- Análisis cuadro por cuadro
- Puntuaciones de confianza
- Conteo de detecciones

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! No dudes en enviar un Pull Request.

## 🙏 Agradecimientos

- [YOLOv5](https://github.com/ultralytics/yolov5) por la arquitectura de detección de objetos
- OpenLogo Dataset por los datos de entrenamiento
- Streamlit por el framework de interfaz web

## 📧 Contacto

Para cualquier pregunta o sugerencia, por favor abre un issue en el repositorio o contacta a los mantenedores.

## Autores
Javier Gregoris y Alejandra Piñango.
