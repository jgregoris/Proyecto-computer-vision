import argparse
import yaml
from pathlib import Path
import logging
import subprocess
import torch
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def verify_dependencies():
    """Verifica que todas las dependencias necesarias estén instaladas"""
    try:
        import torch
        import yaml
        logging.info(f"PyTorch version: {torch.__version__}")
    except ImportError as e:
        logging.error(f"Missing dependency: {e}")
        logging.error("Please install required packages: pip install torch torchvision PyYAML")
        sys.exit(1)

def verify_data_yaml(data_yaml_path: Path):
    """Verifica que el archivo data.yaml existe y tiene el formato correcto"""
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")
    
    with open(data_yaml_path) as f:
        try:
            data = yaml.safe_load(f)
            required_keys = ['train', 'val', 'test', 'nc', 'names']
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                raise ValueError(f"Missing required keys in data.yaml: {missing_keys}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing data.yaml: {e}")

def train_yolo(
    data_yaml: str,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    weights: str = 'yolov5s.pt'
):
    """Entrena el modelo YOLOv5"""
    try:
        # Obtener la ruta al directorio yolov5
        project_root = Path(__file__).parent.parent
        yolov5_dir = project_root / 'yolov5'
        
        # Verificar que existe el directorio yolov5
        if not yolov5_dir.exists():
            raise FileNotFoundError(
                f"YOLOv5 directory not found at: {yolov5_dir}\n"
                "Please clone YOLOv5 repository in the project root:\n"
                "cd logo-detection-project\n"
                "git clone https://github.com/ultralytics/yolov5.git"
            )
        
        # Crear directorio para los modelos si no existe
        models_dir = project_root / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Usar el mismo intérprete de Python que está ejecutando este script
        python_executable = sys.executable
        
        cmd = [
            python_executable,
            str(yolov5_dir / 'train.py'),
            '--img', str(img_size),
            '--batch', str(batch_size),
            '--epochs', str(epochs),
            '--data', data_yaml,
            '--weights', weights,
            '--project', str(models_dir),
            '--name', 'logo_detection'
        ]
        
        logging.info(f"Starting training with command: {' '.join(cmd)}")
        
        # Usar el mismo entorno de Python actual
        env = os.environ.copy()
        env['PYTHONPATH'] = str(yolov5_dir)
        
        subprocess.run(cmd, check=True, env=env)
        logging.info("Training completed successfully")
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Training failed with error: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during training: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model for logo detection')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    args = parser.parse_args()
    
    # Verificar dependencias
    verify_dependencies()
    
    # Obtener ruta del proyecto
    project_path = Path(__file__).parent.parent
    data_yaml = project_path / 'data' / 'data.yaml'
    
    # Verificar archivo data.yaml
    verify_data_yaml(data_yaml)
    
    # Verificar GPU
    if torch.cuda.is_available():
        logging.info(f"Training will use GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("No GPU detected, training will use CPU")
    
    # Entrenar modelo
    try:
        train_yolo(
            data_yaml=str(data_yaml),
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size
        )
    except Exception as e:
        logging.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()