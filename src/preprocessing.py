import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path
import logging
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

class LogoPreprocessor:
    def __init__(self, dataset_path, project_path, target_logo="adidas"):
        """
        Inicializa el preprocesador de logos.
        
        Args:
            dataset_path (str): Ruta al dataset OpenLogo
            project_path (str): Ruta al directorio del proyecto
            target_logo (str): Nombre del logo a detectar
        """
        self.dataset_path = Path(dataset_path).resolve()
        self.project_path = Path(project_path).resolve()
        self.target_logo = target_logo.lower()
        
        # Verificar que las rutas existen
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        if not self.project_path.exists():
            raise FileNotFoundError(f"Project path does not exist: {self.project_path}")
            
        # Verificar estructura del dataset
        self.verify_dataset_structure()
        
        # Configurar rutas del proyecto
        self.processed_data_path = self.project_path / "data" / "processed"
        
        # Crear estructura de directorios antes de procesar
        self.create_base_directories()
        
        logging.info(f"Inicializado preprocesador para logo: {self.target_logo}")
        logging.info(f"Dataset path: {self.dataset_path}")
        logging.info(f"Project path: {self.project_path}")
        logging.info(f"Processed data path: {self.processed_data_path}")

    def verify_dataset_structure(self):
        """Verifica que la estructura del dataset sea correcta."""
        required_dirs = ['JPEGImages', 'Annotations']
        for dir_name in required_dirs:
            dir_path = self.dataset_path / dir_name
            if not dir_path.exists():
                raise FileNotFoundError(
                    f"Required directory '{dir_name}' not found in dataset path: {dir_path}"
                )

    def create_base_directories(self):
        """Crea la estructura base de directorios necesaria."""
        # Crear directorio principal de datos procesados
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Crear subdirectorios para cada split
        for split in ['train', 'val', 'test']:
            split_dir = self.processed_data_path / split
            (split_dir / 'images').mkdir(parents=True, exist_ok=True)
            (split_dir / 'labels').mkdir(parents=True, exist_ok=True)
        
        logging.info("Creada estructura de directorios")

    def create_directory_structure(self):
        """Distribuye las imágenes en los directorios train/val/test."""
        splits = {
            'train': 0.7,
            'val': 0.15,
            'test': 0.15
        }
        
        # Encontrar todas las imágenes con el logo
        all_images = self.find_logo_images()
        
        if not all_images:
            logging.warning(f"No se encontraron imágenes para el logo {self.target_logo}")
            return
        
        # Mezclar aleatoriamente las imágenes
        np.random.shuffle(all_images)
        
        # Dividir el dataset
        total_images = len(all_images)
        train_size = int(total_images * splits['train'])
        val_size = int(total_images * splits['val'])
        
        train_images = all_images[:train_size]
        val_images = all_images[train_size:train_size + val_size]
        test_images = all_images[train_size + val_size:]
        
        # Copiar archivos para cada split
        splits_data = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }
        
        for split_name, images in splits_data.items():
            split_dir = self.processed_data_path / split_name
            self.copy_dataset_files(images, split_dir)
            logging.info(f"Procesadas {len(images)} imágenes para {split_name}")

    def find_logo_images(self):
        """Encuentra todas las imágenes que contienen el logo objetivo."""
        logo_images = []
        annotations_path = self.dataset_path / "Annotations"
        
        for xml_file in annotations_path.glob("*.xml"):
            boxes, labels, _ = self.process_annotation(xml_file)
            if len(labels) > 0:
                image_name = xml_file.stem + ".jpg"
                # Verificar que la imagen existe
                if (self.dataset_path / "JPEGImages" / image_name).exists():
                    logo_images.append(image_name)
                else:
                    logging.warning(f"Imagen no encontrada: {image_name}")
        
        logging.info(f"Encontradas {len(logo_images)} imágenes con el logo {self.target_logo}")
        return logo_images

    def process_annotation(self, xml_path):
        """Procesa un archivo de anotación XML."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        for obj in root.findall('object'):
            name = obj.find('name').text.lower()
            if name == self.target_logo:
                bbox = obj.find('bndbox')
                xmin = int(float(bbox.find('xmin').text))
                ymin = int(float(bbox.find('ymin').text))
                xmax = int(float(bbox.find('xmax').text))
                ymax = int(float(bbox.find('ymax').text))
                
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(name)
                
        return np.array(boxes), labels, (width, height)

    def copy_dataset_files(self, image_names, split_dir):
        """
        Copia las imágenes y genera las etiquetas en formato YOLO.
        
        Args:
            image_names (list): Lista de nombres de imágenes
            split_dir (Path): Directorio de destino para el split
        """
        for image_name in image_names:
            try:
                # Copiar imagen
                src_img = self.dataset_path / 'JPEGImages' / image_name
                dst_img = split_dir / 'images' / image_name
                
                if not src_img.exists():
                    logging.warning(f"Imagen fuente no encontrada: {src_img}")
                    continue
                    
                shutil.copy2(src_img, dst_img)
                
                # Procesar y guardar etiquetas en formato YOLO
                xml_name = Path(image_name).stem + '.xml'
                xml_path = self.dataset_path / 'Annotations' / xml_name
                boxes, labels, (width, height) = self.process_annotation(xml_path)
                
                # Convertir a formato YOLO y guardar
                label_path = split_dir / 'labels' / (Path(image_name).stem + '.txt')
                with open(label_path, 'w') as f:
                    for box in boxes:
                        # Convertir a formato YOLO (normalizado)
                        x_center = ((box[0] + box[2]) / 2) / width
                        y_center = ((box[1] + box[3]) / 2) / height
                        w = (box[2] - box[0]) / width
                        h = (box[3] - box[1]) / height
                        
                        # Clase 0 para el logo objetivo
                        f.write(f"0 {x_center} {y_center} {w} {h}\n")
                        
            except Exception as e:
                logging.error(f"Error procesando {image_name}: {e}")

def main():
    # Obtener el directorio actual donde se ejecuta el script
    current_dir = Path(__file__).parent.resolve()
    
    # Configurar rutas relativas al directorio del proyecto
    project_path = current_dir.parent  # Sube un nivel desde src/
    dataset_path = project_path.parent / "datasets" / "OpenLogo-Dataset"
    
    try:
        # Crear preprocesador
        preprocessor = LogoPreprocessor(
            dataset_path=str(dataset_path),
            project_path=str(project_path),
            target_logo="adidas"
        )
        
        # Procesar el dataset
        preprocessor.create_directory_structure()
        logging.info("Procesamiento completado exitosamente")
        
    except Exception as e:
        logging.error(f"Error durante la ejecución: {e}")
        raise

if __name__ == "__main__":
    main()