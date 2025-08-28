import os
import face_recognition
import logging
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image

class FaceEncoder:

    def __init__(self, base_path: str = "data/Persona"):
        self.base_path = base_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Configura el logger para la clase"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def load_faces(self) -> Tuple[List, List]:
        """
        Carga todas las caras desde la estructura de directorios y genera los encodings
        """
        self.logger.info(f"Iniciando carga de caras desde: {self.base_path}")
        
        # Reiniciar las listas
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Verificar que el directorio base existe
        if not os.path.exists(self.base_path):
            self.logger.error(f"El directorio base no existe: {self.base_path}")
            return [], []
        
        # Recorrer cada carpeta de persona
        person_folders = [f for f in os.listdir(self.base_path) 
                         if os.path.isdir(os.path.join(self.base_path, f))]
        
        self.logger.info(f"Carpetas de personas encontradas: {person_folders}")
        
        if not person_folders:
            self.logger.error("No se encontraron carpetas de personas")
            return [], []
        
        for person_name in person_folders:
            person_dir = os.path.join(self.base_path, person_name)
            self.logger.info(f"Procesando persona: {person_name}")
            
            # Procesar cada imagen en la carpeta de la persona
            image_files = [f for f in os.listdir(person_dir) if self._is_image_file(f)]
            
            if not image_files:
                self.logger.warning(f"No se encontraron imágenes en {person_dir}")
                continue
                
            self.logger.info(f"Imágenes encontradas para {person_name}: {image_files}")
            
            image_count = 0
            for image_file in image_files:
                image_path = os.path.join(person_dir, image_file)
                encoding = self._process_image(image_path, person_name)
                
                if encoding is not None:
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(person_name)
                    image_count += 1
            
            self.logger.info(f"Procesadas {image_count} imágenes para {person_name}")
        
        self.logger.info(f"Carga completada. Total de encodings: {len(self.known_face_encodings)}")
        return self.known_face_encodings, self.known_face_names

    def _is_image_file(self, filename: str) -> bool:
        """Verifica si un archivo es una imagen basándose en su extensión"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        return any(filename.lower().endswith(ext) for ext in image_extensions)

    def _process_image(self, image_path: str, person_name: str) -> Optional[List]:
        """
        Procesa una imagen individual y extrae el encoding facial
        """
        try:
            self.logger.debug(f"Procesando imagen: {image_path}")
            
            # Cargar la imagen con PIL primero para manejar metadatos EXIF
            pil_image = Image.open(image_path)
            
            # Corregir la orientación EXIF si es necesario
            pil_image = self._fix_image_orientation(pil_image)
            
            # Convertir a RGB si es necesario
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Redimensionar la imagen si es muy grande (más de 2000px en cualquier dimensión)
            max_size = 2000
            if max(pil_image.size) > max_size:
                pil_image = self._resize_image(pil_image, max_size)
            
            # Convertir a array numpy
            image = np.array(pil_image)
            
            # Verificar las propiedades de la imagen
            self.logger.debug(f"Formato de imagen: {image.shape}, tipo: {image.dtype}")
            
            # Asegurarse de que la imagen es de 8 bits
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            # Asegurarse de que la imagen es contigua en memoria
            image = np.ascontiguousarray(image)
            
            # Detectar rostros
            face_locations = face_recognition.face_locations(image, model="hog")
            self.logger.debug(f"Rostros detectados: {len(face_locations)}")
            
            if not face_locations:
                self.logger.warning(f"No se encontraron rostros en {image_path}")
                # Intentar con el modelo CNN
                face_locations = face_recognition.face_locations(image, model="cnn")
                self.logger.debug(f"Rostros detectados con CNN: {len(face_locations)}")
                
                if not face_locations:
                    # Guardar imagen para diagnóstico
                    debug_path = os.path.join("debug", f"no_face_{os.path.basename(image_path)}")
                    os.makedirs("debug", exist_ok=True)
                    cv2.imwrite(debug_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    self.logger.warning(f"Imagen guardada para diagnóstico: {debug_path}")
                    return None
            
            # Obtener encodings faciales
            encodings = face_recognition.face_encodings(image, face_locations)
            
            if not encodings:
                self.logger.warning(f"No se pudieron extraer encodings de {image_path}")
                return None
            
            # Si hay múltiples rostros, usar el primero
            if len(encodings) > 1:
                self.logger.warning(f"Múltiples rostros encontrados en {image_path}. Usando el primero.")
            
            return encodings[0]
            
        except Exception as e:
            self.logger.error(f"Error al procesar {image_path}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _fix_image_orientation(self, image):
        """Corrige la orientación de la imagen basándose en metadatos EXIF"""
        try:
            # Verificar si la imagen tiene información EXIF de orientación
            if hasattr(image, '_getexif'):
                exif = image._getexif()
                if exif is not None:
                    orientation_key = 274  # clave EXIF para orientación
                    if orientation_key in exif:
                        orientation = exif[orientation_key]
                        
                        # Rotar la imagen según la orientación EXIF
                        if orientation == 3:
                            image = image.rotate(180, expand=True)
                        elif orientation == 6:
                            image = image.rotate(270, expand=True)
                        elif orientation == 8:
                            image = image.rotate(90, expand=True)
        except Exception as e:
            self.logger.warning(f"Error al corregir orientación EXIF: {e}")
        
        return image

    def _resize_image(self, image, max_size):
        """Redimensiona la imagen manteniendo la relación de aspecto"""
        width, height = image.size
        
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        return image.resize((new_width, new_height), Image.LANCZOS)

    def convert_images_to_rgb(self):
        """
        Convierte todas las imágenes en el directorio base a formato RGB compatible.
        """
        self.logger.info("Convirtiendo imágenes a formato RGB")
        
        if not os.path.exists(self.base_path):
            self.logger.error(f"El directorio base no existe: {self.base_path}")
            return

        for person_name in os.listdir(self.base_path):
            person_dir = os.path.join(self.base_path, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            for image_file in os.listdir(person_dir):
                if not self._is_image_file(image_file):
                    continue
                    
                image_path = os.path.join(person_dir, image_file)
                try:
                    img = Image.open(image_path)
                    # Convertir a RGB si es necesario
                    if img.mode != 'RGB':
                        self.logger.info(f"Convirtiendo {image_path} de {img.mode} a RGB")
                        rgb_img = img.convert("RGB")
                        rgb_img.save(image_path)
                except Exception as e:
                    self.logger.error(f"Error al convertir {image_path}: {str(e)}")

    def get_encodings(self) -> List:
        """Devuelve los encodings faciales cargados"""
        return self.known_face_encodings
    
    def get_names(self) -> List:
        """Devuelve los nombres asociados a los encodings"""
        return self.known_face_names
    
    def get_encodings_dict(self) -> Dict[str, List]:
        """
        Devuelve un diccionario con los encodings organizados por persona
        """
        encodings_dict = {}
        
        for name, encoding in zip(self.known_face_names, self.known_face_encodings):
            if name not in encodings_dict:
                encodings_dict[name] = []
            encodings_dict[name].append(encoding)
        
        return encodings_dict

    def verify_image(self, image_path: str) -> bool:
        """
        Verifica si una imagen contiene rostros detectables
        """
        try:
            # Cargar la imagen con PIL
            pil_image = Image.open(image_path)
            pil_image = pil_image.convert('RGB')
            
            # Redimensionar si es necesario
            if max(pil_image.size) > 2000:
                pil_image = self._resize_image(pil_image, 2000)
            
            image = np.array(pil_image)
            image = np.ascontiguousarray(image)
            
            face_locations = face_recognition.face_locations(image, model="hog")
            
            if face_locations:
                self.logger.info(f"Imagen {image_path} contiene {len(face_locations)} rostros")
                return True
            else:
                # Intentar con CNN
                face_locations = face_recognition.face_locations(image, model="cnn")
                if face_locations:
                    self.logger.info(f"Imagen {image_path} contiene {len(face_locations)} rostros (detectados con CNN)")
                    return True
                else:
                    self.logger.warning(f"Imagen {image_path} no contiene rostros detectables")
                    return False
        except Exception as e:
            self.logger.error(f"Error al verificar imagen {image_path}: {str(e)}")
            return False