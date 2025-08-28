import os
import face_recognition
import logging
from typing import Dict, List, Tuple, Optional



class FaceEncoder:

    def __init__(self, base_path: str = "data/Persona"):
        self.base_path = base_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.logger = self._setup_logger()
    


    def _setup_logger(self) -> logging.Logger:
        """Configura el logger para la clase"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    


    def load_faces(self) -> Tuple[List, List]:
        """
        Carga todas las caras desde la estructura de directorios y genera los encodings
        
        Returns:
            Tuple con dos listas: (encodings, nombres)
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
        for person_name in os.listdir(self.base_path):
            person_dir = os.path.join(self.base_path, person_name)
            
            # Verificar que sea un directorio
            if not os.path.isdir(person_dir):
                continue
            
            self.logger.info(f"Procesando persona: {person_name}")
            
            # Procesar cada imagen en la carpeta de la persona
            image_count = 0
            for image_file in os.listdir(person_dir):
                if not self._is_image_file(image_file):
                    continue
                    
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
        
        Args:
            image_path: Ruta a la imagen
            person_name: Nombre de la persona para logging
            
        Returns:
            Encoding facial o None si no se pudo procesar
        """
        try:
            # Cargar la imagen
            image = face_recognition.load_image_file(image_path)
            
            # Obtener encodings faciales
            encodings = face_recognition.face_encodings(image)
            
            if not encodings:
                self.logger.warning(f"No se encontraron rostros en {image_path}")
                return None
            
            # Si hay múltiples rostros, usar el primero
            if len(encodings) > 1:
                self.logger.warning(f"Múltiples rostros encontrados en {image_path}. Usando el primero.")
            
            return encodings[0]
            
        except Exception as e:
            self.logger.error(f"Error al procesar {image_path}: {str(e)}")
            return None
    
    def get_encodings(self) -> List:
        """Devuelve los encodings faciales cargados"""
        return self.known_face_encodings
    
    def get_names(self) -> List:
        """Devuelve los nombres asociados a los encodings"""
        return self.known_face_names
    
    def get_encodings_dict(self) -> Dict[str, List]:
        """
        Devuelve un diccionario con los encodings organizados por persona
        
        Returns:
            Diccionario donde las claves son nombres y los valores son listas de encodings
        """
        encodings_dict = {}
        
        for name, encoding in zip(self.known_face_names, self.known_face_encodings):
            if name not in encodings_dict:
                encodings_dict[name] = []
            encodings_dict[name].append(encoding)
        
        return encodings_dict