import os
import cv2
from dotenv import load_dotenv
import face_recognition
import numpy as np
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Cargar variables de entorno
load_dotenv()
__dirname = os.path.dirname(__file__)

# Configuración de recursos
NUM_CORES = mp.cpu_count()  # Obtener número de núcleos disponibles
MAX_WORKERS = NUM_CORES * 2  # Regla general: 2 hilos por núcleo

# Cargar imagen de referencia
path_image = os.path.join(__dirname, '1.jpg')
reference_image = face_recognition.load_image_file(path_image)
referencia_face_encodings = face_recognition.face_encodings(reference_image)

if len(referencia_face_encodings) == 0:
    print("Error: No se encontraron rostros en la imagen de referencia.")
    exit()

referencia_encoding = referencia_face_encodings[0]

# Configuración de la fuente de video
api_key = os.getenv('url_amazona')
url_dvr_local = f'{api_key}/Streaming/channels/201/'

# Inicializar captura de video
cap = cv2.VideoCapture(url_dvr_local)
if not cap.isOpened():
    print(f"Error: No se pudo conectar al stream del DVR en la URL: {url_dvr_local}")
    exit()

# Configurar OpenCV para mejor rendimiento
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reducir buffer para menor latencia
cap.set(cv2.CAP_PROP_FPS, 30)  # Establecer FPS deseado

# Función para procesamiento de rostros en paralelo
def process_face(face_data):
    face_location, face_encoding = face_data
    top, right, bottom, left = face_location
    
    # Comparar con el rostro de referencia
    matches = face_recognition.compare_faces([referencia_encoding], face_encoding)
    face_distances = face_recognition.face_distance([referencia_encoding], face_encoding)
    
    name = "Desconocido"
    if len(face_distances) > 0 and matches[0] and face_distances[0] < 0.6:
        name = "Conocido"
    
    return (top, right, bottom, left, name)

# Configurar el executor para procesamiento paralelo
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Bucle principal optimizado
frame_count = 0
skip_frames = 2  # Procesar cada 3 frames para mayor rendimiento

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Saltar frames para mejorar el rendimiento
        if frame_count % (skip_frames + 1) != 0:
            continue
        
        # Reducir resolución para procesamiento más rápido
        scale_factor = 0.5
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detectar rostros (usar modelo HOG para mayor velocidad)
        face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        # Procesar rostros en paralelo
        face_data = zip(face_locations, face_encodings)
        results = list(executor.map(process_face, face_data))
        
        # Dibujar resultados en el frame original
        for top, right, bottom, left, name in results:
            # Escalar coordenadas de vuelta al tamaño original
            top = int(top / scale_factor)
            right = int(right / scale_factor)
            bottom = int(bottom / scale_factor)
            left = int(left / scale_factor)
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        # Mostrar frame
        cv2.imshow('Video con detección de rostros', frame)
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Liberar recursos
    executor.shutdown(wait=True)
    cap.release()
    cv2.destroyAllWindows()