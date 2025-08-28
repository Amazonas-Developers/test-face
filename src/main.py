import os
import cv2
from dotenv import load_dotenv
from face_encoder import FaceEncoder


load_dotenv()

api_key = os.getenv('url_amazona')
url_dvr_local = f'{api_key}/Streaming/channels/101/'




cap = cv2.VideoCapture(url_dvr_local)





if not cap.isOpened():
    print(f"Error: No se pudo conectar al stream del DVR en la URL: {url_nvr_orlando}")
    print("Verifica si la dirección IP, el puerto, el usuario y la contraseña son correctos.")
    print("También, la 'ruta_del_stream' (/cam/realmonitor?...) es muy dependiente del fabricante y podría ser incorrecta.")
    exit()



# Bucle principal para leer y mostrar los fotogramas del stream.
while True:
    # Lee un fotograma.
    ret, frame = cap.read()

    # Si la lectura del fotograma falla (ej. se pierde la conexión), sal del bucle.
    if not ret:
        print("El stream se ha detenido.")
        break

    # Muestra el fotograma en una ventana de OpenCV.
    cv2.imshow('Stream de DVR', frame)

    # Espera 1 milisegundo. Si se presiona la tecla 'q', sal del bucle.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera el objeto de captura y cierra todas las ventanas.
cap.release()
cv2.destroyAllWindows()
