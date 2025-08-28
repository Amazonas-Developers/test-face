import os
import cv2
from dotenv import load_dotenv
from face_encoder import FaceEncoder
import face_recognition
import numpy as np


load_dotenv()

api_key = os.getenv('url_amazona')
url_dvr_local = f'{api_key}/Streaming/channels/202/'




cap = cv2.VideoCapture(url_dvr_local)




if not cap.isOpened():
    print(f"Error: No se pudo conectar al stream del DVR en la URL: {url_dvr_local}")
    print("Verifica si la dirección IP, el puerto, el usuario y la contraseña son correctos.")
    print("También, la 'ruta_del_stream' (/cam/realmonitor?...) es muy dependiente del fabricante y podría ser incorrecta.")
    exit()



# Bucle principal para leer y mostrar los fotogramas del stream.
while True:
    ret, frame = cap.read()
    if not ret:
        print("El stream se ha detenido.")
        break

    cv2.imshow('Stream de DVR', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Libera el objeto de captura y cierra todas las ventanas.
cap.release()
cv2.destroyAllWindows()






