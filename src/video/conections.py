import os
import cv2
from dotenv import load_dotenv


load_dotenv()

url_api = os.getenv('url_amazona')
list_conexion_rtsp = []


url = f"{url_api}/Streaming/Channels/101"
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo leer el frame")
        break

    # Mostrar el frame en una ventana
    cv2.imshow("Vista de cámara", frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




def camara_conection(url: str) -> dict:
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        return {
            "url": url,
            "cap": cap,
            "error": None
        }
    else:
        return {
            "url": url,
            "cap": None,
            "error": f"Error conextion {url}"
        }




for cam in range(1, 20):
    
    conextion_rtsp = f"{url_api}/Streaming/channels/{cam}01"
    print(conextion_rtsp)

    result = camara_conection (conextion_rtsp)
    list_conexion_rtsp.append(result)


   