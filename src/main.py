import cv2

url_dvr_local = 'rtsp://test:Test123456.@72.68.60.117:554/Streaming/channels/101/'

url_nvr_orlando = 'rtsp://admin:sistel2020@108.191.80.47:554/Streaming/channels/401/'


cap = cv2.VideoCapture(url_dvr_local)


# Verifica que el stream se haya abierto correctamente.
if not cap.isOpened():
    print(f"Error: No se pudo conectar al stream del DVR en la URL: {url_nvr_orlando}")
    print("Verifica si la dirección IP, el puerto, el usuario y la contraseña son correctos.")
    print("También, la 'ruta_del_stream' (/cam/realmonitor?...) es muy dependiente del fabricante y podría ser incorrecta.")
    exit()



# Bucle principal para leer y mostrar los fotogramas del video.
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
