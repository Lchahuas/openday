import cv2
import numpy as np
from ultralytics import YOLO

# Cargar el modelo YOLO
model = YOLO("yolov8n-seg.pt")

def detect_qr_and_color(frame):
    qr_detector = cv2.QRCodeDetector()
    data, points, _ = qr_detector.detectAndDecode(frame)
    
    if data:
        if data == "RED":
            color = (0, 0, 255)  
        elif data == "GREEN":
            color = (0, 255, 0)  
        elif data == "BLUE":
            color = (255, 0, 0)  
        else:
            color = None
    else:
        color = None
    
    return data, color

def process_frame(frame, qr_color=None):
    results = model.predict(source=frame)
    
    # Detectar QR y el color
    qr_data, new_qr_color = detect_qr_and_color(frame)
    
    if qr_data:
        print(f"QR Detected: {qr_data} with color {new_qr_color}")
        qr_color = new_qr_color
    else:
        qr_color = None  
    
    output_frame = frame.copy()
    
    if qr_color is not None:
        # Convertir a escala de grises
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        output_frame = gray_frame
    
    for result in results:
        for c in result:
            label = c.names[int(c.boxes.cls.tolist().pop())]
            bbox = c.boxes.xyxy[0].cpu().numpy().astype(int)  # Convertir tensor a array NumPy
            x1, y1, x2, y2 = bbox

            # Dibujar el bounding box en morado
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (128, 0, 128), 2)

            # Añadir el texto sobre la persona detectada
            if label == 'person':
                cv2.putText(output_frame, "Futuro Cachimbo UTEC", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2, cv2.LINE_AA)
                
                if qr_color is not None:
                    b_mask = np.zeros(frame.shape[:2], np.uint8)
                    contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                    cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

                    color_mask = np.zeros_like(frame)
                    color_mask[b_mask == 255] = qr_color

                    output_frame = cv2.bitwise_and(output_frame, output_frame, mask=cv2.bitwise_not(b_mask))
                    output_frame = cv2.add(output_frame, color_mask)
    
    return output_frame, qr_color

# Captura de video en tiempo real
cap = cv2.VideoCapture(0)
qr_color = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result_frame, qr_color = process_frame(frame, qr_color)
    
    cv2.imshow('Real-time Detection', result_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
