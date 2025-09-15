from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
from ultralytics import YOLO
import os

# Load YOLO model (change to your path, e.g. 'best.pt')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "best.pt")

print("Loading model from:", model_path)
model = YOLO(model_path)

def gen_frames():
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Run YOLO inference
            results = model(frame, stream=True)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]  # Emotion class from training

                    # Draw bounding box + label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (36, 255, 12), 2)

            # Encode frame for web streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def index(request):
    return render(request, 'camera/index.html')

def video_feed(request):
    return StreamingHttpResponse(gen_frames(),
                    content_type='multipart/x-mixed-replace; boundary=frame')
