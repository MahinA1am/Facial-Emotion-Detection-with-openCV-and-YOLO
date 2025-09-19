from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
from ultralytics import YOLO
import os
import numpy as np

# Load YOLO emotion model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "best.pt")
print("Loading model from:", model_path)
model = YOLO(model_path)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --- Enhancement functions ---
def histogram_equalization(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

def gamma_correction(face, gamma=1.5):
    invGamma = 1.0 / gamma
    table = (np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])).astype("uint8")
    return cv2.LUT(face, table)

def denoise(face):
    return cv2.fastNlMeansDenoisingColored(face, None, 10, 10, 7, 21)

def sharpen(face):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(face, -1, kernel)

# Choose enhancement method
ENHANCEMENT = "hist_eq"  # options: "hist_eq", "gamma", "denoise", "sharpen", None

def enhance_face(face):
    if ENHANCEMENT == "hist_eq":
        return histogram_equalization(face)
    elif ENHANCEMENT == "gamma":
        return gamma_correction(face, gamma=1.5)
    elif ENHANCEMENT == "denoise":
        return denoise(face)
    elif ENHANCEMENT == "sharpen":
        return sharpen(face)
    else:
        return face  # no enhancement

# --- Video streaming generator ---
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]

            # Enhance face
            enhanced_face = enhance_face(face_roi)

            # Resize to YOLO input size
            resized_face = cv2.resize(enhanced_face, (640, 640))

            # Run YOLO inference
            results = model(resized_face, conf=0.15)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    # Draw box and label on original frame
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

        # Encode frame for streaming
        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

# --- Django views ---
def index(request):
    return render(request, "camera/index.html")

def video_feed(request):
    return StreamingHttpResponse(
        gen_frames(),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )
