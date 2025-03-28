import pyrealsense2 as rs
import cv2
import numpy as np
import pickle
import os
import tkinter as tk
from tkinter import ttk
import time


class PiFaceRecognizer:
    def __init__(self):
        # Verify training files
        if not all(os.path.exists(f) for f in ["labels.pickle", "trainer.yml"]):
            raise FileNotFoundError("Run face_trainer.py first to generate training data")

        # Load label mappings
        with open("labels.pickle", "rb") as f:
            self.labels = {v: k for k, v in pickle.load(f).items()}

        # Load trained model
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("trainer.yml")

        # RealSense pipeline configuration for Pi
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)  # Lower FPS for Pi

        # Start pipeline
        self.pipeline.start(config)

        # Pi-optimized face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Lightweight GUI for Pi (using Tkinter with Pi-friendly settings)
        self.root = tk.Tk()
        self.root.title("Pi Face Recognition")
        self.root.geometry("300x100")  # Smaller window for Pi
        self.root.configure(bg='white')

        # Use framebuffer display if running on Pi console
        if os.environ.get('DISPLAY', '') == '':
            os.environ.__setitem__('DISPLAY', ':0.0')

        # Simple UI elements
        self.name_label = ttk.Label(self.root, text="Loading...", font=('Helvetica', 16))
        self.name_label.pack()
        self.confidence_label = ttk.Label(self.root, text="", font=('Helvetica', 10))
        self.confidence_label.pack()

        # Performance tracking
        self.frame_time = time.time()
        self.last_gui_update = 0

    def setup_pi_display(self):
        """Configure display for Raspberry Pi hardware"""
        # Set appropriate backend for cv2
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4)  # Use all 4 cores on Pi

        # Configure window properties for Pi
        cv2.namedWindow('Pi Face Recognition', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Pi Face Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def preprocess_face(self, face_img):
        """Optimized preprocessing for Pi"""
        face_img = cv2.equalizeHist(face_img)
        return cv2.resize(face_img, (180, 180))  # Smaller size for Pi

    def update_gui(self, name, confidence):
        """Efficient GUI updates for Pi"""
        current_time = time.time()
        if current_time - self.last_gui_update > 0.5:  # Throttle updates
            self.name_label.config(text=name)
            self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
            self.last_gui_update = current_time
            self.root.update()

    def run_recognition(self):
        self.setup_pi_display()
        try:
            while True:
                # RealSense frame capture
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Convert to numpy array
                frame = np.asanyarray(color_frame.get_data())

                # Pi-optimized processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,  # Higher for better Pi detection
                    minNeighbors=4,  # Lower for Pi performance
                    minSize=(80, 80)  # Smaller minimum size
                )

                for (x, y, w, h) in faces:
                    face_roi = gray[y:y + h, x:x + w]
                    processed_face = self.preprocess_face(face_roi)

                    label, confidence = self.recognizer.predict(processed_face)
                    name = self.labels.get(label, "Unknown") if confidence < 70 else "Stranger"
                    color = (0, 255, 0) if name != "Stranger" else (0, 0, 255)

                    # Draw on frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, name, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

                    self.update_gui(name, confidence)

                # Pi-optimized display
                cv2.imshow('Pi Face Recognition', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            self.root.destroy()


if __name__ == "__main__":
    try:
        app = PiFaceRecognizer()
        app.root.after(100, app.run_recognition)
        app.root.mainloop()
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)