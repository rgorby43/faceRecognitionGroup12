import cv2
import numpy as np
import pickle
import os
import tkinter as tk
from tkinter import ttk


class FaceRecognizer:
    def __init__(self):
        # Check if required files exist
        if not os.path.exists("labels.pickle") or not os.path.exists("trainer.yml"):
            raise FileNotFoundError("Training files missing. Please run face_trainer.py first")

        # Load trained model and labels
        with open("labels.pickle", 'rb') as f:
            self.labels = pickle.load(f)
            self.labels = {v: k for k, v in self.labels.items()}  # Invert mapping

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("trainer.yml")

        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Face Recognition")
        self.name_label = ttk.Label(self.root, text="", font=('Helvetica', 24))
        self.name_label.pack(pady=20)
        self.confidence_label = ttk.Label(self.root, text="", font=('Helvetica', 14))
        self.confidence_label.pack()

        # Start video capture
        self.cap = cv2.VideoCapture(0)
        self.update()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]

                # Recognize face
                id_, confidence = self.recognizer.predict(roi_gray)

                if confidence < 60:  # Confidence threshold
                    name = self.labels.get(id_, "Unknown")
                    color = (0, 255, 0)  # Green
                else:
                    name = "Stranger"
                    color = (0, 0, 255)  # Red

                # Update GUI
                self.name_label.config(text=name)
                self.confidence_label.config(text=f"Confidence: {confidence:.1f}")

                # Draw on frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.imshow('Face Recognition', frame)

        self.root.after(10, self.update)  # Continuous update

    def run(self):
        self.root.mainloop()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        recognizer = FaceRecognizer()
        recognizer.run()
    except Exception as e:
        print(f"Error: {e}")