import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk
import time

class FaceRecognizer:
    def __init__(self):
        # Suppress macOS warnings
        os.environ['TK_SILENCE_DEPRECATION'] = '1'

        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.labels = {}
        self.last_movement = None
        self.last_recognized = None

        # Initialize Tkinter
        self.root = tk.Tk()
        self.root.title("Face Recognition")
        self.root.geometry("400x150")

        self.name_label = ttk.Label(
            self.root,
            text="Initializing...",
            font=('Helvetica', 24)
        )
        self.name_label.pack(pady=10)

        self.status_label = ttk.Label(
            self.root,
            text="Status: Loading",
            font=('Helvetica', 12)
        )
        self.status_label.pack()

        self.prev_time = time.time()

    def load_training_data(self, dataset_path, max_images=8):
        faces = []
        labels = []
        label_id = 0

        for person_name in sorted(os.listdir(dataset_path)):
            person_path = os.path.join(dataset_path, person_name)
            if not os.path.isdir(person_path):
                continue

            self.labels[label_id] = person_name
            print(f"\nTraining on {person_name}...")
            trained_images = 0

            # Get first 8 valid images
            for img_name in sorted(os.listdir(person_path))[:max_images * 2]:  # Check extra in case some fail
                if trained_images >= max_images:
                    break

                img_path = os.path.join(person_path, img_name)

                # Skip non-image files
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    print(f"Skipping non-image file: {img_name}")
                    continue

                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Could not read image: {img_name}")
                    continue

                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detect faces with more sensitive parameters
                faces_rect = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(50, 50)
                )

                if len(faces_rect) == 0:
                    # Try equalizing histogram
                    gray_eq = cv2.equalizeHist(gray)
                    faces_rect = self.face_cascade.detectMultiScale(
                        gray_eq,
                        scaleFactor=1.05,
                        minNeighbors=3,
                        minSize=(50, 50)
                    )

                if len(faces_rect) == 0:
                    print(f"No faces detected in: {img_name} - skipping")
                    continue

                # Use first face found
                (x, y, w, h) = faces_rect[0]
                face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                faces.append(face_roi)
                labels.append(label_id)
                trained_images += 1
                print(f"Added {img_name} ({trained_images}/{max_images})")

            print(f"Trained on {trained_images} images for {person_name}")
            label_id += 1

        if len(faces) == 0:
            print("\nERROR: No faces found in any training images!")
            return None, None

        return faces, np.array(labels)

    def train(self, data_dir="images", images_per_person=8):
        print("\n=== Training Started ===")
        faces, labels = self.load_training_data(data_dir, images_per_person)

        if faces is None:
            raise ValueError("No valid training data found!")

        print(f"\nTraining on {len(faces)} face samples...")
        self.recognizer.train(faces, labels)
        print(f"Training complete! Learned {len(self.labels)} identities")
        print("=== Ready for Recognition ===\n")

    def update_display(self, name, status, confidence):
        self.name_label.config(text=name)
        self.status_label.config(
            text=f"Status: {status} | Confidence: {confidence:.1f}"
        )
        self.root.update()

    def robot_movement(self, direction):
        """Improved movement with cooldown and same-person check"""
        current_time = time.time()

        # Only allow movement if:
        # - First detection, or
        # - New person detected, or
        # - 5 seconds passed since last movement
        if (self.last_movement is None or
                self.last_recognized != direction or
                (current_time - self.last_movement) > 5.0):
            print(f"Robot would move {direction} 2 feet")
            self.last_movement = current_time
            self.last_recognized = direction

            # Update status
            self.status_label.config(
                text=f"Moving {direction} | {time.strftime('%H:%M:%S')}"
            )

    def run_recognition(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.update_display("Error", "Camera not found", 0)
            return

        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.update_display("Ready", "Camera active", 0)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Mirror the frame
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(100, 100)
                )

                recognized = False

                for (x, y, w, h) in faces:
                    # Recognize face
                    face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                    label, confidence = self.recognizer.predict(face_roi)

                    # Determine identity
                    if confidence < 70:  # Lower = better match
                        name = self.labels.get(label, "Unknown")
                        status = "Friend"
                        color = (0, 255, 0)  # Green
                        self.robot_movement(name)  # Use name as direction identifier
                    else:
                        name = "Stranger"
                        status = "Danger!"
                        color = (0, 0, 255)  # Red
                        self.robot_movement("backward")

                    # Draw rectangle and label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, name, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.putText(frame, f"{confidence:.1f}", (x, y+h+20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # Update GUI
                    self.update_display(name, status, confidence)
                    recognized = True

                if not recognized:
                    self.update_display("No Face", "Waiting...", 0)
                    self.last_recognized = None

                # Display frame
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = FaceRecognizer()

    try:
        recognizer.train("images", images_per_person=8)
        recognizer.run_recognition()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        recognizer.root.destroy()