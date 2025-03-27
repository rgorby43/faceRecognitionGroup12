import cv2
import numpy as np
import os
import pickle
from PIL import Image
from PIL import UnidentifiedImageError


def train_faces():
    # Initialize paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, "images")
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")

    # Initialize face detector and recognizer
    face_cascade = cv2.CascadeClassifier(cascade_path)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids = {}
    training_data = []
    labels = []
    skipped_files = []

    print("Starting training process...")
    print(f"Looking for images in: {image_dir}")

    # Step through each person's folder
    for root, dirs, files in os.walk(image_dir):
        if root == image_dir:
            continue  # Skip the base directory

        # Get the label name from folder name
        label = os.path.basename(root).lower()
        if label not in label_ids:
            label_ids[label] = current_id
            current_id += 1

        print(f"\nProcessing {label}'s images...")
        processed_count = 0

        # Process each image
        for file in files:
            if not file.lower().endswith(('png', 'jpg', 'jpeg')):
                continue

            path = os.path.join(root, file)

            try:
                # Try to open and convert the image
                with Image.open(path) as img:
                    pil_image = img.convert("L")
                    image_array = np.array(pil_image, "uint8")

                # Detect faces
                faces = face_cascade.detectMultiScale(
                    image_array,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(100, 100)
                )

                if len(faces) == 0:
                    print(f"  No faces found in {file}")
                    skipped_files.append(path)
                    continue

                for (x, y, w, h) in faces:
                    roi = image_array[y:y + h, x:x + w]
                    training_data.append(roi)
                    labels.append(label_ids[label])

                processed_count += 1
                print(f"  Processed {file} successfully")

            except UnidentifiedImageError:
                print(f"  ERROR: Could not identify image file {file}")
                skipped_files.append(path)
            except Exception as e:
                print(f"  ERROR processing {file}: {str(e)}")
                skipped_files.append(path)

        print(f"Processed {processed_count} images for {label}")

    if not training_data:
        print("\nERROR: No valid training data found!")
        if skipped_files:
            print("\nProblem files:")
            for f in skipped_files:
                print(f"  {f}")
        return

    # Save label mapping
    with open("labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)

    # Train and save recognizer
    print("\nTraining model...")
    recognizer.train(training_data, np.array(labels))
    recognizer.save("trainer.yml")

    print(f"\nTraining complete! Processed {len(training_data)} face samples.")
    print(f"Created labels.pickle and trainer.yml")

    if skipped_files:
        print("\nSkipped files (check if these are valid images):")
        for f in skipped_files:
            print(f"  {f}")


if __name__ == "__main__":
    train_faces()