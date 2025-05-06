import time
from faceRecognition import RealSenseFaceDetector

if __name__ == "__main__":
    print("Main script started - Face Detection Pre-Check.")

    required_consistency = 1

    print(f"\nAttempting face detection pre-check (need {required_consistency}s consistency)...")

    my_detector = RealSenseFaceDetector()

    face_detected = my_detector.wait_for_consistent_face(duration=required_consistency)

    if face_detected:
        print("\nSUCCESS: Consistent face detected!")
        print("Proceeding with the next series of calls...")
    else:
        print("\nFAILURE: Face detection did not meet consistency requirement or was stopped manually.")
        print("Cannot proceed with subsequent calls.")
