import cv2
import cv2.aruco as aruco
import numpy as np

# Camera calibration parameters (These should be replaced with actual values for your camera)
camera_matrix = np.array([[600, 0, 320],
                          [0, 600, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)  # Assuming no lens distortion

# ArUco dictionary and detection parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_params = aruco.DetectorParameters()

# OpenCV to capture video
cap = cv2.VideoCapture(0)  # Using the first camera (you can change the index if needed)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert the image to grayscale (ArUco marker detection works better in grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejected = aruco.detectMarkers(
        gray, aruco_dict, parameters=aruco_params
    )

    if ids is not None:
        # Estimate pose of the markers
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, 100, camera_matrix, dist_coeffs
        )

        for i in range(len(ids)):
            # Draw detected markers and their axes
            aruco.drawDetectedMarkers(frame, corners)
            # Use cv2.drawFrameAxes instead of aruco.drawAxis
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 50)

            marker_id = ids[i][0]  # Marker ID
            x, y, z = tvecs[i][0]  # Position of the marker (translation vector)

            # Determine whether to pass on the left or right side of the marker
            side = "LEFT" if marker_id % 2 == 1 else "RIGHT"

            # Display marker information on the frame
            cv2.putText(frame, f"ID: {marker_id} Pass: {side}", (10, 40 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Show marker's position (X, Y in millimeters)
            cv2.putText(frame, f"X: {x:.1f} Y: {y:.1f} Z: {z:.1f}", (10, 80 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display the frame with marker detection results
    cv2.imshow("ArUco Marker Detection", frame)

    # Break the loop if the 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
