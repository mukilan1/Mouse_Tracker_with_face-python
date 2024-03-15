import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Start capturing video from the default camera.
cap = cv2.VideoCapture(0)

screen_width, screen_height = pyautogui.size()
camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Exponential Moving Average (EMA) smoothing factors
smoothing_factor_x = 0.5
smoothing_factor_y = 0.5
ema_x, ema_y = camera_width // 2, camera_height // 2  # Initialize to center

def apply_non_linear_scale(value, scale=1.5, midpoint=0.5):
    """Apply a non-linear scaling to enhance sensitivity of smaller movements."""
    # Normalize value based on its midpoint
    normalized = (value - midpoint) / (1 - midpoint) if value > midpoint else value / midpoint
    return np.sign(normalized) * (abs(normalized) ** scale)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[1]
            x = int(nose_tip.x * camera_width)
            y = int(nose_tip.y * camera_height)

            # Apply EMA Smoothing
            ema_x = (x * smoothing_factor_x) + (ema_x * (1 - smoothing_factor_x))
            ema_y = (y * smoothing_factor_y) + (ema_y * (1 - smoothing_factor_y))

            # Convert to a range from 0 to 1 for non-linear scaling
            normalized_x = ema_x / camera_width
            normalized_y = ema_y / camera_height

            # Apply non-linear scaling
            scaled_x = apply_non_linear_scale(normalized_x) * screen_width
            scaled_y = apply_non_linear_scale(normalized_y) * screen_height
# Move the mouse to the scaled position, ensuring it stays within screen bounds
            clamped_x = np.clip(scaled_x, 0, screen_width)
            clamped_y = np.clip(scaled_y, 0, screen_height)
            pyautogui.moveTo(clamped_x, clamped_y)



            mp_drawing.draw_landmarks(image=image,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=drawing_spec,
                                      connection_drawing_spec=drawing_spec)

    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
