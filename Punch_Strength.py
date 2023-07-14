import cv2
import mediapipe as mp
import math
import time

cap = cv2.VideoCapture(0)
pTime = 0

# Drawing Points
mpDraw = mp.solutions.drawing_utils
# Pose Detection
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Set target point coordinates
target_point = (300, 200)  # Example target point coordinates (x, y)

# Initialize hand point positions and times
hand_points_positions = {}
hand_points_times = {}

while True:
    success, img = cap.read()
    # Conversion because mediapipe works on RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks,
                              mpPose.POSE_CONNECTIONS, landmark_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 0),
                                                                                                 thickness=2,
                                                                                                 circle_radius=2),
                              connection_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2)
                              )
        # Each id/point
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Track hand point positions and times
            if id in [mpPose.PoseLandmark.LEFT_WRIST.value, mpPose.PoseLandmark.LEFT_INDEX.value,
                      mpPose.PoseLandmark.LEFT_THUMB.value, mpPose.PoseLandmark.RIGHT_WRIST.value,
                      mpPose.PoseLandmark.RIGHT_INDEX.value, mpPose.PoseLandmark.RIGHT_THUMB.value]:
                if id not in hand_points_positions:
                    hand_points_positions[id] = []
                hand_points_positions[id].append((cx, cy))
                hand_points_times[id] = time.time()

        # Calculate hand point speeds and distances from target point
        hand_points_speeds = {}
        hand_points_distances = {}
        for id, positions in hand_points_positions.items():
            if len(positions) >= 2:
                x1, y1 = positions[-2]
                x2, y2 = positions[-1]
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                time_diff = hand_points_times[id] - pTime
                speed = distance / time_diff if time_diff > 0 else 0
                hand_points_speeds[id] = speed
                hand_points_distances[id] = math.sqrt((x2 - target_point[0]) ** 2 + (y2 - target_point[1]) ** 2)

        # Calculate punch power based on speed and distance from target point
        punch_power = sum(hand_points_speeds.values()) * sum(hand_points_distances.values()) if hand_points_speeds and hand_points_distances else 0
        punch_power = punch_power / 1000
        # Display punch power on the screen
        cv2.putText(img, f"Punch Power: {punch_power:.2f}", (70, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255), 2)

        # Draw power point with purple color
        cv2.circle(img, target_point, 8, (128, 0, 128), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (70, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)  # 1 millisecond delay