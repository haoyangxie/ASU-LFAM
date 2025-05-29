import numpy as np
import cv2


pts1 = np.float32([[36, 15], [284, 17], [9, 232], [311, 231]])
pts2 = np.float32([[0, 256], [0, 0], [320, 256], [320, 0]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
coordinates_from_gcode = np.loadtxt('coordinates_from_gcode.txt', dtype=int)

cap = cv2.VideoCapture('../input/hexagon/hexagon_video.wmv')
while True:
    ret, frame = cap.read()

    if not ret:
        break

    transformer_frame = cv2.warpPerspective(frame, matrix, (320, 256))
    for coordinate in coordinates_from_gcode:
        cv2.circle(transformer_frame, (coordinate[0], coordinate[1]), 1, (0, 255, 0), -1)
    cv2.imshow('transformed video', transformer_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()