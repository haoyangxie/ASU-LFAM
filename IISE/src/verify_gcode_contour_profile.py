import numpy as np
import cv2


contour = np.loadtxt('./coordinates_from_gcode.txt')
contour = contour.reshape(-1, 1, 2)
pts1 = np.float32([[36, 15], [284, 17], [9, 232], [311, 231]])
pts2 = np.float32([[0, 256], [0, 0], [320, 256], [320, 0]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
inverse_matrix = np.linalg.inv(matrix)
original_points = cv2.perspectiveTransform(contour, inverse_matrix)
original_points = original_points.reshape(-1, 2)
image = cv2.imread('../input/hexagon/hexagon_image/hex_with_rect_372.png')

for point in original_points:
    cv2.circle(image, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)

cv2.imshow('image', image)
cv2.waitKey(0)

