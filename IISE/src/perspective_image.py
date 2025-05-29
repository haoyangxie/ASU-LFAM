import cv2
import re
import matplotlib.pyplot as plt
import numpy as np


coordinates = []
with open('../input/hexagon/gcode.mpf', 'r') as file:
    for line in file:
        x_match = re.search(r'X(-?\d+\.\d+)', line)
        y_match = re.search(r'Y(-?\d+\.\d+)', line)

        if x_match and y_match:
            x_value = float(x_match.group(1))
            y_value = float(y_match.group(1))
            coordinates.append((x_value, y_value))

'scale to 320x256'
max_x = max(coordinates, key=lambda item: item[0])[0]
min_x = min(coordinates, key=lambda item: item[0])[0]
max_y = max(coordinates, key=lambda item: item[1])[1]
min_y = min(coordinates, key=lambda item: item[1])[1]
new_width, new_height = 320, 256
scale_x = new_width / (max_x - min_x)
scale_y = new_height / (max_y - min_y)


def transform_coordinates(x, y):
    new_x = (x - min_x) * scale_x
    new_y = (y - min_y) * scale_y
    return new_y, -new_x


new_x_values = []
new_y_values = []
for x, y in coordinates:
    new_x, new_y = transform_coordinates(x, y)
    new_x_values.append(new_x)
    new_y_values.append(new_y)

# plt.plot(new_x_values, new_y_values)
# plt.show()

image = cv2.imread('../input/hexagon/hexagon_image/hex_with_rect_372.png')
pts1 = np.float32([[36, 15], [284, 17], [9, 232], [311, 231]])
pts2 = np.float32([[0, 256], [0, 0], [320, 256], [320, 0]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
img_output = cv2.warpPerspective(image, matrix, (320, 256))

for x, y in zip(new_x_values, new_y_values):
    cv2.circle(img_output, (int(-y), int(x)), 1, (0, 255, 0), -1)

final_coordinates = [(int(-y), int(x)) for x, y in zip(new_x_values, new_y_values)]
final_coordinates_set = set(final_coordinates)
save_coordinates = np.array(list(final_coordinates_set))
np.savetxt('coordinates_from_gcode.txt', save_coordinates, fmt='%d')

cv2.imshow('image', img_output)
cv2.waitKey(0)