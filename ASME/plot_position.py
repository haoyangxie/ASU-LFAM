import cv2

image = cv2.imread('./totem_6155.png')
# (80, 116), (50, 165), (141, 87), (258, 165)
coordinates = [(80, 116), (50, 165), (141, 87), (258, 165)]
for (x,y) in coordinates:
    cv2.circle(image, (x,y), 5, (0,255,0), -1)
cv2.imshow('image', image)
cv2.imwrite('./paper_figures/totems_positions.png', image)
# cv2.waitKey(0)