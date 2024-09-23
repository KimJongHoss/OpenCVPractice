import cv2
import numpy as np
from numpy.ma.core import resize
from numpy.polynomial.polynomial import polyline
import matplotlib.pyplot as plt


# img = cv2.imread("testImage.jpg")
#
# cv2.imshow('Image', img)
#
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()

# img = cv2.imread("testImage.jpg")
#
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# resized_img = cv2.resize(gray_img, (300, 300))
#
# # cv2.imshow("Grayscale Image" , gray_img)
#
# cv2.imshow("Resized Image", resized_img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.imread("testImage.jpg")

# original_height, original_width = img.shape[:2]
#
# new_width = 400
# new_height = int(original_height * (new_width/original_width))
#
# resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
#
# cv2.imshow("original Image", img)
# cv2.imshow("Resized Image with Aspect Ratio", resized_img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cropped_img = img[50:500, 100:600]
# cv2.imshow("Cropped Image", cropped_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# rows, cols = img.shape[:2]
#
# M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 0.5)
#
# rotated_img = cv2.warpAffine(img, M, (cols, rows))
#
# cv2.imshow("Rotated_img", rotated_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# flipped_img_h = cv2.flip(img, 1)
# flipped_img_v = cv2.flip(img, 0)
# flipped_img_hv = cv2.flip(img, -1)
#
# cv2.imshow("Flipped Horizontally", flipped_img_h)
# cv2.imshow("Flipped Vertically", flipped_img_v)
# cv2.imshow("Flipped Horizontally and Vertically", flipped_img_hv)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.line(img, (50, 50), (200, 50), (255, 0, 0), 5)
#
# cv2.rectangle(img, (50, 100), (200, 200), (0, 255, 0), 3)
#
# cv2.circle(img,(300, 150), 50, (0, 0, 255), -1)
#
# cv2.imshow("Shapes", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# polyline_pt = np.array([[50, 50], [50, 100], [75, 250], [100, 150], [75, 50]], np.int32)
#
# cv2.polylines(img, polyline_pt, True, (255, 0, 0), 3)
# cv2.ellipse(img, (150, 150), (150, 200), 0, 0, 360, (0, 255, 0), 2 )
#
# cv2.imshow("ellipse", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.putText(img, "View", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
#
# cv2.imshow("Text on Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img1 = cv2.imread("testImage.jpg")
# img2 = cv2.imread("testRamen.jpg")
# resizeImg1 = cv2.resize(img1, (1000, 700))
# resizeImg2 = cv2.resize(img2, (1000, 700))
#
# alpha = 0.7
# beta = 0.3
#
# blended_image = cv2.addWeighted(resizeImg1, alpha, resizeImg2, beta, 0)
#
# cv2.imshow("blended Image", blended_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.imread("testImage.jpg")
#
# mask = np.zeros(img.shape[:2], dtype="uint8")

# cv2.rectangle(mask, (200, 500), (300, 300), 255, -1)
#
# masked_image = cv2.bitwise_and(img, img, mask=mask)
#
# cv2.imshow("Mask", mask)
# cv2.imshow("Masked Image", masked_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.imread("testImage.jpg")
#
# mask = np.zeros(img.shape[:2], dtype="uint8")
#
# cv2.circle(mask, (250, 250), 100, 255, -1)
# pts = np.array([[300, 200], [400, 400], [200, 400]], np.int32)
# pts = pts.reshape((-1, 1, 2))
# cv2.fillPoly(mask, [pts], 255)
#
# masked_Image = cv2.bitwise_and(img, img, mask=mask)
#
# cv2.imshow("Complex Mask", mask)
# cv2.imshow("Masked Image with Complex Mask", masked_Image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.imread("testRamen.jpg")
#
# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1,3,1)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))
# plt.title("Original Image")
#
# plt.subplot(1,3,2)
# plt.imshow(gray_image, cmap="gray")
# plt.title("Grayscale Image")
#
# plt.subplot(1,3,3)
# plt.imshow(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR))
# plt.title("HSV Image")
#
# plt.show()


img = cv2.imread("testImage.jpg")

B, G, R = cv2.split(img)

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(B, cmap="gray")
plt.title("Blue Channel")

plt.subplot(1, 3, 2)
plt.imshow(G, cmap="gray")
plt.title("Green Channel")

plt.subplot(1, 3, 3)
plt.imshow(R, cmap="gray")
plt.title("Red Channel")

plt.show()

merged_image = cv2.merge([B, G, R])

plt.show(cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB))
plt.title("Merged Image")
plt.show()

# img = cv2.imread("testRamen.jpg")
# hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
# lower_red = np.array([0,120,70])
# upper_red = np.array([10,255,255])
#
# red_mask = cv2.inRange(hsv_img, lower_red, upper_red)
#
# red_segment = cv2.bitwise_and(img, img, mask=red_mask)
#
# cv2.imshow("Red Segment", red_segment)
# cv2.waitKey(0)
# cv2.destroyAllWindows()