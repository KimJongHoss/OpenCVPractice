import cv2
import numpy as np
import matplotlib.pyplot as plt

# 평균 필터
# img = cv2.imread("testImage.jpg")
#
# blurred_image = cv2.blur(img, (5,5))
#
# cv2.imshow("origin Image", img)
# cv2.imshow("Blurred Image", blurred_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#가우시안 필터
# img = cv2.imread("testImage.jpg")
#
# gaussian_blur = cv2.GaussianBlur(img, (15, 15), 0)
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.title("원본 이미지")
# plt.axis("off")
#
# plt.subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB))
# plt.title("가우시안 필터 적용")
# plt.axis("off")
#
# plt.show()

#중앙값 필터
# img = cv2.imread("testImage.jpg")
#
# median_blur = cv2.medianBlur(img, 15)
#
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.title("original Image")
# plt.axis("off")
#
#
# plt.subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(median_blur, cv2.COLOR_BGR2RGB))
# plt.title("blurr Image")
# plt.axis("off")
#
# plt.show()

#양면 필터

# img = cv2.imread("testImage.jpg")
#
# bilateral_filter = cv2.bilateralFilter(img, 15, 75, 75)
#
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.title("original Image")
# plt.axis("off")
#
# plt.subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(bilateral_filter, cv2.COLOR_BGR2RGB))
# plt.title("bilateral Filter Image")
# plt.axis("off")
#
# plt.show()

#소벨 필터

# img = cv2.imread("testImage.jpg", cv2.IMREAD_GRAYSCALE)
#
# sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
#
# sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
#
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 3, 1)
# plt.imshow(img, cmap="gray")
# plt.title("original Image")
# plt.axis("off")
#
# plt.subplot(1, 3, 2)
# plt.imshow(sobelx, cmap="gray")
# plt.title("sobel-x")
# plt.axis("off")
#
# plt.subplot(1, 3, 3)
# plt.imshow(sobely, cmap="gray")
# plt.title("sobel-y")
# plt.axis("off")
#
# plt.show()

#샤르 필터 - 미세한 디테일, 날카로운 모서리 감지

# img = cv2.imread("testImage.jpg", cv2.IMREAD_GRAYSCALE)
#
# scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
#
# scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
#
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 3, 1)
# plt.imshow(img, cmap="gray")
# plt.title("original Image")
# plt.axis("off")
#
# plt.subplot(1, 3, 2)
# plt.imshow(scharrx, cmap="gray")
# plt.title("scharr-x")
# plt.axis("off")
#
# plt.subplot(1, 3, 3)
# plt.imshow(scharry, cmap="gray")
# plt.title("scharr-y")
# plt.axis("off")
#
# plt.show()

#라플라시안 필터 - 이미지의 엣지를 감지하는데 사용

# img = cv2.imread("testRamen.jpg", cv2.IMREAD_GRAYSCALE)
#
# laplacian = cv2.Laplacian(img, cv2.CV_64F)
#
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.imshow(img, cmap="gray")
# plt.title("original Image")
# plt.axis("off")
#
# plt.subplot(1, 2, 2)
# plt.imshow(laplacian, cmap="gray")
# plt.title("Laplacian Image")
# plt.axis("off")
#
# plt.show()

#임계값 처리

# img = cv2.imread("testRobot.jpg", cv2.IMREAD_GRAYSCALE)
#
# _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#
# cv2.imshow("Original Image", img)
# cv2.imshow("Binary Image", binary_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#적응형 임계값 처리 - 서로 다른 임계값을 계산하여 이진화하는 기법

# img = cv2.imread("testRobot.jpg", cv2.IMREAD_GRAYSCALE)
# adaptive_thresh = cv2.adaptiveThreshold(
#     img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY, 11, 2
# )
#
# cv2.imshow("Adaptive Thresholded Image", adaptive_thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

###########
# img = cv2.imread("testRobot.jpg", cv2.IMREAD_GRAYSCALE)
# mean_thresh  = cv2.adaptiveThreshold(
#     img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY, 11, 2
# )
#
# gaussian_thresh = cv2.adaptiveThreshold(
#     img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY, 11, 2
# )
#
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 3, 1)
# plt.imshow(img, cmap="gray")
# plt.title("Original Image")
# plt.axis("off")
#
# plt.subplot(1, 3, 2)
# plt.imshow(mean_thresh, cmap="gray")
# plt.title("Mean Thresh")
# plt.axis("off")
#
# plt.subplot(1, 3, 3)
# plt.imshow(gaussian_thresh, cmap="gray")
# plt.title("Gaussian Thresh")
# plt.axis("off")
#
# plt.show()

#############자동 임계값

# img = cv2.imread("testRobot.jpg", cv2.IMREAD_GRAYSCALE)
#
# _, otsu_thresh = cv2.threshold(img, 0, 255,
# cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.hist(img.ravel(), 256, [0, 256])
# plt.title("Original Image")
#
# plt.subplot(1, 2, 2)
# plt.imshow(otsu_thresh, cmap="gray")
# plt.title("Otsu Image")
# plt.axis("off")
#
# plt.show()

#경계 검출 : 가우시안 블러, Sobel, 히스테리시스 임계값 처리

img = cv2.imread("testRobot.jpg", cv2.IMREAD_GRAYSCALE)

canny_edges = cv2.Canny(img, 100, 200)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(canny_edges, cmap="gray")
plt.title("Canny Edges Image")
plt.axis("off")

plt.show()