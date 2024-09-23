import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.testutils import approx

#커널

# # 사각형 커널
# rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#
# # 타원형 커널
# ellipsis_kenel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#
# # 십자가형 커널
# cross_kenel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
#
# # 출력
# print("Rectangular Kernel:\n", rect_kernel)
# print("Elliptical Kernel:\n", ellipsis_kenel)
# print("Cross-shaped Kernel:\n", cross_kenel)

# 확장

#이미지 로드 및 이진화
# img = cv2.imread("testRobot.jpg", 0)
# _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#
# #커널 생성
# kernel = np.ones((5, 5), np.uint8)
#
# #확장 적용
# dilation = cv2.dilate(binary_image, kernel, iterations=1)
#
# #결과 표시
#
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.imshow(binary_image, cmap="gray")
# plt.title("Original Image")
# plt.axis("off")
#
# plt.subplot(1, 2, 2)
# plt.imshow(dilation, cmap="gray")
# plt.title("Dilation Image")
# plt.axis("off")
#
# plt.show()

#침식

# #이미지 로드 및 이진화
# img = cv2.imread("testRobot.jpg", 0)
# _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#
# #커널 생성
# kernel = np.ones((5, 5), np.uint8)
#
# #확장 적용
# erosion = cv2.erode(binary_image, kernel, iterations=1)
#
# #결과 표시
#
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.imshow(binary_image, cmap="gray")
# plt.title("Original Image")
# plt.axis("off")
#
# plt.subplot(1, 2, 2)
# plt.imshow(erosion, cmap="gray")
# plt.title("Erosion Image")
# plt.axis("off")
#
# plt.show()

#열기 : 침식 후 팽창 ->작은 소음 제거

# img = cv2.imread("testRobot.jpg", 0)
# _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#
# #커널 생성
# kernel = np.ones((5, 5), np.uint8)
#
# #열기 적용
# opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
#
# #결과 표시
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.imshow(binary_image, cmap="gray")
# plt.title("Original Image")
# plt.axis("off")
#
# plt.subplot(1, 2, 2)
# plt.imshow(opening, cmap="gray")
# plt.title("Opening Image")
# plt.axis("off")
#
# plt.show()

# 종결 : 팽창 후 침식 -> 물체의 작은 구멍을 메우는 기능

# img = cv2.imread("testRobot.jpg", 0)
# _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#
# #커널 생성
# kernel = np.ones((5, 5), np.uint8)
#
# #열기 적용
# closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
#
# #결과 표시
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.imshow(binary_image, cmap="gray")
# plt.title("Original Image")
# plt.axis("off")
#
# plt.subplot(1, 2, 2)
# plt.imshow(closing, cmap="gray")
# plt.title("Clossing Image")
# plt.axis("off")
#
# plt.show()

# 그라데이션 : 이미지의 팽창과 침식의 차이

# img = cv2.imread("testRobot.jpg", 0)
# _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#
# #커널 생성
# kernel = np.ones((5, 5), np.uint8)
#
# #열기 적용
# gradient = cv2.morphologyEx(binary_image, cv2.MORPH_GRADIENT, kernel)
#
# #결과 표시
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.imshow(binary_image, cmap="gray")
# plt.title("Original Image")
# plt.axis("off")
#
# plt.subplot(1, 2, 2)
# plt.imshow(gradient, cmap="gray")
# plt.title("Gradient Image")
# plt.axis("off")
#
# plt.show()

# 윤곽 탐지 및 분석

# #이미지 로드 및 그레이스케일로 변환
# img = cv2.imread("testRobot.jpg", 0)
#
# #이진화
# _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#
# #컨투어 찾기
# contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL,
#                                        cv2.CHAIN_APPROX_SIMPLE)
#
# # 원본 이미지에 컨투어 그리기
# contour_image = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 2)
#
# #결과 표시
#
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.imshow(binary_image, cmap="gray")
# plt.title("Binary Image")
# plt.axis("off")
#
# plt.subplot(1, 2, 2)
# plt.imshow(contour_image, cmap="gray")
# plt.title("Contour Image")
# plt.axis("off")
#
# plt.show()

# #이미지 로드 및 그레이스케일로 변환2
# img = cv2.imread("testRobot.jpg")
#
# #이진화
# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
#
# #컨투어 찾기
# contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE,
#                                        cv2.CHAIN_APPROX_SIMPLE)
#
# # 원본 이미지에 컨투어 그리기
# contour_image = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 2)
#
# #결과 표시
# cv2.imshow("Contours", contour_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #이미지 로드 및 그레이스케일로 변환3
# img = cv2.imread("testRamen.jpg")
#
# #이진화
# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
#
# #컨투어 찾기
# contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE,
#                                        cv2.CHAIN_APPROX_SIMPLE)
# # 첫 번째 컨투어 선택
# cnt = contours[0]
#
# #컨투어 면적 계산
# area = cv2.contourArea(cnt)
#
# perimeter = cv2.arcLength(cnt, True)
#
# # 컨투어의 경계 사각형 그리기
# x, y, w, h =cv2.boundingRect(cnt)
# bounding_image = cv2.rectangle(img.copy(), (x, y), (x+w, y+h), (255, 0, 0), 2)
#
# #결과 출력
# print(f"Contour Area: {area}")
# print(f"Contour Area: {perimeter}")
#
# #경계 사각형이 그려진 이미지 표시
# cv2.imshow("Bounding Rectangle", bounding_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #윤곽 탐지 및 분석
#
# img = cv2.imread("testImage.jpg")
#
# #이진화
# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
#
# #컨투어 찾기
# contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE,
#                                        cv2.CHAIN_APPROX_SIMPLE)
#
# # 첫 번째 컨투어의 면적 계산
# eplision = 0.02 * cv2.arcLength(contours[0], True)
# approx = cv2.approxPolyDP(contours[0], eplision, True)
#
# #원본 이미지에 근사화된 컨투어 그리기
# approx_image = cv2.drawContours(img.copy(), [approx], -1, (0, 0, 255), 2)
#
# #결과 표시
# cv2.imshow("Approx polyDP", approx_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#윤곽 탐지 및 분석2
#윤곽 탐지 및 분석

img = cv2.imread("testRamen.jpg")

#이진화
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

#컨투어 찾기
contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

# 첫 번째 컨투어의 면적 계산
hull = cv2.convexHull(contours[0])

#원본 이미지에 근사화된 컨투어 그리기
hull_image = cv2.drawContours(img.copy(), [hull], -1, (0, 255, 255), 2)

#결과 표시
cv2.imshow("Convex Hull", hull_image)
cv2.waitKey(0)
cv2.destroyAllWindows()