from tempfile import template

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 히스토그램

# #이미지 로드 및 그레이스케일로 전환
# img = cv2.imread("testRobot.jpg", cv2.IMREAD_GRAYSCALE)
#
# # 히스토그램 계산
# hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# # 히스토그램 플로팅
# plt.figure(figsize=(8, 6))
# plt.plot(hist, color="black")
# plt.title("히스토그램")
# plt.xlabel("픽셀 값")
# plt.ylabel("픽셀 수")
# plt.xlim([0, 256])
# plt.show()

# #히스토그램 균등화
#
# #이미지 로드 및 그레이스케일로 전환
# img = cv2.imread("testRobot.jpg", cv2.IMREAD_GRAYSCALE)
#
# # 히스토그램 평활화 적용
# equalized_image = cv2.equalizeHist(img)
#
# plt.figure(figsize=(12, 6))
#
# plt.subplot(2, 2, 1)
# plt.imshow(img, cmap="gray")
# plt.title("원본 이미지")
# plt.axis("off")
#
# plt.subplot(2, 2, 2)
# plt.hist(img.ravel(), 256, [0, 256])
# plt.title("원본 히스토그램")
#
# plt.subplot(2, 2, 3)
# plt.imshow(equalized_image, cmap="gray")
# plt.title("히스토그램 평활화 이미지")
# plt.axis("off")
#
# plt.subplot(2, 2, 4)
# plt.hist(equalized_image.ravel(), 256, [0, 256])
# plt.title("히스토그램 평활화 히스토그램")
#
# plt.show()

#히스토그램 균등화

# #이미지 로드 및 그레이스케일로 전환
# img = cv2.imread("testRobot.jpg", cv2.IMREAD_GRAYSCALE)
#
# # CLAHE 객체 생성
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#
# #CLAHE 적용
# clahe_image = clahe.apply(img)
#
# #원본 및 CLAHE 이미지 비교
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.imshow(img, cmap="gray")
# plt.title("원본 이미지")
# plt.axis("off")
#
# plt.subplot(1, 2, 2)
# plt.imshow(clahe_image, cmap="gray")
# plt.title("CLAHE 이미지")
# plt.axis("off")
#
# plt.show()

# 가우스 피라미드

# 이미지 로드
# img = cv2.imread("testRobot.jpg")
#
# # 가우스 피라미드 생성
# layer = img.copy()
# gaussian_pyramid = [layer]
#
# for i in range(3): #3단계 피라미드 생성
#     layer = cv2.pyrDown(layer)
#     gaussian_pyramid.append(layer)
#
# # 피라미드 단계별 이미지 표시
# plt.figure(figsize=(12, 6))
# for i in range(4):
#     plt.subplot(1, 4, i+1)
#     plt.imshow(cv2.cvtColor(gaussian_pyramid[i], cv2.COLOR_BGR2RGB))
#     plt.title(f'label {i}')
#     plt.axis("off")
#
# plt.show()
#
# #라플라시안 피라미드
#
# # 이미지 로드
# img = cv2.imread("testRobot.jpg")
#
# # 가우스 피라미드 생성
# layer = img.copy()
# gaussian_pyramid = [layer]
#
# for i in range(3): #3단계 피라미드 생성
#     layer = cv2.pyrDown(layer)
#     gaussian_pyramid.append(layer)
#
# # 라플라시안 피라미드 생성
# laplacian_pyramid = []
#
# for i in range(3, 0, -1): #3단계 피라미드 생성
#     gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i])
#
# # gaussian_expanded를 이전 레벨의 이미지 크기에 맞춤
# gaussian_expanded = cv2.resize(gaussian_expanded, (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0]))
#
# laplacian = cv2.subtract(gaussian_pyramid[i-1], gaussian_expanded)
# laplacian_pyramid.append(laplacian)
#
# # 라플라시안 피라미드 단계별 이미지 표시
# plt.figure(figsize=(12, 6))
# for i in range(3):
#     plt.subplot(1, 3, i+1)
#     plt.imshow(cv2.cvtColor(gaussian_pyramid[i], cv2.COLOR_BGR2RGB))
#     plt.title(f'Laplacian Level {i}')
#     plt.axis("off")
#
# plt.show()

# # 피라미드를 이용한 이미지 혼합
#
# # 이미지 로드
# A = cv2.imread("orange2.png")
# B = cv2.imread("apple2.png")
#
# # 가우시안 피라미드 생성
# A_copy = A.copy()
# B_copy = B.copy()
# gpA = [A_copy]
# gpB = [B_copy]
#
# for i in range(3):
#     A_copy = cv2.pyrDown(A_copy)
#     B_copy = cv2.pyrDown(B_copy)
#     gpA.append(A_copy)
#     gpB.append(B_copy)
#
# # 라플라시안 피라미드 생성
# lpA = [gpA[3]]
# lpB = [gpB[3]]
#
# for i in range(3, 0, -1):
#     # cv2.pyrUp()로 확장된 이미지를 이전 단계와 같은 크기로 맞춤
#     gaussian_expanded_A = cv2.pyrUp(gpA[i])
#     gaussian_expanded_B = cv2.pyrUp(gpB[i])
#
#     # 크기를 이전 이미지와 동일하게 맞춤
#     gaussian_expanded_A = cv2.resize(gaussian_expanded_A, (gpA[i - 1].shape[1], gpA[i - 1].shape[0]))
#     gaussian_expanded_B = cv2.resize(gaussian_expanded_B, (gpB[i - 1].shape[1], gpB[i - 1].shape[0]))
#
#     LA = cv2.subtract(gpA[i - 1], gaussian_expanded_A)
#     LB = cv2.subtract(gpB[i - 1], gaussian_expanded_B)
#     lpA.append(LA)
#     lpB.append(LB)
#
# # 피라미드 블랜딩
# LS = []
# for la, lb in zip(lpA, lpB):
#     # 두 배열의 높이가 동일한지 확인하고, 동일하지 않으면 크기를 조정
#     if la.shape[0] != lb.shape[0]:
#         lb = cv2.resize(lb, (lb.shape[1], la.shape[0]))  # lb 이미지를 la의 높이에 맞춤
#
#     rows, cols, dpt = la.shape
#     ls = np.hstack((la[:, 0:cols//2], lb[:, cols//2:]))
#     LS.append(ls)
#
# # 블렌딩된 이미지 재구성
# blended = LS[0]
# for i in range(1, 4):
#     # 크기를 맞추어 피라미드를 재구성
#     blended_up = cv2.pyrUp(blended)
#
#     # cv2.pyrUp(blended)의 크기와 LS[i]의 크기를 일치시킴
#     if blended_up.shape != LS[i].shape:
#         LS[i] = cv2.resize(LS[i], (blended_up.shape[1], blended_up.shape[0]))
#
#     # 두 이미지의 채널 수 확인 및 일치
#     if blended_up.shape[2] != LS[i].shape[2]:
#         LS[i] = cv2.cvtColor(LS[i], cv2.COLOR_GRAY2BGR)
#
#     # 두 이미지를 더하기
#     blended = cv2.add(blended_up, LS[i])
#
# # 결과 표시
# cv2.imshow("Blended Image", blended)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 템플릿 매칭
#
# # 이미지와 탬플릿 로드
# img = cv2.imread("mainImage.png")
# template = cv2.imread("tiger.png", cv2.IMREAD_GRAYSCALE)
#
# # 그레이스케일로 전환
# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # 템플릿 매칭 수행
# result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
#
# # 최적의 매칭 위치 찾기
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#
# # 템플릿의 크기 구하기
# h, w = template.shape
#
# # 최적의 위치에 사각형 그리기
# top_left = max_loc
# bottom_right = (top_left[0] + w, top_left[1] + h)
# cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
#
# # 결과 표시
# plt.figure(figsize=(8, 6))
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.title("템플릿 매칭 결과")
# plt.axis("off")
# plt.show()

# 템플릿 매칭

## 이미지와 템플릿 로드
# img = cv2.imread("oranges.png")
# template = cv2.imread("orange2.png", cv2.IMREAD_GRAYSCALE)
# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # 템플릿 매칭 수행
# result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
#
# # 일치 임계값 설정
# threshold = 0.8
# loc = np.where(result >= threshold)
#
# # 일치하는 모든 위치에 사각형 그리기
# h, w = template.shape
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
#
# # 결과 표시
# cv2.imshow("MUltiple Matches", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 이미지 분할
#
# # 이미지 로드 및 그레이스케일로 전환
# img = cv2.imread("testImage.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # 이진환 처리
# _, binary = cv2.threshold(gray, 0, 255,
#                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#
# # 모폴로지 커널 설정
# kernel = np.ones((3, 3), np.uint8)
#
# # 모폴로지 연산 (Opening) 적용
# opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
#
# # 확실한 배경 영역 찾기
# sure_bg = cv2.dilate(opening, kernel, iterations=3)
#
# # 거리 변환 후 확실한 전경 영역 찾기
# dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
# _, sure_fg = cv2.threshold(dist_transform,
#                            0.7 * dist_transform.max(), 255, 0)
#
# # 확실한 배경에서 확실한 전경을 뺀 영역 찾기 (Unknown)
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg, sure_fg)
#
# # 마커 레이블 생성 및 워터셰드 알고리즘 적용
# _, markers = cv2.connectedComponents(sure_fg)
# markers = markers + 1
# markers[markers + 1]
# markers[unknown == 255] = 0
# markers = cv2.watershed(img, markers)
# img[markers == -1] = [255, 0, 0]
#
# # 결과 이미지 출력
# cv2.imshow('Segmented Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# GrabCut 적용

# 이미지 로드
img = cv2.imread("../image/mainImage.png")
mask = np.zeros(img.shape[:2], np.uint8)

# 사전 정의된 배경 및 전경 모델 생성
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# 초기 사각형 설정
rect = (50, 50, 450, 290)

# GrabCut 알고리즘 적용
cv2.grabCut(img,mask, rect, bgd_model,
            fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# 마스크 업데이트 및 결과 이미지 생성
mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
result = img * mask2[:, :, np.newaxis]

# 결과 표시
cv2.imshow("GrabCut 결과", result)
cv2.waitKey(0)
cv2.destroyAllWindows()