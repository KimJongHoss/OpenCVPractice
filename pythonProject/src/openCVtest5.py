from hashlib import algorithms_available
from pydoc import describe
from tempfile import template

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# # Haar Cascades
#
# # 이미지 로드
# img = cv2.imread("winter.png")
#
# # 그레이스케일로 전환
# gray_image = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
#
# # Haar cascade 파일 경로
# cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#
# # 얼굴 인식기 초기화
# face_cascade = cv2.CascadeClassifier(cascade_path)
#
# # 얼굴 검출
# faces = face_cascade.detectMultiScale(gray_image,
#                                       scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
# # 검출된 얼굴에 사각형 그리기
# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
#
# #  결과 표시
# cv2.imshow("Detected Faces", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #비디오 캡쳐
#
# cap = cv2.VideoCapture(0)
#
# while True:
# # 프레임 읽기
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # 그레이스케일로 전환
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Haar cascade 파일 경로
#     cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#
#     # 얼굴 인식기 초기화
#     face_cascade = cv2.CascadeClassifier(cascade_path)
#
#     # 얼굴 검출
#     faces = face_cascade.detectMultiScale(gray_frame,
#                                           scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#     #검출된 얼굴에 사각형 그리기
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
#
#     # 결과 출력
#     cv2.imshow("video - Face Detection", frame)
#
#     # 'q' 키를 누르면 종료
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# #캡션 종료 및 윈도우 닫기
# cap.release()
# cv2.destroyAllWindows()

# # SIFT 객체 생성
#
# img = cv2.imread("testRamen.jpg")
#
# sift = cv2.SIFT_create()
#
# # 키포인트 및 디스크립터 검출
# keypoints, descriptors = sift.detectAndCompute(img, None)

# # ORB 객체 생성
#
# img = cv2.imread("testRamen.jpg")
#
# orb = cv2.ORB_create()
#
# # 키포인트 및 디스크립터 검출
# keypoints, descriptors = orb.detectAndCompute(img, None)

# # Brute-Force 매칭
#
# # 이미지 로드
# img1 = cv2.imread('testRobot.jpg', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('testRamen.jpg', cv2.IMREAD_GRAYSCALE)
#
# # SIFT 검출기 생성
# sift = cv2.SIFT_create()
#
# # 특징점과 디스크립터 계산
# keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
# keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
#
# # Brute-Force 매처 생성 및 매칭 수행
# bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
# matches = bf.match(descriptors1, descriptors2)
#
# # 매칭 결과 정렬 (매칭 거리 기준)
# matches = sorted(matches, key=lambda x: x.distance)
#
# # 매칭 결과 시각화
# img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#
# # 결과 이미지 표시
# cv2.imshow('Matches', img_matches)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# FLANN

# 이미지 로드
img1 = cv2.imread('../image/orange2.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('../image/apple2.png', cv2.IMREAD_GRAYSCALE)

# SIFT 검출기 생성
sift = cv2.SIFT_create()

# 특징점과 디스크립터 계산
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# FLANN 매처 생성 및 매칭 수행
index_params = dict(algorithm=1, trees=5)  # 'algorithms'를 'algorithm'으로 수정
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# k-NN 매칭 수행
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# 매칭 결과 필터링 (Lowe's ratio test)
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 매칭 결과 시각화
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 결과 이미지 표시
cv2.imshow('Good Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()