import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# # 비디오 파일 경로
# video_path = "C:/Users/jongho/PycharmProjects/pythonProject/video/input.mp4"
#
# # 비디오 캡처 객체 생성
# cap = cv2.VideoCapture(video_path)
#
# # 비디오가 열렸는지 확인
# if not cap.isOpened():
#     print("비디오 파일을 열 수 없습니다.")
# else:
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break  # 비디오의 끝에 도달했을 때
#
#         # 프레임을 화면에 표시
#         cv2.imshow('Video', frame)
#
#         # 'q' 키를 누르면 종료
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#
# # 자원 해제
# cap.release()
# cv2.destroyAllWindows()

# # 체스보드 패턴 크기
# rows, cols = 6, 9  # 코너 개수, (6x9 내부 코너를 생성)
# square_size = 1    # 각 사각형의 크기 (단위는 마음대로)
#
# # 체스보드 패턴 생성
# pattern = np.zeros((rows, cols))
#
# # 흑백 패턴 그리기
# for i in range(rows):
#     for j in range(cols):
#         if (i + j) % 2 == 0:
#             pattern[i, j] = 1  # 흰색(1), 검정색(0) 패턴 생성
#
# # 이미지 크기 확대
# plt.figure(figsize=(cols, rows))
# plt.imshow(pattern, cmap='gray', extent=(0, cols * square_size, 0, rows * square_size))
# plt.axis('off')  # 축 제거
#
# # 이미지 저장
# plt.savefig('../image/generated_chessboard.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.show()

# # 체스보드 패턴 크기
# rows, cols = 6, 9  # 코너 개수, (6x9 내부 코너를 생성)
# square_size = 2    # 각 사각형의 크기 (기본 크기보다 2배 큼)
#
# # 체스보드 패턴 생성
# pattern = np.zeros((rows, cols))
#
# # 흑백 패턴 그리기
# for i in range(rows):
#     for j in range(cols):
#         if (i + j) % 2 == 0:
#             pattern[i, j] = 1  # 흰색(1), 검정색(0) 패턴 생성
#
# # 이미지 크기 확대
# plt.figure(figsize=(cols, rows))
# plt.imshow(pattern, cmap='gray', extent=(0, cols * square_size, 0, rows * square_size))
# plt.axis('off')  # 축 제거
#
# # 이미지 저장
# plt.savefig('../image/generated_chessboard_9x6_large.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.show()

# # 체스보드 패턴 크기
# rows, cols = 6, 9  # 코너 개수, (6x9 내부 코너를 생성)
# square_size = 1    # 각 사각형의 크기 (단위는 마음대로)
#
# # 체스보드 패턴 생성
# pattern = np.zeros((rows, cols))
#
# # 흑백 패턴 그리기
# for i in range(rows):
#     for j in range(cols):
#         if (i + j) % 2 == 0:
#             pattern[i, j] = 1  # 흰색(1), 검정색(0) 패턴 생성
#
# # 이미지 크기 확대 및 회전
# plt.figure(figsize=(cols, rows))
# plt.imshow(np.rot90(pattern), cmap='gray', extent=(0, cols * square_size, 0, rows * square_size))
# plt.axis('off')  # 축 제거
#
# # 이미지 저장
# plt.savefig('../image/generated_chessboard_9x6_rotated.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.show()

# # 체스보드 패턴 크기
# rows, cols = 6, 9  # 코너 개수, (6x9 내부 코너를 생성)
# square_size = 1    # 각 사각형의 크기 (단위는 마음대로)
#
# # 컬러 체스보드 패턴 생성
# pattern = np.zeros((rows, cols, 3))
#
# # 흑백 패턴 그리기
# for i in range(rows):
#     for j in range(cols):
#         if (i + j) % 2 == 0:
#             pattern[i, j] = [1, 1, 1]  # 흰색(1, 1, 1), 검정색(0, 0, 0) 패턴 생성
#         else:
#             pattern[i, j] = [0.2, 0.6, 1]  # 청색 계열의 색상 추가
#
# # 이미지 크기 확대
# plt.figure(figsize=(cols, rows))
# plt.imshow(pattern, extent=(0, cols * square_size, 0, rows * square_size))
# plt.axis('off')  # 축 제거
#
# # 이미지 저장
# plt.savefig('../image/generated_chessboard_9x6_color.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.show()

# # 체크보드 패턴의 행, 열 개수
# chessboard_size = (9, 6)
#
# # 실세계에서의 각 코너의 3D 좌표 설정(Z=0)
# objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
# objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
#
# # 3D 좌표와 이미지 좌표를 저장할 배열
# objpoints = []  # 실세계 좌표(3D)
# imgpoints = []  # 이미지 좌표 (2D)
#
# # 예제 이미지들 로드
# images = ["../image/pattern.png"]
#
# for image_file in images:
#     img = cv2.imread(image_file)
#     if img is None:
#         print(f"이미지 파일 {image_file}을(를) 열 수 없습니다.")
#         continue
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # 체스보드 코너 찾기
#     ret, corners = cv2.findChessboardCorners(gray, chessboard_size, cv2.CALIB_CB_FAST_CHECK)
#
#     if ret:
#         objpoints.append(objp)
#         imgpoints.append(corners)
#
#         # 코너 그리기
#         cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
#         cv2.imshow('Chessboard Corners', img)
#         cv2.waitKey(500)  # 500ms 동안 표시
#     else:
#         print(f"체스보드 코너를 찾을 수 없습니다. 이미지: {image_file}")
#
# # 카메라 보정 수행 (여기서는 여러 이미지에서 수집한 코너 점을 이용)
# if len(objpoints) > 0:
#     ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#
#     # 카메라 보정 정보 출력
#     print("Camera Matrix:\n", camera_matrix)
#     print("Distortion Coefficients:\n", dist_coeffs)
#
#     # 왜곡 보정
#     img = cv2.imread("../image/testImage.jpg")
#     h, w = img.shape[:2]
#     new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
#
#     # 보정된 이미지 얻기
#     dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
#
#     # 보정 결과 시각화
#     cv2.imshow("Undistorted", dst)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("체스보드 코너를 찾을 수 없습니다. 이미지가 제대로 로드되었는지 확인하세요.")

# # 광류
#
# # 비디오 파일 경로
# video_path = "C:/Users/jongho/PycharmProjects/pythonProject/video/input.mp4"
#
# # 비디오 캡처 객체 생성
# cap = cv2.VideoCapture(video_path)
#
# # 첫 번째 프레임 읽기 및 그레이스케일로 변환
# ret, first_frame = cap.read()
# prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
#
# # HSV 색상 공간으로 초기화
# hsv = np.zeros_like(first_frame)
# hsv[..., 1] = 255
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # 현재 프레임을 그레이스케일로 변환
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Farneback 알고리즘을 사용하여 광류 계산
#     flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
#                                         None, 0.5, 3, 15, 3, 5, 1.2, 0)
#
#     # 광류의 크기와 방향 계산
#     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#
#     # 각도를 색상으로 매핑
#     hsv[..., 0] = ang * 180 / np.pi / 2
#     hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#
#     # HSV에서 BGR로 변환
#     bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#
#     # 결과 출력
#     cv2.imshow("Dense Optical Flow", bgr)
#
#     # 'q' 키를 누르면 종료
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#     # 현재 프레임을 이전 프레임으로 설정
#     prev_gray = gray
#
# cap.release()
# cv2.destroyAllWindows()

# Sparse Optical Flow

# # 비디오 파일 경로
# video_path = "C:/Users/jongho/PycharmProjects/pythonProject/video/input.mp4"
#
# # 비디오 캡처 객체 생성
# cap = cv2.VideoCapture(video_path)
#
# # 첫 번째 프레임 읽기 및 그레이스케일로 변환
# ret, first_frame = cap.read()
# prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
#
# # 특징점 검출기 생성 (Good Features to Track)
# feature_params = dict(maxCorners=100, qualityLevel=0.3,
#                       minDistance=7, blockSize=7)
# prev_points = cv2.goodFeaturesToTrack(prev_gray,
#                                       mask=None, **feature_params)
#
# # LK 파라미터 설정
# lk_params = dict(winSize=(15, 15), maxLevel=2,
#                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # 현재 프레임을 그레이스케일로 변환
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Lucas-kanade 알고리즘을 사용하여 Optical Flow 계산
#     next_points, status, err = (
#         cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params))
#
#     # 상태가 1인 점들만 선택
#     good_new = next_points[status == 1]
#     good_old = prev_points[status == 1]
#
#     # 점들 사이에 선 그리기
#     for i, (new, old) in enumerate(zip(good_new, good_old)):
#         a, b = new.ravel()
#         c, d = old.ravel()
#         frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
#         frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
#
#     # 결과 출력
#     cv2.imshow("Sparse Optical Flow", frame)
#
#     # 'q' 키를 누르면 종료
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#     # 다음 계산을 위해 현재 점들 업데이트
#     prev_gray = gray.copy()
#     prev_points = good_new.reshape(-1, 1, 2)
#
# cap.release()
# cv2.destroyAllWindows()

# Optical Flow

# 비디오 파일 경로
video_path = "C:/Users/jongho/PycharmProjects/pythonProject/video/input.mp4"

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 첫 번째 프레임 읽기 및 그레이스케일로 변환
ret, first_frame = cap.read()
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# 추적할 좋은 특징점 찾기 (코너)
feature_params = dict(maxCorners=100, qualityLevel=0.3,
                      minDistance=7, blockSize=7)
prev_points = cv2.goodFeaturesToTrack(prev_gray,
                                      mask=None, **feature_params)

# Lucas-Kanade Optical Flow 파라미터 설정
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 색상 설정
color = np.random.randint(0, 255, (100, 3))

# 추적 경로를 그리기 위한 빈 이미지 생성
mask = np.zeros_like(first_frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 현재 프레임을 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 이전 프레임과 현재 프레임 간의 Optical Flow 계산
    next_points, status, err = (
        cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params))

    # 상태가 1인(추적 성공한) 점들만 선택
    good_new = next_points[status == 1]
    good_old = prev_points[status == 1]

    # 추적 경로 그리기
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

    # 현재 프레임과 마스크를 결합
    img = cv2.add(frame, mask)

    # 결과 출력
    cv2.imshow("Sparse Optical Flow", img)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 다음 계산을 위해 현재 점들 업데이트
    prev_gray = gray.copy()
    prev_points = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()