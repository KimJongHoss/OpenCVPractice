import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 비디오 파일 경로
video_path = "C:/Users/jongho/PycharmProjects/pythonProject/video/input.mp4"

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 비디오가 열렸는지 확인
if not cap.isOpened():
    print("비디오 파일을 열 수 없습니다.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 비디오의 끝에 도달했을 때

        # 프레임을 화면에 표시
        cv2.imshow('Video', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# 자원 해제
cap.release()
cv2.destroyAllWindows()