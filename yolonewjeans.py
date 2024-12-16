import cv2
import torch
import face_recognition
from yolov5 import YOLOv5
import numpy as np

# YOLO 모델 로드
yolo = YOLOv5("yolov5s.pt", device="cuda" if torch.cuda.is_available() else "cpu")

# 멤버 얼굴 데이터 준비
members = {
    "Hanni": ["/Users/baeyeongjae/Downloads/source/ch13/members/hanni.jpeg",
              "/Users/baeyeongjae/Downloads/source/ch13/members/hanni2.jpg",
              "/Users/baeyeongjae/Downloads/source/ch13/members/hanni3.jpg"],
    "Minji": ["/Users/baeyeongjae/Downloads/source/ch13/members/minji.jpeg",
              "/Users/baeyeongjae/Downloads/source/ch13/members/minji2.jpg",
              "/Users/baeyeongjae/Downloads/source/ch13/members/minji3.jpg"],
    "Haerin": ["/Users/baeyeongjae/Downloads/source/ch13/members/haerin.jpeg",
               "/Users/baeyeongjae/Downloads/source/ch13/members/haerin2.jpg",
               "/Users/baeyeongjae/Downloads/source/ch13/members/haerin3.jpg"],
    "Hyein": ["/Users/baeyeongjae/Downloads/source/ch13/members/hyein.jpeg",
              "/Users/baeyeongjae/Downloads/source/ch13/members/hyein2.jpg",
              "/Users/baeyeongjae/Downloads/source/ch13/members/hyein3.jpg"],
    "Danielle": ["/Users/baeyeongjae/Downloads/source/ch13/members/daniel.jpeg",
                 "/Users/baeyeongjae/Downloads/source/ch13/members/daniel2.jpg",
                 "/Users/baeyeongjae/Downloads/source/ch13/members/daniel3.jpg"],
}

# 멤버별 평균 얼굴 인코딩 계산
member_encodings = {}
for name, image_paths in members.items():
    encodings = []
    for path in image_paths:
        image = face_recognition.load_image_file(path)
        if image is not None:
            encoding = face_recognition.face_encodings(image)
            if encoding:
                encodings.append(encoding[0])
    if encodings:
        member_encodings[name] = np.mean(encodings, axis=0)  # 평균 얼굴 인코딩 계산

# 초기 설정
selected_member = None  # 현재 선택된 멤버
previous_member = None  # 이전 선택된 멤버

# 웹캠 또는 영상 입력
video_path = "/Users/baeyeongjae/Downloads/source/ch13/newjeans.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read the frame.")
        break

    # YOLO로 객체 탐지
    detections = yolo.predict(frame, size=640)

    det_boxes = []
    det_scores = []

    for det in detections.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        if cls == 0:  # 사람 클래스만 필터링
            width = x2 - x1
            height = y2 - y1
            det_boxes.append([x1, y1, width, height])  # [x, y, width, height]
            det_scores.append(float(conf))  # confidence score

    if len(det_boxes) == 0:
        continue

    for (x, y, w, h), score in zip(det_boxes, det_scores):
        if score >= 0.5:  # 신뢰도 기준 설정
            roi = frame[int(y):int(y + h), int(x):int(x + w)]

            # OpenCV에서 BGR을 RGB로 변환
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            # 얼굴 인식
            face_locations = face_recognition.face_locations(roi_rgb)

            if face_locations:
                for face_location in face_locations:
                    # 얼굴 인코딩
                    face_encoding = face_recognition.face_encodings(roi_rgb, [face_location])[0]

                    # 가장 유사한 멤버 얼굴을 찾기
                    distances = {name: face_recognition.face_distance([encoding], face_encoding)[0] for name, encoding in member_encodings.items()}
                    best_match = min(distances, key=distances.get)
                    label = best_match if distances[best_match] < 0.6 else "Unknown"

                    # 선택된 멤버일 경우 ROI 확대
                    if label == selected_member:
                        # 확대할 ROI 설정
                        focused_frame = roi.copy()
                        zoomed_frame = cv2.resize(focused_frame, (frame.shape[1], frame.shape[0]))

                        # 확대된 프레임으로 현재 프레임 대체
                        frame = zoomed_frame
            else:
                label = "Unknown"  # 얼굴 인식 실패 시

            # 사람의 위치를 박스로 표시하고, 이름을 텍스트로 추가
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 원본 프레임 또는 확대 프레임 표시
    cv2.imshow("Person Detection with Name", frame)

    # 키 입력 대기
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        selected_member = "Hanni" if selected_member != "Hanni" else None
    elif key == ord('2'):
        selected_member = "Minji" if selected_member != "Minji" else None
    elif key == ord('3'):
        selected_member = "Haerin" if selected_member != "Haerin" else None
    elif key == ord('4'):
        selected_member = "Hyein" if selected_member != "Hyein" else None
    elif key == ord('5'):
        selected_member = "Danielle" if selected_member != "Danielle" else None
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
