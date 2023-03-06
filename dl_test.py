import cv2
import mediapipe as mp  # pose estimation libraries
import numpy as np

mp_drawing = mp.solutions.drawing_utils     # visualizing pose
mp_pose = mp.solutions.pose     # pose estimation model 

cap = cv2.VideoCapture(0)    # 실시간 카메라, 웹캠 넘버? 비디오 자료라고 함.

# Curl counter variables
counter = 0
stage = None

# Set Up mediapipe instance
# 카메라창 계속 떠있게 만드는 코드인 듯. 정확성을 더 높이고 싶으면 더 올리기. 너무 떨어지는게 아니면 0.5가 괜찮은듯
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Detect stuff and render
        # Recolor image. mediapipe에 사용할 수 있게 convert
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection. pose는 위에서 만든 Model 참조
        results = pose.process(image)   # detection 결과 저장

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            #print(landmarks[mp_pose.PoseLandmark.NOSE.value].x)   # nose 움직임 랜드마크의 x좌표만 출력. 다른 y, z 좌표나 visibility도 가능
            
            # Get coordinates 
            # 팔 굽히는 예제
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
       
            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Visualize angle : render angle to Actual Screen
            cv2.putText(image, str(angle),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)   # 글씨체, 크기, 색깔, 선 굵기, 선 종류(?)
            
            # Curl counter Logic
            if angle > 160: 
                stage = "down"
            if angle < 30 and stage == "down":
                stage = "up"
                counter += 1
                print(counter)

        except:
            pass 

        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

        # Rep data
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Rep data
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'STAGE', (75, 12),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (60, 60),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
         

        # Render detections
        # 가공된 실시간 영상 = image, 점 = pose_landmarks, 라인 = POSE_CONNECTIONS, 뒤 요소들은 색깔 변화를 위해. 없애면 default 색깔임
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color = (245, 117, 66), thickness = 2, circle_radius = 2), # 점 색깔
                                  mp_drawing.DrawingSpec(color = (245, 66, 230), thickness = 2, circle_radius = 2)) # 라인 색깔

        cv2.imshow('Mediapipe Feed', image)


        def calculate_angle(a,b,c):
            a = np.array(a) # First 
            b = np.array(b) # Mid 
            c = np.array(c) # End 

            # 0 : x, 1 : y 끼리 빼기
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi) # To absolute values

            if angle > 180.0:   
                angle = 360 - angle
            return angle
        

        # q 누르면 종료됨
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()