# ONLY mediapipe 
# 낙상 지속 시간(초) 카운터 기능 추가
# 블랙박스 기능 추가 – 낙상 감지 시 영상 자동 녹화 및 저장
# 안전 구역 기능 추가 – 소파나 침대 등 낙상으로 인식하지 않을 영역 Safe_Zone_Creator.py 로 설정 가능

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import json
from collections import deque
from datetime import datetime

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """
    세 점 a, b, c가 이루는 각도(도)를 계산 (b가 꼭지점)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def is_fall_condition_met(keypoints):
    """
    몸통(어깨-엉덩이) 기울기 기반으로 넘어짐(누움) 상태 판별.

    - 다리 들어올림 동작에서도 몸통은 수직에 가까우므로 오탐을 줄임
    - 몸통이 수평에 가까울수록(수직과의 각도 ↑) 넘어짐으로 간주
    """
    try:
        left_shoulder = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = keypoints[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # 가시성 체크
        if (
            left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5 or
            left_hip.visibility < 0.5 or right_hip.visibility < 0.5
        ):
            return False

        # 어깨/엉덩이 중앙점
        shoulder = np.array([
            (left_shoulder.x + right_shoulder.x) / 2.0,
            (left_shoulder.y + right_shoulder.y) / 2.0,
        ])
        hip = np.array([
            (left_hip.x + right_hip.x) / 2.0,
            (left_hip.y + right_hip.y) / 2.0,
        ])

        torso_vec = hip - shoulder
        torso_len = np.linalg.norm(torso_vec)
        if torso_len < 1e-6:
            return False

        # 수직 벡터 (영상 좌표계에서 아래쪽이 +y)
        vertical = np.array([0.0, 1.0])
        cos_tilt = np.dot(torso_vec, vertical) / (torso_len * np.linalg.norm(vertical))
        cos_tilt = np.clip(cos_tilt, -1.0, 1.0)
        tilt_deg = np.degrees(np.arccos(cos_tilt))

        # 55도 이상이면 몸통이 수평에 가까움 -> 넘어짐으로 판정
        return tilt_deg > 55.0
    except Exception:
        return False


# ============================
# Safe zone (load-only) + blackbox helpers
# ============================

EXCLUSION_ZONES = []  # rectangles [x1,y1,x2,y2] or polygons [[x,y], ...] in normalized coords on flipped image
SAFE_ZONES_CONFIG = 'safe_zones.json'

def load_safe_zones():
    global EXCLUSION_ZONES
    try:
        if os.path.isfile(SAFE_ZONES_CONFIG):
            with open(SAFE_ZONES_CONFIG, 'r', encoding='utf-8') as f:
                zones = json.load(f)
                if isinstance(zones, list):
                    valid = []
                    for z in zones:
                        # Rectangle
                        if (
                            isinstance(z, (list, tuple)) and len(z) == 4 and
                            all(isinstance(v, (int, float)) for v in z)
                        ):
                            x1, y1, x2, y2 = z
                            x1, y1 = float(max(0.0, min(1.0, x1))), float(max(0.0, min(1.0, y1)))
                            x2, y2 = float(max(0.0, min(1.0, x2))), float(max(0.0, min(1.0, y2)))
                            x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
                            y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
                            valid.append([x_min, y_min, x_max, y_max])
                        # Polygon
                        elif (
                            isinstance(z, (list, tuple)) and len(z) >= 3 and
                            all(isinstance(p, (list, tuple)) and len(p) == 2 for p in z)
                        ):
                            norm_poly = []
                            for pt in z:
                                px, py = pt
                                if not (isinstance(px, (int, float)) and isinstance(py, (int, float))):
                                    norm_poly = []
                                    break
                                nx = float(max(0.0, min(1.0, px)))
                                ny = float(max(0.0, min(1.0, py)))
                                norm_poly.append([nx, ny])
                            if len(norm_poly) >= 3:
                                valid.append(norm_poly)
                    EXCLUSION_ZONES = valid
    except Exception:
        pass

def point_in_polygon(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / ((y2 - y1) if (y2 - y1) != 0 else 1e-9) + x1
        )
        if intersects:
            inside = not inside
    return inside

def point_in_zones(nx, ny, zones):
    for z in zones:
        if isinstance(z, (list, tuple)) and len(z) == 4 and all(isinstance(v, (int, float)) for v in z):
            x1, y1, x2, y2 = z
            if x1 <= nx <= x2 and y1 <= ny <= y2:
                return True
        elif isinstance(z, (list, tuple)) and len(z) >= 3 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in z):
            if point_in_polygon((nx, ny), z):
                return True
    return False

def draw_zones(image, zones, color=(0, 255, 0)):
    h, w = image.shape[:2]
    for z in zones:
        if isinstance(z, (list, tuple)) and len(z) == 4 and all(isinstance(v, (int, float)) for v in z):
            x1, y1, x2, y2 = z
            p1 = (int(x1 * w), int(y1 * h))
            p2 = (int(x2 * w), int(y2 * h))
            overlay = image.copy()
            cv2.rectangle(overlay, p1, p2, color, -1)
            image[:] = cv2.addWeighted(overlay, 0.15, image, 0.85, 0)
            cv2.rectangle(image, p1, p2, color, 2)
            cv2.putText(image, 'SAFE ZONE', (p1[0] + 6, p1[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        elif isinstance(z, (list, tuple)) and len(z) >= 3 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in z):
            pts = np.array([[int(px * w), int(py * h)] for (px, py) in z], dtype=np.int32)
            overlay = image.copy()
            cv2.fillPoly(overlay, [pts], color)
            image[:] = cv2.addWeighted(overlay, 0.15, image, 0.85, 0)
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
            p1 = (pts[0][0], pts[0][1])
            cv2.putText(image, 'SAFE ZONE', (p1[0] + 6, p1[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def get_person_center_normalized(results):
    try:
        if not results or not results.pose_landmarks:
            return None
        ls = results.pose_landmarks.landmark
        xs, ys = [], []
        for lm in ls:
            if hasattr(lm, 'visibility') and lm.visibility is not None and lm.visibility >= 0.5:
                xs.append(lm.x)
                ys.append(lm.y)
        if not xs or not ys:
            return None
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
    except Exception:
        return None

class FrameBuffer:
    def __init__(self, keep_seconds=80.0, target_fps=10, jpeg_quality=80):
        self.keep_seconds = float(keep_seconds)
        self.target_fps = float(target_fps)
        self.jpeg_quality = int(jpeg_quality)
        self.buf = deque()
        self._last_push_time = 0.0

    def _should_sample(self, t):
        if self._last_push_time == 0.0:
            return True
        return (t - self._last_push_time) >= (1.0 / max(1.0, self.target_fps))

    def append(self, frame_bgr, t):
        try:
            if not self._should_sample(t):
                self._evict(t)
                return
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            ok, enc = cv2.imencode('.jpg', frame_bgr, encode_param)
            if ok:
                self.buf.append((t, enc))
                self._last_push_time = t
            self._evict(t)
        except Exception:
            pass

    def _evict(self, now_t):
        cutoff = now_t - self.keep_seconds
        while self.buf and self.buf[0][0] < cutoff:
            self.buf.popleft()

    def slice_window(self, start_t, end_t):
        frames = []
        for (t, enc) in self.buf:
            if start_t <= t <= end_t:
                frames.append((t, enc))
        return frames

def save_incident_clip(frames_enc_list, out_path, frame_size, fps=10):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(out_path, fourcc, float(fps), frame_size)
        for (_, enc) in frames_enc_list:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if img is None:
                continue
            if (img.shape[1], img.shape[0]) != frame_size:
                img = cv2.resize(img, frame_size)
            vw.write(img)
        vw.release()
        return True
    except Exception:
        return False

PRE_SECONDS = 30.0
POST_SECONDS = 30.0
BUFFER_FPS = 10
JPEG_QUALITY = 80
REPORT_DIR = 'reports'

load_safe_zones()
cap = cv2.VideoCapture('fall_final.mp4')

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # 지속 상태 추적 변수
    fall_candidate_since = None   # 넘어짐 조건이 연속 유지되기 시작한 시각
    fall_state = False            # 넘어짐 상태 확정 여부
    fall_state_since = None       # 넘어짐 상태 확정 시각
    required_duration = 0.2       # 짧은 디바운스(초)
    # 깜빡임(플래싱) 효과 변수
    flashing = False
    mask_visible = False
    last_flash_time = 0.0
    flash_interval = 0.5          # 초 단위 간격

    # tracking lost 후 돌아올 때 복원 위한 상태 변수
    RESUME_GRACE_SEC = 5.0
    paused_elapsed = 0.0
    last_tracking_lost_time = None
    was_in_fall_when_lost = False
    # classification flip 이후 복원 위한 상태 변수
    last_fall_ended_time = None
    was_in_fall_when_ended = False
    paused_elapsed_end = 0.0

    # rolling buffer for blackbox (-30s/+30s)
    frame_buffer = FrameBuffer(keep_seconds=PRE_SECONDS + POST_SECONDS + 20, target_fps=10, jpeg_quality=80)

    # incident state
    incident_pending = False
    incident_event_time = None
    incident_saving = False
    last_saved_time = None
    incident_done = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("비디오를 불러오지 못했습니다.")
            break

        frame = cv2.resize(frame, (1000, 600))

        # 좌우 반전 (캠 쓸때 거울 효과용) 및 RGB 변환
        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        current_time = time.time()

        if results.pose_landmarks:
            keypoints = results.pose_landmarks.landmark

            fall_condition = is_fall_condition_met(keypoints)
            # safe zone exclusion using person center
            center = get_person_center_normalized(results)
            if center is not None and EXCLUSION_ZONES:
                if point_in_zones(center[0], center[1], EXCLUSION_ZONES):
                    fall_condition = False
            current_time = time.time()

            # 디바운스 및 상태 확정/유지
            # 낙상 중 추적이 끊기더라도 일정 허용 범위 (현재 5초) 내에서 추적 복원 -> 낙상 시간 유지
            if (not fall_state) and was_in_fall_when_lost and last_tracking_lost_time is not None and (current_time - last_tracking_lost_time) <= RESUME_GRACE_SEC:
                if fall_condition:
                    fall_state = True
                    fall_state_since = current_time - paused_elapsed
                    was_in_fall_when_lost = False
                    last_tracking_lost_time = None
                    paused_elapsed = 0.0
                else:
                    if (current_time - last_tracking_lost_time) > RESUME_GRACE_SEC:
                        was_in_fall_when_lost = False
                        last_tracking_lost_time = None
                        paused_elapsed = 0.0

            # normal state update
            if fall_state:
                if not fall_condition:
                    # classification-based exit 에서 마지막 상태 기록
                    if fall_state_since is not None:
                        paused_elapsed_end = max(0.0, current_time - fall_state_since)
                    else:
                        paused_elapsed_end = 0.0
                    last_fall_ended_time = current_time
                    was_in_fall_when_ended = True
                    fall_candidate_since = None
                    fall_state = False
                    fall_state_since = None
            else:
                if fall_condition:
                    if fall_candidate_since is None:
                        fall_candidate_since = current_time
                    if (current_time - fall_candidate_since) >= required_duration:
                        fall_state = True
                        fall_state_since = fall_candidate_since
                        # new event: arm incident at timer 0s
                        incident_pending = True
                        incident_event_time = fall_state_since
                        incident_done = False
                else:
                    fall_candidate_since = None

            # 낙상 중 잠시 자세가 바뀌거나 분류가 뒤집혀도 (classification flip) 일정 허용 범위 (현재 5초) 내에서 낙상 시간 유지
            if (not fall_state) and was_in_fall_when_ended and last_fall_ended_time is not None and (current_time - last_fall_ended_time) <= RESUME_GRACE_SEC:
                if fall_condition:
                    fall_state = True
                    fall_state_since = current_time - paused_elapsed_end
                    was_in_fall_when_ended = False
                    last_fall_ended_time = None
                    paused_elapsed_end = 0.0
                else:
                    if (current_time - last_fall_ended_time) > RESUME_GRACE_SEC:
                        was_in_fall_when_ended = False
                        last_fall_ended_time = None
                        paused_elapsed_end = 0.0

            # 상태 텍스트 + 플래싱 + 경과시간 표시 (지속 모니터링)
            if fall_state and fall_state_since is not None:
                # 플래싱 토글 업데이트
                flashing = True
                if current_time - last_flash_time >= flash_interval:
                    mask_visible = not mask_visible
                    last_flash_time = current_time

                # 마스크 적용 (반투명 붉은색)
                if mask_visible:
                    overlay = image.copy()
                    red_color = (0, 0, 255)  # BGR
                    alpha = 0.35
                    cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), red_color, -1)
                    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

                # 경과 시간(초) 계산 및 표시
                elapsed = max(0.0, current_time - fall_state_since)
                seconds_str = f"{elapsed:0.1f}s"

                # 상태 텍스트
                if elapsed >= 30.0:
                    status_text = 'EMERGENCY'
                else:
                    status_text = 'FALL DETECTION'

                cv2.putText(
                    image, status_text, (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA
                )
                cv2.putText(
                    image, seconds_str, (50, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA
                )
            else:
                # 상태 해제 시 플래싱도 리셋
                flashing = False
                mask_visible = False

            # 랜드마크 시각화
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 화면 출력
        if not results.pose_landmarks:
            # tracking lost
            if fall_state and fall_state_since is not None:
                paused_elapsed = max(0.0, current_time - fall_state_since)
                last_tracking_lost_time = current_time
                was_in_fall_when_lost = True
                fall_candidate_since = None
                fall_state = False
                fall_state_since = None
            else:
                if last_tracking_lost_time is not None and (current_time - last_tracking_lost_time) > RESUME_GRACE_SEC:
                    was_in_fall_when_lost = False
                    last_tracking_lost_time = None
                    paused_elapsed = 0.0
            # 추적 끊길 시에 분류 복원 관련 변수 (flags) 초기화
            was_in_fall_when_ended = False
            last_fall_ended_time = None
            paused_elapsed_end = 0.0
            flashing = False
            mask_visible = False

        # draw zones (view-only)
        if EXCLUSION_ZONES:
            draw_zones(image, EXCLUSION_ZONES)

        # push to buffer and handle incident saving
        frame_buffer.append(image, current_time)
        if incident_pending and (incident_event_time is not None) and ((current_time - incident_event_time) >= POST_SECONDS) and (not incident_saving):
            incident_saving = True
            start_t = incident_event_time - PRE_SECONDS
            end_t = incident_event_time + POST_SECONDS
            frames_window = frame_buffer.slice_window(start_t, end_t)
            if frames_window:
                ts_str = datetime.fromtimestamp(incident_event_time).strftime('%Y%m%d_%H%M%S')
                out_path = os.path.join('reports', f'incident_{ts_str}.mp4')
                ok = save_incident_clip(frames_window, out_path, (1000, 600), fps=10)
            incident_pending = False
            incident_saving = False
            incident_done = True

        cv2.imshow('Fall Detection', image)

        # 종료 키 - waitKey 매개변수 값 줄이면 cpu usage 는 늘어나지만 영상처리는 빨라짐
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
