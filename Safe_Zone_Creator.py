import cv2
import mediapipe as mp
import numpy as np
import time
import os
import json

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ============================
# Safe zones (polygon-only UI)
# ============================

EXCLUSION_ZONES = []  # mix: rectangles [x1,y1,x2,y2] or polygons [[x,y], ...]
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
                        # Rectangle: [x1, y1, x2, y2]
                        if (
                            isinstance(z, (list, tuple)) and len(z) == 4 and
                            all(isinstance(v, (int, float)) for v in z)
                        ):
                            x1, y1, x2, y2 = z
                            # clamp + normalize ordering
                            x1, y1 = float(max(0.0, min(1.0, x1))), float(max(0.0, min(1.0, y1)))
                            x2, y2 = float(max(0.0, min(1.0, x2))), float(max(0.0, min(1.0, y2)))
                            x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
                            y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
                            valid.append([x_min, y_min, x_max, y_max])
                        # Polygon: [[x,y], ...]
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

def save_safe_zones():
    try:
        with open(SAFE_ZONES_CONFIG, 'w', encoding='utf-8') as f:
            json.dump(EXCLUSION_ZONES, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
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

# UI state for polygon-only editing
ui_edit_mode = False
ui_poly_points = []
ui_frame_size = (1000, 600)

def on_mouse(event, x, y, flags, param):
    global ui_edit_mode, ui_poly_points, ui_frame_size, EXCLUSION_ZONES
    if not ui_edit_mode:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        ui_poly_points.append((x, y))

def main():
    global ui_edit_mode, ui_poly_points
    load_safe_zones()
    cap = cv2.VideoCapture('fall_final.mp4')

    # minimal pose setup to render and align UI scale; detection not required for editing
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        cv2.namedWindow('Safe Zone Creator')
        cv2.setMouseCallback('Safe Zone Creator', on_mouse)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (ui_frame_size[0], ui_frame_size[1]))
            image = cv2.flip(frame, 1)

            # draw zones
            if EXCLUSION_ZONES:
                draw_zones(image, EXCLUSION_ZONES)

            # helper + polygon preview
            cv2.putText(image, 'EDIT: Z toggle  Click add points  F finish  X undo-pt  U undo-zone  C clear  S save  Q quit',
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2, cv2.LINE_AA)
            if ui_edit_mode and ui_poly_points:
                cv2.polylines(image, [np.array(ui_poly_points, dtype=np.int32)], isClosed=False, color=(0, 255, 0), thickness=2)
                for p in ui_poly_points:
                    cv2.circle(image, p, 3, (0, 255, 0), -1)

            cv2.imshow('Safe Zone Creator', image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('z'):
                ui_edit_mode = not ui_edit_mode
            elif key == ord('x'):
                if ui_poly_points:
                    ui_poly_points.pop()
            elif key == ord('f'):
                if len(ui_poly_points) >= 3:
                    w, h = ui_frame_size
                    norm = []
                    for (px, py) in ui_poly_points:
                        nx = max(0.0, min(1.0, px / max(1, w)))
                        ny = max(0.0, min(1.0, py / max(1, h)))
                        norm.append([nx, ny])
                    EXCLUSION_ZONES.append(norm)
                    save_safe_zones()
                    ui_poly_points = []
            elif key == ord('u'):
                if EXCLUSION_ZONES:
                    EXCLUSION_ZONES.pop()
                    save_safe_zones()
            elif key == ord('c'):
                EXCLUSION_ZONES.clear()
                save_safe_zones()
            elif key == ord('s'):
                save_safe_zones()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
