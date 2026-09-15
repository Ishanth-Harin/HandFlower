import cv2
import mediapipe as mp
import numpy as np
import random
import math

STEM_COLOR       = (255, 140, 40)    
STEM_COLOR_TIP   = (255, 210, 150)   
FLOWER_COLOR     = (60, 40, 230)    
FLOWER_CENTER    = (90, 160, 255)    
GROW_UI_COLOR    = (255, 170, 60)    
BLOOM_UI_COLOR   = (60, 40, 230)     
WHITE            = (255, 255, 255)

MAX_DEPTH = 6
TREE_X_FRAC = 0.72  # 0.5 = center, closer to 1.0 = further right

def build_tree(width, height, seed=1337, x_frac=0.5):
    rng = random.Random(seed)
    branches = []
    base_x = width * x_frac
    base_y = height * 0.92
    base_len = min(width, height) * 0.16

    def grow(x, y, angle, length, depth):
        x2 = x + math.cos(angle) * length
        y2 = y + math.sin(angle) * length
        start_frac = depth / MAX_DEPTH
        branches.append({
            "x1": x, "y1": y, "x2": x2, "y2": y2,
            "depth": depth,
            "is_leaf": depth == MAX_DEPTH - 1,
            "start_frac": start_frac,
        })
        if depth >= MAX_DEPTH - 1:
            return

        n_children = 3 if depth == 0 else (2 if rng.random() > 0.35 else 3)
        spread = 0.5 + rng.random() * 0.35
        for i in range(n_children):
            t = 0.5 if n_children == 1 else i / (n_children - 1)
            offset = (t - 0.5) * 2 * spread + (rng.random() - 0.5) * 0.18
            child_angle = angle + offset
            child_len = length * (0.68 + rng.random() * 0.12)
            grow(x2, y2, child_angle, child_len, depth + 1)

    grow(base_x, base_y, -math.pi / 2, base_len, 0)
    return branches


def smoothstep(edge0, edge1, x):
    if edge1 == edge0:
        return 0.0 if x < edge0 else 1.0
    t = (x - edge0) / (edge1 - edge0)
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def draw_flower(layer, x, y, angle, size, bloom):
    if bloom <= 0.01 or size <= 0:
        return
    petals = 6
    r_outer = size * (0.35 + bloom * 0.9)
    cx, cy = int(x), int(y)

    for i in range(petals):
        pa = angle + (i / petals) * 2 * math.pi
        px = cx + math.cos(pa) * r_outer
        py = cy + math.sin(pa) * r_outer
        axis = (max(2, int(r_outer * 0.42)), max(1, int(r_outer * 0.16)))
        center = (int(cx + (px - cx) * 0.55), int(cy + (py - cy) * 0.55))
        color = lerp_color((90, 60, 200), FLOWER_COLOR, bloom)
        cv2.ellipse(layer, center, axis, math.degrees(pa), 0, 360, color, -1, cv2.LINE_AA)

    core_r = max(2, int(size * 0.16 + bloom * 3))
    cv2.circle(layer, (cx, cy), core_r, FLOWER_CENTER, -1, cv2.LINE_AA)


def draw_tree(base_layer, glow_layer, branches, grow_disp, bloom_disp):
    grow_span = (1 / MAX_DEPTH) * 1.5

    for b in branches:
        end_frac = min(1.0, b["start_frac"] + grow_span)
        frac = smoothstep(b["start_frac"], end_frac, grow_disp)
        if frac <= 0:
            continue
        ex = b["x1"] + (b["x2"] - b["x1"]) * frac
        ey = b["y1"] + (b["y2"] - b["y1"]) * frac

        depth_t = b["depth"] / (MAX_DEPTH - 1)
        color = lerp_color(STEM_COLOR, STEM_COLOR_TIP, depth_t)
        thickness = max(1, int((MAX_DEPTH - b["depth"]) * 1.1))

        p1 = (int(b["x1"]), int(b["y1"]))
        p2 = (int(ex), int(ey))
        cv2.line(base_layer, p1, p2, color, thickness, cv2.LINE_AA)
        cv2.line(glow_layer, p1, p2, color, thickness + 3, cv2.LINE_AA)

        if b["is_leaf"] and frac > 0.7:
            angle = math.atan2(b["y2"] - b["y1"], b["x2"] - b["x1"])
            local_bloom = bloom_disp * smoothstep(0.7, 1.0, frac)
            draw_flower(base_layer, ex, ey, angle, 16 + depth_t * 5, local_bloom)
            draw_flower(glow_layer, ex, ey, angle, 20 + depth_t * 5, local_bloom)


FINGER_TIPS = (4, 8, 12, 16, 20)  


def hand_metrics(hand_landmarks, w, h):
    """Return palm center, pixel points for all landmarks, and a scale-
    invariant 'openness' score based on how far the fingertips are spread
    from the palm (closed fist -> low, open hand -> high)."""
    pts = [(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark]
    wrist = pts[0]
    palm_idx = (0, 5, 9, 13, 17)
    palm_x = sum(pts[i][0] for i in palm_idx) / len(palm_idx)
    palm_y = sum(pts[i][1] for i in palm_idx) / len(palm_idx)

    scale = math.hypot(pts[9][0] - wrist[0], pts[9][1] - wrist[1]) + 1e-6

    openness = sum(
        math.hypot(pts[i][0] - palm_x, pts[i][1] - palm_y) / scale
        for i in FINGER_TIPS
    ) / len(FINGER_TIPS)

    return (palm_x, palm_y), pts, openness


def draw_hand_veins(layer, palm, pts, color):
    """Draw glowing lines from the palm out to each fingertip, so the
    visualization visibly tracks finger spread, not just hand position."""
    px, py = int(palm[0]), int(palm[1])
    for i in FINGER_TIPS:
        tx, ty = int(pts[i][0]), int(pts[i][1])
        cv2.line(layer, (px, py), (tx, ty), color, 2, cv2.LINE_AA)
        cv2.circle(layer, (tx, ty), 4, color, -1, cv2.LINE_AA)
    cv2.circle(layer, (px, py), 6, color, -1, cv2.LINE_AA)


class Calibrator:
    """Maps a raw signal to 0..1 using an auto-shrinking min/max window
    (like AGC). It instantly captures new extremes, but also slowly relaxes
    its bounds toward whatever you're currently doing - so if your natural
    hand movement only covers a small range, the window shrinks to fit that
    range and small movements will still swing all the way from 0 to 1."""
    def __init__(self, lo=0.95, hi=1.35, relax=0.008):
        self.lo, self.hi, self.relax = lo, hi, relax

    def normalize(self, raw):
        self.lo = min(self.lo, raw)
        self.hi = max(self.hi, raw)
        self.lo += (raw - self.lo) * self.relax
        self.hi += (raw - self.hi) * self.relax
        span = max(0.05, self.hi - self.lo)
        return max(0.0, min(1.0, (raw - self.lo) / span))


def draw_hand_guide(frame, x, value, color, label):
    h = frame.shape[0]
    bottom_y = int(h * 0.95)
    top_y = int(h * 0.15)
    marker_y = int(bottom_y - (bottom_y - top_y) * value)

    cv2.line(frame, (x, bottom_y), (x, top_y), color, 1, cv2.LINE_AA)
    # diamond marker
    s = 9
    pts = np.array([[x, marker_y - s], [x + s, marker_y],
                     [x, marker_y + s], [x - s, marker_y]], np.int32)
    cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)

    cv2.putText(frame, f"{label}: {value:.2f}", (x - 55, marker_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions / index.")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read from webcam.")
    h, w = frame.shape[:2]
    branches = build_tree(w, h, x_frac=TREE_X_FRAC)

    grow_val, bloom_val = 0.0, 0.0        # raw target values from finger spread
    grow_disp, bloom_disp = 0.0, 0.0      # smoothed/eased values
    grow_x, bloom_x = int(w * 0.22), int(w * 0.78)
    grow_cal = Calibrator()
    bloom_cal = Calibrator()
    grow_raw_smooth, bloom_raw_smooth = None, None
    RAW_SMOOTHING = 0.35 

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        vein_layer = np.zeros_like(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                (palm_x, palm_y), pts, raw_openness = hand_metrics(hand_landmarks, w, h)

                if palm_x < w / 2:
                    grow_raw_smooth = raw_openness if grow_raw_smooth is None else \
                        grow_raw_smooth + (raw_openness - grow_raw_smooth) * RAW_SMOOTHING
                    grow_val = grow_cal.normalize(grow_raw_smooth)
                    grow_x = int(palm_x)
                    draw_hand_veins(vein_layer, (palm_x, palm_y), pts, GROW_UI_COLOR)
                else:
                    bloom_raw_smooth = raw_openness if bloom_raw_smooth is None else \
                        bloom_raw_smooth + (raw_openness - bloom_raw_smooth) * RAW_SMOOTHING
                    bloom_val = bloom_cal.normalize(bloom_raw_smooth)
                    bloom_x = int(palm_x)
                    draw_hand_veins(vein_layer, (palm_x, palm_y), pts, BLOOM_UI_COLOR)
        grow_disp += (grow_val - grow_disp) * 0.22
        bloom_disp += (bloom_val - bloom_disp) * 0.22

        glow_layer = vein_layer.copy()
        draw_tree(frame, glow_layer, branches, grow_disp, bloom_disp)

        vein_blurred = cv2.GaussianBlur(vein_layer, (0, 0), sigmaX=4, sigmaY=4)
        frame = cv2.add(frame, vein_blurred)

        glow_blurred = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=9, sigmaY=9)
        frame = cv2.add(frame, glow_blurred)

        draw_hand_guide(frame, grow_x, grow_disp, GROW_UI_COLOR, "Grow")
        draw_hand_guide(frame, bloom_x, bloom_disp, BLOOM_UI_COLOR, "Bloom")

        cv2.putText(frame, "spread left hand to grow  -  spread right hand to bloom",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)

        cv2.imshow("Hand-Grown Flower", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
