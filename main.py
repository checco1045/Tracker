"""
Motion Tracker — Traccia l'oggetto più veloce
=============================================
Rileva tutti gli oggetti in movimento, li mostra tutti,
ma traccia la traiettoria solo di quello che si muove di più.

Dipendenze:
    pip install opencv-python numpy

Controlli:
    Q       → Esci
    C       → Cancella la traccia
    S       → Salva screenshot
    +/-     → Aumenta/diminuisci sensibilità
    SPAZIO  → Pausa/Riprendi
"""

import cv2
import numpy as np
from collections import deque
import time

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────
TRAIL_LENGTH      = 100
SENSITIVITY       = 100
MIN_AREA          = 1500
BLUR_RADIUS       = 21

TRAIL_COLOR_START = (0, 255, 100)
TRAIL_COLOR_END   = (0, 80, 255)
DOT_COLOR         = (255, 255, 255)

# Colori per gli oggetti "non scelti" (visibili ma non tracciati)
IDLE_BOX_COLOR    = (160, 160, 160)   # grigio
IDLE_LABEL_COLOR  = (200, 200, 200)

# Colore per l'oggetto attivo (il più veloce)
ACTIVE_BOX_COLOR  = (0, 200, 255)
ACTIVE_LABEL_COLOR= (0, 230, 255)

# Quanti frame usare per calcolare la velocità media (smussamento)
SPEED_SMOOTH      = 5


# ─────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────

def interpolate_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def draw_trail(canvas, trail):
    n = len(trail)
    for i in range(1, n):
        if trail[i - 1] is None or trail[i] is None:
            continue
        t = i / max(n - 1, 1)
        color = interpolate_color(TRAIL_COLOR_START, TRAIL_COLOR_END, t)
        thickness = max(1, int(3 * t) + 1)
        cv2.line(canvas, trail[i - 1], trail[i], color, thickness, cv2.LINE_AA)

    if trail and trail[-1]:
        cv2.circle(canvas, trail[-1], 8, DOT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(canvas, trail[-1], 10, TRAIL_COLOR_END, 2, cv2.LINE_AA)


def draw_hud(frame, fps, sensitivity, paused, n_points, n_objects):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 175), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    lines = [
        (f"FPS: {fps:.1f}", (200, 200, 200)),
        (f"Sensibilita': {sensitivity}  (+/-)", (200, 200, 200)),
        (f"Oggetti rilevati: {n_objects}", (200, 200, 200)),
        (f"Punti traccia: {n_points}", (200, 200, 200)),
        ("C=Cancella  S=Salva  Q=Esci", (200, 200, 200)),
        ("PAUSA" if paused else "IN ESECUZIONE",
         (0, 120, 255) if paused else (0, 230, 100)),
    ]
    for i, (line, color) in enumerate(lines):
        cv2.putText(frame, line, (10, 25 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def draw_corner_box(frame, x, y, w, h, color, thickness=2):
    """Bounding box con soli gli angolini invece del rettangolo pieno."""
    corner = min(16, w // 4, h // 4)
    pts = [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]
    dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    for (px, py), (dx, dy) in zip(pts, dirs):
        cv2.line(frame, (px, py), (px + dx*corner, py), color, thickness)
        cv2.line(frame, (px, py), (px, py + dy*corner), color, thickness)


# ─────────────────────────────────────────
# SPEED TRACKER PER OGGETTO
# ─────────────────────────────────────────

class ObjectSpeedTracker:
    """
    Associa un ID a ogni blob rilevato tra frame consecutivi
    e calcola la sua velocità media su una finestra scorrevole.
    """

    def __init__(self, smooth=SPEED_SMOOTH, max_dist=120):
        self.smooth   = smooth
        self.max_dist = max_dist           # distanza massima per l'associazione
        self.objects  = {}                 # id -> deque di centroidi recenti
        self._next_id = 0

    def update(self, centroids):
        """
        Riceve la lista di centroidi (x,y) del frame corrente.
        Restituisce una lista di (centroid, speed, obj_id).
        """
        if not centroids:
            # nessun blob: decadimento (non usiamo gli id vecchi)
            self.objects.clear()
            return []

        # ── Associazione greedy per distanza minima ──
        old_ids    = list(self.objects.keys())
        old_cents  = [self.objects[i][-1] for i in old_ids]
        used_old   = set()
        result     = []

        for cx, cy in centroids:
            best_id   = None
            best_dist = self.max_dist

            for oi, (ox, oy) in enumerate(old_cents):
                if oi in used_old:
                    continue
                d = np.hypot(cx - ox, cy - oy)
                if d < best_dist:
                    best_dist = d
                    best_id   = oi

            if best_id is not None:
                obj_id = old_ids[best_id]
                used_old.add(best_id)
            else:
                obj_id = self._next_id
                self._next_id += 1
                self.objects[obj_id] = deque(maxlen=self.smooth + 1)

            self.objects[obj_id].append((cx, cy))

        # Rimuovi oggetti non aggiornati
        active_ids = {r[2] for r in result} if result else set()
        # ricostruiamo result dopo aver popolato objects
        result = []
        for cx, cy in centroids:
            # trova l'id associato (ricerca veloce)
            for oid, hist in self.objects.items():
                if hist[-1] == (cx, cy):
                    speed = self._speed(oid)
                    result.append(((cx, cy), speed, oid))
                    break

        # Pulisce oggetti non presenti nel frame corrente
        current_ids = {r[2] for r in result}
        for dead_id in list(self.objects.keys()):
            if dead_id not in current_ids:
                del self.objects[dead_id]

        return result

    def _speed(self, obj_id):
        hist = self.objects.get(obj_id)
        if not hist or len(hist) < 2:
            return 0.0
        # velocità media sugli ultimi N step
        dists = [np.hypot(hist[i][0]-hist[i-1][0], hist[i][1]-hist[i-1][1])
                 for i in range(1, len(hist))]
        return sum(dists) / len(dists)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Webcam non trovata.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, first = cap.read()
    if not ret:
        print("❌  Impossibile leggere dalla webcam.")
        return

    h, w = first.shape[:2]

    sensitivity  = SENSITIVITY
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=sensitivity, detectShadows=False)

    trail        = deque(maxlen=TRAIL_LENGTH)
    trail_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    speed_tracker = ObjectSpeedTracker()

    paused    = False
    prev_time = time.time()

    print("✅  Motion Tracker avviato  (traccia l'oggetto più veloce)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        key   = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            trail.clear()
            trail_canvas[:] = 0
        elif key == ord('s'):
            fname = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(fname, cv2.addWeighted(frame, 0.7, trail_canvas, 0.9, 0))
            print(f"📸  Salvato: {fname}")
        elif key in (ord('+'), ord('=')):
            sensitivity = max(5, sensitivity - 5)
            bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=300, varThreshold=sensitivity, detectShadows=False)
            trail.clear(); trail_canvas[:] = 0
        elif key == ord('-'):
            sensitivity = min(100, sensitivity + 5)
            bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=300, varThreshold=sensitivity, detectShadows=False)
            trail.clear(); trail_canvas[:] = 0
        elif key == ord(' '):
            paused = not paused

        if paused:
            fps = 0.0
            active_center = None
            n_objects = 0
        else:
            # ── Rilevamento movimento ────────────────────
            fg = bg_sub.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
            fg = cv2.GaussianBlur(fg, (BLUR_RADIUS, BLUR_RADIUS), 0)
            _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            blobs = []   # (centroid, bbox) per ogni contorno valido
            for cnt in contours:
                if cv2.contourArea(cnt) < MIN_AREA:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x, y, bw, bh = cv2.boundingRect(cnt)
                blobs.append(((cx, cy), (x, y, bw, bh)))

            centroids = [b[0] for b in blobs]
            bbox_map  = {b[0]: b[1] for b in blobs}

            # ── Calcola velocità per ogni blob ───────────
            tracked = speed_tracker.update(centroids)
            # tracked = [((cx,cy), speed, obj_id), ...]

            # ── Scegli il più veloce ─────────────────────
            active_center = None
            n_objects = len(tracked)

            if tracked:
                tracked.sort(key=lambda r: r[1], reverse=True)
                fastest = tracked[0]
                active_center = fastest[0]
                active_speed  = fastest[1]

                for (cent, speed, oid) in tracked:
                    bbox = bbox_map.get(cent)
                    if bbox is None:
                        continue
                    bx, by, bw, bh = bbox

                    is_active = (cent == active_center)

                    if is_active:
                        # box luminoso + etichetta velocità
                        draw_corner_box(frame, bx, by, bw, bh,
                                        ACTIVE_BOX_COLOR, thickness=2)
                        label = f"v={speed:.1f}px/f"
                        cv2.putText(frame, label, (bx, by - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    ACTIVE_LABEL_COLOR, 1, cv2.LINE_AA)
                        # mirino
                        cv2.drawMarker(frame, cent, ACTIVE_BOX_COLOR,
                                       cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
                    else:
                        # box grigio semitrasparente
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (bx, by), (bx+bw, by+bh),
                                      IDLE_BOX_COLOR, 1)
                        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                        cv2.putText(frame, f"v={speed:.1f}", (bx, by - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                    IDLE_LABEL_COLOR, 1, cv2.LINE_AA)

            trail.append(active_center)
            if len(trail) > 1:
                draw_trail(trail_canvas, list(trail))

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

        # ── Compositing ──────────────────────────────────
        display = cv2.addWeighted(frame, 0.65, trail_canvas, 0.95, 0)
        draw_hud(display, fps if not paused else 0,
                 sensitivity, paused,
                 len([p for p in trail if p]),
                 n_objects if not paused else 0)

        cv2.imshow("Motion Tracker  |  Q=Esci", display)

    cap.release()
    cv2.destroyAllWindows()
    print("👋  Chiuso.")


if __name__ == "__main__":
    main()