"""Vision part of the leader: ArUco + QR detection, QR<->marker matching, board pose (solvePnP).
Pure OpenCV/numpy so it can be tested without ROS.

Frames:
  camera optical frame : x right, y down, z forward
  OpenCV marker frame  : x right, y up, z out of the face (object points below)
  board frame (task)   : +X out of the face, +Y reader's right, +Z up, origin = ArUco centre
  => board X = marker z, board Y = marker x, board Z = marker y
"""
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

MARKER_SIDE = 0.24        # clue-board ArUco side (m)
TAG_SIDE = 0.12           # leader's back tag (m)
QR_OFFSET = 0.30          # QR centre, to the reader's right of the ArUco centre (m)
_DICT_ID = cv2.aruco.DICT_4X4_50

# marker frame -> board frame:  columns are the board axes written in marker coordinates
M_MARKER_TO_BOARD = np.array([[0, 1, 0],
                              [0, 0, 1],
                              [1, 0, 0]], dtype=float)

# ------------------------------------------------------------------ ArUco (old + new OpenCV API)
_NEW_API = hasattr(cv2.aruco, 'ArucoDetector')
_dict = cv2.aruco.getPredefinedDictionary(_DICT_ID)
if _NEW_API:
    _p = cv2.aruco.DetectorParameters()
    _p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    _detector = cv2.aruco.ArucoDetector(_dict, _p)
else:  # OpenCV 4.5.x (Ubuntu 22.04 apt)
    _p = cv2.aruco.DetectorParameters_create()
    _p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX


def _detect_markers(gray):
    if _NEW_API:
        corners, ids, _ = _detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, _dict, parameters=_p)
    return corners, ids


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr


def detect_all_markers(bgr):
    """Return {marker_id: [corners(4x2) ...]} - several boards may share an id.
    Tries plain gray first, then CLAHE (lighting changes in hidden worlds)."""
    gray = _to_gray(bgr)
    out = {}
    for img in (gray, _clahe.apply(gray)):
        corners, ids = _detect_markers(img)
        if ids is None:
            continue
        for c, i in zip(corners, ids.ravel()):
            c = c.reshape(4, 2).astype(np.float64)
            lst = out.setdefault(int(i), [])
            ctr = c.mean(axis=0)
            if not any(np.linalg.norm(ctr - o.mean(axis=0)) < 8 for o in lst):  # de-duplicate
                lst.append(c)
        if out:
            break
    return out


# ------------------------------------------------------------------ pose
def marker_object_points(side):
    h = side / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def estimate_marker_pose(corners, K, dist, side):
    """solvePnP (IPPE_SQUARE). Returns (T_cam_marker 4x4, reproj_err px) or None.
    Chooses the IPPE solution with the lowest error whose normal faces the camera."""
    obj = marker_object_points(side)
    img = np.asarray(corners, np.float64).reshape(4, 1, 2)
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        return None
    best = None
    for rv, tv, e in zip(rvecs, tvecs, np.ravel(errs)):
        R, _ = cv2.Rodrigues(rv)
        if R[2, 2] > 0:          # marker z (face normal) must point towards the camera (-z_cam)
            continue
        if best is None or e < best[2]:
            best = (R, tv.reshape(3), float(e))
    if best is None:
        return None
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = best[0], best[1]
    return T, best[2]


def marker_to_board(T_cam_marker):
    """Same origin, axes renamed to the task's board frame."""
    T = T_cam_marker.copy()
    T[:3, :3] = T_cam_marker[:3, :3] @ M_MARKER_TO_BOARD
    return T


# ------------------------------------------------------------------ QR next to the marker
PPM = 1500          # rectified pixels per metre
RECT_HALF = 0.20    # half-size of the rectified QR window (m)


def rectify_qr_window(gray, corners):
    """Perspective-correct the area around the QR position, using the marker as the plane reference.
    Because the window is placed relative to THIS marker, a QR is always tied to its own marker."""
    s = MARKER_SIDE / 2.0
    # marker corners in a 'plane' frame (x right, y down) in metres
    plane = np.array([[-s, -s], [s, -s], [s, s], [-s, s]], np.float32)
    x0, y0 = QR_OFFSET - RECT_HALF, -RECT_HALF
    dst = ((plane - np.array([x0, y0], np.float32)) * PPM).astype(np.float32)
    H = cv2.getPerspectiveTransform(np.asarray(corners, np.float32), dst)
    size = int(2 * RECT_HALF * PPM)
    return cv2.warpPerspective(gray, H, (size, size), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


_qr = cv2.QRCodeDetector()


def decode_qr_for_marker(bgr, corners) -> Optional[str]:
    """Decode the QR beside this marker. Tries several scales / contrast variants because the
    OpenCV decoder is sensitive to module size and noise."""
    gray = _to_gray(bgr)
    rect = rectify_qr_window(gray, corners)
    for scale in (1.0, 0.6, 0.45, 0.8, 0.35):
        base = rect if scale == 1.0 else cv2.resize(rect, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        variants = (base, _clahe.apply(base), cv2.GaussianBlur(base, (3, 3), 0),
                    cv2.threshold(base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
        for img in variants:
            img = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
            try:
                txt, _, _ = _qr.detectAndDecode(img)
            except cv2.error:
                txt = ''
            if txt:
                return txt
    return None


# ------------------------------------------------------------------ public API
@dataclass
class BoardDetection:
    marker_id: int
    corners: np.ndarray
    qr_text: Optional[str]
    T_cam_board: Optional[np.ndarray]
    reproj_err: float
    pixel_size: float          # marker side in pixels (bigger = closer / more reliable)


def detect_boards(bgr, K, dist=None, read_qr=True, min_px=18) -> List[BoardDetection]:
    """All clue-board candidates in the image (ArUco id != 49), with QR text if readable."""
    if dist is None:
        dist = np.zeros(5)
    out = []
    for mid, lst in detect_all_markers(bgr).items():
        if mid == 49:
            continue
        for c in lst:
            px = float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))
            if px < min_px:
                continue
            res = estimate_marker_pose(c, K, dist, MARKER_SIDE)
            T = marker_to_board(res[0]) if res else None
            err = res[1] if res else 1e9
            txt = decode_qr_for_marker(bgr, c) if read_qr else None
            out.append(BoardDetection(mid, c, txt, T, err, px))
    return out


def detect_leader_tag(bgr, K, dist=None):
    """Follower: pose of the leader's back tag (id 49, 0.12 m) in the camera optical frame.
    Returns (T_cam_tag 4x4, corners) or None.  Tag frame = OpenCV marker frame."""
    if dist is None:
        dist = np.zeros(5)
    lst = detect_all_markers(bgr).get(49)
    if not lst:
        return None
    best = max(lst, key=lambda c: cv2.contourArea(c.astype(np.float32)))
    res = estimate_marker_pose(best, K, dist, TAG_SIDE)
    return (res[0], best) if res else None
