"""Pure-python clue logic (no ROS, no OpenCV): token chain, parsing, geometry."""
import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

CLUE_RE = re.compile(r'^HUNT:(\d+):([0-9A-Fa-f]{4}):(TREASURE\s+)?(.+)$')


def token_for(prev_text: str) -> str:
    """First 4 hex chars (upper case) of SHA-1 of the previous valid clue text ('START' for board 1)."""
    return hashlib.sha1(prev_text.encode()).hexdigest()[:4].upper()


@dataclass
class Clue:
    text: str
    id: int
    token: str
    treasure: bool
    cmd: str                      # GOTO / PILLAR / BETWEEN / REL
    args: List = field(default_factory=list)


def parse_clue(text: str) -> Optional[Clue]:
    """Parse 'HUNT:<id>:<token>:[TREASURE ]<command>'. Returns None if malformed."""
    if text is None:
        return None
    text = text.strip()
    m = CLUE_RE.match(text)
    if not m:
        return None
    cid, tok, tre, rest = int(m.group(1)), m.group(2).upper(), m.group(3), m.group(4).strip()
    parts = rest.split()
    if not parts:
        return None
    cmd, raw = parts[0].upper(), parts[1:]
    try:
        if cmd == 'GOTO' and len(raw) == 2:
            args = [float(raw[0]), float(raw[1])]
        elif cmd == 'PILLAR' and len(raw) == 1:
            args = [raw[0].upper()]
        elif cmd == 'BETWEEN' and len(raw) == 3:
            args = [raw[0].upper(), raw[1].upper(), float(raw[2])]
        elif cmd == 'REL' and len(raw) == 2:
            args = [float(raw[0]), float(raw[1])]
        else:
            return None
    except ValueError:
        return None
    return Clue(text=text, id=cid, token=tok, treasure=tre is not None, cmd=cmd, args=args)


class ChainValidator:
    """Accepts only the next board of the chain: right id AND right token.
    Decoy = wrong id.  Look-alike = right id, wrong token."""

    def __init__(self):
        self.prev_text = 'START'
        self.expected_id = 1
        self.valid: List[Clue] = []

    def check(self, text: str) -> Tuple[Optional[Clue], str]:
        clue = parse_clue(text)
        if clue is None:
            return None, 'malformed'
        if clue.id != self.expected_id:
            return None, f'decoy (id {clue.id}, expected {self.expected_id})'
        if clue.token != token_for(self.prev_text):
            return None, f'look-alike (id {clue.id}, bad token {clue.token})'
        return clue, 'ok'

    def accept(self, clue: Clue):
        self.valid.append(clue)
        self.prev_text = clue.text
        self.expected_id = clue.id + 1


# ---------------------------------------------------------------- geometry
def quat_to_matrix(x, y, z, w) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def transform_to_matrix(tx, ty, tz, qx, qy, qz, qw) -> np.ndarray:
    """geometry_msgs Transform (translation + quaternion) -> 4x4 matrix."""
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(qx, qy, qz, qw)
    T[:3, 3] = [tx, ty, tz]
    return T


def board_xy_and_axes(T_map_board: np.ndarray):
    """Board origin (= ArUco centre), board +X (out of face) and +Y (reader's right) in map, 2D."""
    p = T_map_board[:3, 3]
    X = T_map_board[:3, 0]
    Y = T_map_board[:3, 1]
    return p[:2], X[:2], Y[:2]


def rel_target(T_map_board: np.ndarray, a: float, b: float) -> np.ndarray:
    """REL a b: ArUco centre + a*X + b*Y (board frame), returned as map (x, y)."""
    p, X, Y = board_xy_and_axes(T_map_board)
    return p + a * X + b * Y


def between_point(A, B, f: float) -> np.ndarray:
    A, B = np.asarray(A, float), np.asarray(B, float)
    return A + f * (B - A)


def ring_viewpoints(center, radius=1.2, n=8, include_center=True):
    """Candidate search viewpoints around a target point: (x, y, yaw facing the target)."""
    c = np.asarray(center, float)
    pts = []
    if include_center:
        pts.append((c[0], c[1], 0.0))
    for k in range(n):
        ang = 2 * math.pi * k / n
        p = c + radius * np.array([math.cos(ang), math.sin(ang)])
        yaw = math.atan2(c[1] - p[1], c[0] - p[0])
        pts.append((float(p[0]), float(p[1]), yaw))
    return pts


def approach_goal(T_map_board: np.ndarray, dist=1.3, offset_deg=25.0):
    """Goal pose in front of a board, `dist` m away, facing the board.
    offset_deg turns the viewpoint away from the face normal: a slightly oblique view gives a much
    better board orientation (head-on solvePnP can be ~5 deg off, which ruins REL targets at 3+ m)."""
    p, X, _ = board_xy_and_axes(T_map_board)
    n = X / (np.linalg.norm(X) or 1.0)
    a = math.radians(offset_deg)
    d = np.array([n[0] * math.cos(a) - n[1] * math.sin(a), n[0] * math.sin(a) + n[1] * math.cos(a)])
    g = p + dist * d
    yaw = math.atan2(p[1] - g[1], p[0] - g[0])
    return float(g[0]), float(g[1]), yaw
