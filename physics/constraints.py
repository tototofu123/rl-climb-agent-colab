import numpy as np

MAX_REACH_STATIC = 150.0
MAX_REACH_DYNO = 250.0


def is_reachable(from_pos, to_pos, max_reach=MAX_REACH_STATIC):
    dist = np.linalg.norm(np.array(from_pos) - np.array(to_pos))
    return dist <= max_reach


def distance(pos1, pos2):
    return float(np.linalg.norm(np.array(pos1) - np.array(pos2)))
