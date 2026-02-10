import numpy as np
from numba import njit


@njit
def _iter(img: np.ndarray, dist: np.ndarray, forward: bool) -> bool:
    changed = False

    if forward:
        DL = [0, -1, -1, -1]
        DC = [-1, -1, 0, 1]
        start_l = 0
        start_c = 0
        end_l = img.shape[0]
        end_c = img.shape[1]
        inc = 1
    else:
        DL = [0, 1, 1, 1]
        DC = [1, 1, 0, -1]
        start_l = img.shape[0] - 1
        start_c = img.shape[1] - 1
        end_l = -1
        end_c = -1
        inc = -1

    for l in range(start_l, end_l, inc):
        for c in range(start_c, end_c, inc):
            for dl, dc in zip(DL, DC):
                nl = l + dl
                nc = c + dc
                if nl >= 0 and nc >= 0 and nl < img.shape[0] and nc < img.shape[1]:
                    d_new = dist[nl, nc] + abs(float(img[l, c]) - float(img[nl, nc]))
                    if d_new < dist[l, c]:
                        dist[l, c] = d_new
                        changed = True
    return changed


def geodesic_distance_transform(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    res = np.full(img.shape, dtype=np.float32, fill_value=1e10)
    res[mask] = 0

    changed = True
    while changed:
        c1 = _iter(img, res, True)
        c2 = _iter(img, res, False)
        changed = c1 or c2

    return res
