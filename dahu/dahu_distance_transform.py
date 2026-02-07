import numpy as np

from .pqueue import PQueue
from .utils import in_domain, is_2_face, C4, clamp


def dahu_distance_transform(
    m: np.ndarray, M: np.ndarray, seeds: list[tuple[int, int]] = [(0, 0)]
) -> np.ndarray:
    deja_vu = np.zeros(m.shape, dtype=bool)
    min_img = np.empty_like(m)
    max_img = np.empty_like(m)
    F = np.empty_like(m)
    q = PQueue()

    for l, c in seeds:
        assert in_domain(m.shape, l, c) and is_2_face(l, c)
        F[l, c] = m[l, c]
        min_img[l, c] = m[l, c]
        max_img[l, c] = m[l, c]
        deja_vu[l, c] = True
        q.push((l, c), 0)

    while not q.empty():
        (l, c) = q.pop()
        d = q.distance
        for dl, dc in C4:
            nl, nc = l + dl, c + dc
            if in_domain(m.shape, nl, nc) and not deja_vu[nl, nc]:
                f = clamp(F[l, c], m[nl, nc], M[nl, nc])
                F[nl, nc] = f
                diff = abs(int(F[l, c]) - int(F[nl, nc]))
                min_img[nl, nc] = min(f, min_img[l, c])
                max_img[nl, nc] = max(f, max_img[l, c])
                deja_vu[nl, nc] = True
                q.push((nl, nc), diff)

    return max_img - min_img
