import numpy as np

from .pqueue import PQueue
from .utils import in_domain, C4, is_2_face, clamp


def level_lines_distance_transform(
    m: np.ndarray, M: np.ndarray, seeds: list[tuple[int, int]] = [(0, 0)]
) -> tuple[np.ndarray, np.ndarray]:
    # Initialize structures
    UNSEEN = np.iinfo(np.uint32).max
    F = np.empty_like(m)
    D = np.full(m.shape, dtype=np.uint32, fill_value=UNSEEN)
    q = PQueue()

    # Process initial points
    for l, c in seeds:
        assert in_domain(m.shape, l, c) and is_2_face(l, c)
        D[l, c] = 0
        F[l, c] = m[l, c]  # Here m[l, c] == M[l, c] (it is a 2 face)
        q.push((l, c), 0)

    while not q.empty():
        (l, c) = q.pop()
        d = q.distance
        for dl, dc in C4:
            nl, nc = l + dl, c + dc
            if in_domain(m.shape, nl, nc) and D[nl, nc] == UNSEEN:
                F[nl, nc] = clamp(F[l, c], m[nl, nc], M[nl, nc])
                diff = abs(int(F[l, c]) - int(F[nl, nc]))
                D[nl, nc] = d + diff
                q.push((nl, nc), diff)

    return F, D
