C4 = ((0, -1), (1, 0), (0, 1), (-1, 0))
C8 = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))


def in_domain(shape: tuple[int, int], l: int, c: int) -> bool:
    return l >= 0 and c >= 0 and l < shape[0] and c < shape[1]


def clamp(v: int, vmin: int, vmax: int) -> int:
    if v < vmin:
        return vmin
    elif v > vmax:
        return vmax
    else:
        return v


def is_2_face(l: int, c: int) -> bool:
    return l % 2 == 0 and c % 2 == 0
