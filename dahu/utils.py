import numpy as np

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


def get_coordinates(mask: np.ndarray) -> np.ndarray:
    ml, mc = np.where(mask == 1)
    return np.vstack((ml, mc)).T


def get_marker_image(img: np.ndarray, markers: np.ndarray):
    res = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        res[:, :, i] = img
    res[markers == 1] = [0, 0, 255]
    res[markers == 2] = [255, 0, 0]
    return res
