import numpy as np


def add_border(img: np.ndarray, v: int) -> np.ndarray:
    res = np.empty((img.shape[0] + 2, img.shape[1] + 2), dtype=img.dtype)

    res[1:-1, 1:-1] = img
    res[0] = v
    res[-1] = v
    res[:, 0] = v
    res[:, -1] = v

    return res


def add_median_border(img: np.ndarray) -> np.ndarray:
    border_values = np.concatenate((img[0], img[-1], img[:, 0], img[:, -1]))
    v = border_values[border_values.size // 2]
    return add_border(img, v)
