import numpy as np


def immersion(input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = input.shape
    out_h = 2 * h - 1
    out_w = 2 * w - 1

    M = np.empty((out_h, out_w), dtype=input.dtype)
    m = np.empty_like(M)

    M[::2, ::2] = input
    m[::2, ::2] = input

    M[::2, 1::2] = np.maximum(input[:, :-1], input[:, 1:])
    m[::2, 1::2] = np.minimum(input[:, :-1], input[:, 1:])

    M[1::2, ::2] = np.maximum(input[:-1, :], input[1:, :])
    m[1::2, ::2] = np.minimum(input[:-1, :], input[1:, :])

    tmp_max1 = np.maximum(input[:-1, :-1], input[:-1, 1:])
    tmp_max2 = np.maximum(input[1:, :-1], input[1:, 1:])
    M[1::2, 1::2] = np.maximum(tmp_max1, tmp_max2)

    tmp_min1 = np.minimum(input[:-1, :-1], input[:-1, 1:])
    tmp_min2 = np.minimum(input[1:, :-1], input[1:, 1:])
    m[1::2, 1::2] = np.minimum(tmp_min1, tmp_min2)

    return (m, M)
