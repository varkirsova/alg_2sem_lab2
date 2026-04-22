import numpy as np
import math


def adaption_quant_table(quality, table):
    if quality < 50:
        S = 5000 / quality
    else:
        S = 200 - 2 * quality
    res = np.zeros_like(table)

    for y in range(8):
        for x in range(8):
            q = table[y, x]
            q_new = math.ceil((q * S) / 100)
            q_new = max(1, min(255, q_new))
            res[y, x] = q_new

    return res
