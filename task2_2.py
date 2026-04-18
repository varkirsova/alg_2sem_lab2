import numpy as np

def zigzag(matrix):
    n, m = matrix.shape
    res = []
    for d in range((n - 1) + (m - 1) + 1): # d номер диагонали
        diag = []
        for i in range(n):
            j = d - i
            if 0 <= j < m:
                diag.append((i, j))
        if d % 2 == 0:
            diag.reverse()
        for i, j in diag:
            res.append(matrix[i, j])

    return res
