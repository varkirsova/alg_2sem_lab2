def dc_diff(dc):
    if dc is None or len(dc) == 0:
        return []

    res = [dc[0]]
    for i in range(1, len(dc)):
        res.append(dc[i] - dc[i -1])
    return res

def rle(ac):
    res = []
    run = 0
    for i in ac:
        if i == 0:
            run += 1
            # не больше 15 нулей подряд
            if run == 16:
                res.append((15, 0))
                run = 0
        else:
            res.append((run, i))
            run = 0
    # остались только нули, спец код (0,0)
    if run > 0:
        res.append((0, 0))
    return res
