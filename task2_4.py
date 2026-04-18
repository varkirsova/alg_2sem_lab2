def coding_varible_length_dc(dc_diff):
    res = []
    for i in dc_diff:
        if i == 0:
            res.append((0, ""))
            continue

        abs_val = abs(i)
        size = abs_val.bit_length()
        if i > 0:
            amplitude = format(i, f'0{size}b')
        else:
            a = format(abs_val, f'0{size}b')
            amplitude = ''.join('1' if b == '0' else '0' for b in a)

        res.append((size, amplitude))
    return res


def coding_varible_length_ac(ac):
    res = []
    zeros = 0
    for i in ac:
        if i == 0:
            zeros += 1
            if zeros == 16:
                res.append((0xF0, ""))
                zeros = 0

        else:
            abs_val = int(abs(i))
            size = abs_val.bit_length()

            runsize = (zeros << 4) | size
            if i > 0:
                amplitude = format(i, f'0{size}b')
            else:
                a = format(abs_val, f'0{size}b')
                amplitude = ''.join('1' if b == '0' else '0' for b in a)

            res.append((runsize, amplitude))
            zeros = 0
    if zeros > 0:
        res.append((0x00, ""))

    return res

