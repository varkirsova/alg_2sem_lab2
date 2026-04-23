import numpy as np
from PIL import Image

from task2 import rgb_ycbcr, ycbcr_rgb
from task2_2 import zigzag
from task2_4 import coding_varible_length_ac
from task2_5 import encode_dc_huffman, encode_ac_huffman
from task2_6 import adaption_quant_table
from task4 import quant, blocks8, dct_reverse_matr, dct_matr

huffmann_ac = {
        0x00: "1010",
        0xF0: "11111111001",
        0x01: "00",
        0x02: "01",
        0x03: "100",
        0x04: "1011",
        0x05: "11010",
        0x06: "1111000",
        0x07: "11111000",
        0x08: "1111110110",
        0x09: "1111111110000010",
        0x0A: "1111111110000011",
        0x11: "1100",
        0x12: "11011",
        0x13: "1111001",
        0x14: "111110110",
        0x15: "11111110110",
        0x16: "1111111110000100",
        0x17: "1111111110000101",
        0x18: "1111111110000110",
        0x19: "1111111110000111",
        0x1A: "1111111110001000",
        0x21: "11100",
        0x22: "11111001",
        0x23: "1111110111",
        0x24: "111111110100",
        0x25: "1111111110001001",
        0x26: "1111111110001010",
        0x27: "1111111110001011",
        0x28: "1111111110001100",
        0x29: "1111111110001101",
        0x2A: "1111111110001110",
        0x31: "111010",
        0x32: "111110111",
        0x33: "111111110101",
        0x34: "1111111110001111",
        0x35: "1111111110010000",
        0x36: "1111111110010001",
        0x37: "1111111110010010",
        0x38: "1111111110010011",
        0x39: "1111111110010100",
        0x3A: "1111111110010101",
        0x41: "111011",
        0x42: "1111111000",
        0x43: "1111111110010110",
        0x44: "1111111110010111",
        0x45: "1111111110011000",
        0x46: "1111111110011001",
        0x47: "1111111110011010",
        0x48: "1111111110011011",
        0x49: "1111111110011100",
        0x4A: "1111111110011101",
        0x51: "1111010",
        0x52: "11111110111",
        0x53: "1111111110011110",
        0x54: "1111111110011111",
        0x55: "1111111110100000",
        0x56: "1111111110100001",
        0x57: "1111111110100010",
        0x58: "1111111110100011",
        0x59: "1111111110100100",
        0x5A: "1111111110100101",
        0x61: "1111011",
        0x62: "111111110110",
        0x63: "1111111110100110",
        0x64: "1111111110100111",
        0x65: "1111111110101000",
        0x66: "1111111110101001",
        0x67: "1111111110101010",
        0x68: "1111111110101011",
        0x69: "1111111110101100",
        0x6A: "1111111110101101",
        0x71: "11111010",
        0x72: "111111110111",
        0x73: "1111111110101110",
        0x74: "1111111110101111",
        0x75: "1111111110110000",
        0x76: "1111111110110001",
        0x77: "1111111110110010",
        0x78: "1111111110110011",
        0x79: "1111111110110100",
        0x7A: "1111111110110101",
        0x81: "111111000",
        0x82: "111111111000000",
        0x83: "1111111110110110",
        0x84: "1111111110110111",
        0x85: "1111111110111000",
        0x86: "1111111110111001",
        0x87: "1111111110111010",
        0x88: "1111111110111011",
        0x89: "1111111110111100",
        0x8A: "1111111110111101",
        0x91: "111111001",
        0x92: "1111111110111110",
        0x93: "1111111110111111",
        0x94: "1111111111000000",
        0x95: "1111111111000001",
        0x96: "1111111111000010",
        0x97: "1111111111000011",
        0x98: "1111111111000100",
        0x99: "1111111111000101",
        0x9A: "1111111111000110",
        0xA1: "111111010",
        0xA2: "1111111111000111",
        0xA3: "1111111111001000",
        0xA4: "1111111111001001",
        0xA5: "1111111111001010",
        0xA6: "1111111111001011",
        0xA7: "1111111111001100",
        0xA8: "1111111111001101",
        0xA9: "1111111111001110",
        0xAA: "1111111111001111",
        0xB1: "1111111001",
        0xB2: "1111111111010000",
        0xB3: "1111111111010001",
        0xB4: "1111111111010010",
        0xB5: "1111111111010011",
        0xB6: "1111111111010100",
        0xB7: "1111111111010101",
        0xB8: "1111111111010110",
        0xB9: "1111111111010111",
        0xBA: "1111111111011000",
        0xC1: "1111111010",
        0xC2: "1111111111011001",
        0xC3: "1111111111011010",
        0xC4: "1111111111011011",
        0xC5: "1111111111011100",
        0xC6: "1111111111011101",
        0xC7: "1111111111011110",
        0xC8: "1111111111011111",
        0xC9: "1111111111100000",
        0xCA: "1111111111100001",
        0xD1: "11111111000",
        0xD2: "1111111111100010",
        0xD3: "1111111111100011",
        0xD4: "1111111111100100",
        0xD5: "1111111111100101",
        0xD6: "1111111111100110",
        0xD7: "1111111111100111",
        0xD8: "1111111111101000",
        0xD9: "1111111111101001",
        0xDA: "1111111111101010",
        0xE1: "1111111111101011",
        0xE2: "1111111111101100",
        0xE3: "1111111111101101",
        0xE4: "1111111111101110",
        0xE5: "1111111111101111",
        0xE6: "1111111111110000",
        0xE7: "1111111111110001",
        0xE8: "1111111111110010",
        0xE9: "1111111111110011",
        0xEA: "1111111111110100",
        0xF1: "1111111111110101",
        0xF2: "1111111111110110",
        0xF3: "1111111111110111",
        0xF4: "1111111111111000",
        0xF5: "1111111111111001",
        0xF6: "1111111111111010",
        0xF7: "1111111111111011",
        0xF8: "1111111111111100",
        0xF9: "1111111111111101",
        0xFA: "1111111111111110"
    }

huffmann_dc = {
        0: "00",
        1: "010",
        2: "011",
        3: "100",
        4: "101",
        5: "110",
        6: "1110",
        7: "11110",
        8: "111110",
        9: "1111110",
        10: "11111110",
        11: "111111110"
    }

def bits_to_bytes(bitstring):
    if len(bitstring) % 8 != 0:
        bitstring += "0" * (8 - len(bitstring) % 8)

    out = bytearray()
    for i in range(0, len(bitstring), 8):
        byte = bitstring[i:i+8]
        out.append(int(byte, 2))
    return bytes(out)

def bytes_to_bits(data, length):
    bits = ''.join(f'{b:08b}' for b in data)
    return bits[:length]


def write_compressed_binary(
    filename,
    mode,
    width,
    height,
    qY, blocksY,
    qCb, blocksCb,
    qCr, blocksCr,
):
    buffer = bytearray()
    buffer.extend(mode)
    buffer.extend(width.to_bytes(4, "big"))
    buffer.extend(height.to_bytes(4, "big"))


    for v in qY.flatten():
        buffer.extend(int(v).to_bytes(2, "big"))
    for v in qCb.flatten():
        buffer.extend(int(v).to_bytes(2, "big"))
    for v in qCr.flatten():
        buffer.extend(int(v).to_bytes(2, "big"))
    def write_blocks(blocks):
        buffer.extend(len(blocks).to_bytes(4, "big"))

        for blk in blocks:
            dc_bits = blk["dc"]
            dc_bytes = bits_to_bytes(dc_bits)

            buffer.extend(len(dc_bits).to_bytes(2, "big"))
            buffer.extend(len(dc_bytes).to_bytes(2, "big"))
            buffer.extend(dc_bytes)

            buffer.extend(len(blk["ac"]).to_bytes(2, "big"))

            for ac in blk["ac"]:
                ac_bytes = bits_to_bytes(ac)

                buffer.extend(len(ac).to_bytes(2, "big"))
                buffer.extend(len(ac_bytes).to_bytes(2, "big"))
                buffer.extend(ac_bytes)
    write_blocks(blocksY)
    write_blocks(blocksCb)
    write_blocks(blocksCr)
    with open(filename, "wb") as f:
        f.write(buffer)

def read_compressed_binary(filename: str):
    with open(filename, "rb") as f:
        mode = f.read(1)

        width = int.from_bytes(f.read(4), "big")
        height = int.from_bytes(f.read(4), "big")


        qY  = np.array([int.from_bytes(f.read(2), "big") for _ in range(64)]).reshape(8, 8)
        qCb = np.array([int.from_bytes(f.read(2), "big") for _ in range(64)]).reshape(8, 8)
        qCr = np.array([int.from_bytes(f.read(2), "big") for _ in range(64)]).reshape(8, 8)

        def read_blocks():
            n = int.from_bytes(f.read(4), "big")
            blocks = []

            for _ in range(n):
                dc_bit_len = int.from_bytes(f.read(2), "big")
                dc_byte_len = int.from_bytes(f.read(2), "big")

                dc_bytes = f.read(dc_byte_len)
                dc_bits = bytes_to_bits(dc_bytes, dc_bit_len)

                ac_count = int.from_bytes(f.read(2), "big")
                ac_bits = []

                for _ in range(ac_count):
                    bit_len = int.from_bytes(f.read(2), "big")
                    byte_len = int.from_bytes(f.read(2), "big")

                    ac_bytes = f.read(byte_len)
                    ac_bits.append(bytes_to_bits(ac_bytes, bit_len))

                blocks.append({"dc": dc_bits, "ac": ac_bits})

            return blocks

        blocksY  = read_blocks()
        blocksCb = read_blocks()
        blocksCr = read_blocks()

        return (
            mode,
            width, height,
            qY, blocksY,
            qCb, blocksCb,
            qCr, blocksCr,
        )


def encode_channel(channel, q):
    blocks = blocks8(channel)
    compressed = []
    prev_dc = 0

    for i, block in enumerate(blocks):

        dct_block = dct_matr(block)
        cq = quant(dct_block, q)
        zz = zigzag(cq)

        dc = int(zz[0])
        dc_diff_val = dc - prev_dc
        prev_dc = dc

        dc_bits = encode_dc_huffman([dc_diff_val])[0]

        ac_vals = zz[1:]
        ac_rle = coding_varible_length_ac(ac_vals)
        ac_bits = encode_ac_huffman(ac_rle)

        compressed.append({"dc": dc_bits, "ac": ac_bits})
    return compressed


def encode_image(img, quality, q_base, filename):
    if img.ndim == 2:
        unique_vals = np.unique(img)
        if len(unique_vals) <= 2:
            mode = b'B'
        else:
            mode = b'G'
    else:
        mode = b'C'

    if mode in [b'G', b'B']:
        Y = img

        qY = adaption_quant_table(quality, q_base)
        blocksY = encode_channel(Y, qY)

        write_compressed_binary(
            filename,
            mode,
            Y.shape[1], Y.shape[0],
            qY, blocksY,
            np.zeros((8,8), dtype=np.int32), [],
            np.zeros((8,8), dtype=np.int32), []
        )
        return
    ycbcr = rgb_ycbcr(img)

    Y = ycbcr[:, :, 0]
    Cb = ycbcr[:, :, 1]
    Cr = ycbcr[:, :, 2]

    Cb_full = Cb
    Cr_full = Cr

    qY = adaption_quant_table(quality, q_base)
    qCb = qY
    qCr = qY

    blocksY = encode_channel(Y, qY)
    blocksCb = encode_channel(Cb_full, qCb)
    blocksCr = encode_channel(Cr_full, qCr)

    write_compressed_binary(
        filename,
        mode,
        Y.shape[1], Y.shape[0],
        qY, blocksY,
        qCb, blocksCb,
        qCr, blocksCr
    )


def build_reverse_dc(huff):
    return {v: k for k, v in huff.items()}

def build_reverse_ac(huff):
    return {v: k for k, v in huff.items()}

def decode_dc_symbol(bits, rev_dc):
    for L in range(1, 12):
        prefix = bits[:L]
        if prefix in rev_dc:
            size = rev_dc[prefix]
            if size == 0:
                return 0
            amp = bits[L:L+size]

            if amp[0] == '1':
                return int(amp, 2)

            inv = ''.join('1' if b=='0' else '0' for b in amp)
            return -int(inv, 2)
    raise ValueError("DC decode error")




def decode_ac_symbol(bits, rev_ac_sorted):
    for code, runsize in rev_ac_sorted:
        L = len(code)

        if bits[:L] == code:
            rest = bits[L:]

            if runsize == 0x00:
                return (0, 0)
            if runsize == 0xF0:
                return (15, 0)

            run = runsize >> 4
            size = runsize & 0x0F

            if size == 0:
                return (run, 0)

            if len(rest) < size:
                raise ValueError("AC amplitude truncated")

            amp = rest[:size]

            if amp[0] == '1':
                val = int(amp, 2)
            else:
                inv = ''.join('1' if b == '0' else '0' for b in amp)
                val = -int(inv, 2)

            return (run, val)

    raise ValueError("AC decode error")



def ac_from_rle(rle):
    out = []
    for run, val in rle:
        if (run, val) == (0, 0):
            while len(out) < 63:
                out.append(0)
            break
        if (run, val) == (15, 0):
            out.extend([0] * 16)
        else:
            out.extend([0] * run)
            out.append(val)

    if len(out) < 63:
        out.extend([0] * (63 - len(out)))

    return out[:63]


def inverse_zigzag(arr):
    out = np.zeros((8, 8), dtype=np.float64)

    zigzag_index = [
        (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
        (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
        (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
        (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
        (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
        (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
        (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
        (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7)
    ]

    for idx, (i, j) in enumerate(zigzag_index):
        out[i, j] = arr[idx]

    return out


def decode_block(dc_bits, ac_bits, qY, prev_dc, rev_dc, rev_ac_sorted):

    dc_diff = decode_dc_symbol(dc_bits, rev_dc)
    dc = prev_dc + dc_diff
    rle = []
    for bits in ac_bits:
        run, val = decode_ac_symbol(bits, rev_ac_sorted)
        rle.append((run, val))
        if (run, val) == (0, 0):
            break

    ac = ac_from_rle(rle)

    coeffs = [dc] + ac
    block_q = inverse_zigzag(coeffs)

    block_dct = block_q * qY
    block = dct_reverse_matr(block_dct)
    block = np.clip(np.round(block), 0, 255).astype(np.uint8)
    return block, dc

def decode_blocks(blocks, qY, height, width, rev_dc, rev_ac_sorted):

    H = (height + 7) // 8 * 8
    W = (width + 7) // 8 * 8

    out = np.zeros((H, W), dtype=np.uint8)
    prev_dc = 0
    idx = 0

    for y in range(0, H, 8):
        for x in range(0, W, 8):
            block = blocks[idx]
            block, prev_dc = decode_block(
                block["dc"],
                block["ac"],
                qY,
                prev_dc,
                rev_dc,
                rev_ac_sorted
            )

            out[y:y+8, x:x+8] = block

            idx += 1

    return out[:height, :width]

def decode_image(filename):
    (
        mode,
        width, height,
        qY, blocksY,
        qCb, blocksCb,
        qCr, blocksCr
    ) = read_compressed_binary(filename)

    rev_dc = build_reverse_dc(huffmann_dc)
    rev_ac = build_reverse_ac(huffmann_ac)
    rev_ac_sorted = sorted(rev_ac.items(), key=lambda x: len(x[0]))

    if mode == b'G':
        Y = decode_blocks(blocksY, qY, height, width, rev_dc, rev_ac_sorted)
        return Y

    elif mode == b'B':
        Y = decode_blocks(blocksY, qY, height, width, rev_dc, rev_ac_sorted)
        Y = (Y > 127).astype(np.uint8)
        return Y

    else:
        Y = decode_blocks(blocksY, qY, height, width, rev_dc, rev_ac_sorted)
        Cb = decode_blocks(blocksCb, qCb, height, width, rev_dc, rev_ac_sorted)
        Cr = decode_blocks(blocksCr, qCr, height, width, rev_dc, rev_ac_sorted)

        ycbcr = np.stack([Y, Cb, Cr], axis=2)
        return ycbcr_rgb(ycbcr)




def test_real_image(path):
    img_pil = Image.open(path)

    if img_pil.mode == "1":
        img = np.array(img_pil.convert("L"))
    elif img_pil.mode == "L":
        img = np.array(img_pil)
    else:
        img = np.array(img_pil.convert("RGB"))

    q = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    encode_image(img, 5, q, "compressed_data.bin")
    img_rec = decode_image("compressed_data.bin")
    if img_rec.ndim == 2 and np.max(img_rec) <= 1:
        Image.fromarray(img_rec * 255).convert("1").save("decoded.png")
    else:
        Image.fromarray(img_rec).save("decoded.png")

test_real_image("lena.png")



