import math

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np

def blocks8(img: np.ndarray):
    h, w = img.shape[:2]
    blocks = []
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            block = img[y:y+8, x:x+8]
            if block.shape[0] == 8 and block.shape[1] == 8:
                blocks.append(block)
                continue
            mean_val = block.mean(axis=(0, 1), dtype=float)
            if img.ndim == 2:
                added = np.full((8, 8), mean_val, dtype=img.dtype)
                added[:block.shape[0], :block.shape[1]] = block
            else:
                added = np.full((8, 8, img.shape[2]), mean_val, dtype=img.dtype)
                added[:block.shape[0], :block.shape[1], :] = block

            blocks.append(added)

    return blocks


# imag = Image.open("test_files/color2.png")
# img = np.array(imag, dtype=np.uint8)
#
# padded = blocks8(img)
# print("Исходная форма:", img.shape)
# print("После padding:", padded.shape)
# print(padded)


#2

def dcp(block):
    koef = np.zeros((8, 8), dtype=np.float64)
    block = block.astype(np.float64)
    n, m = block.shape

    for p in range(n): #частота вертикаль
        for q in range(m): # частота горизонталь
            p_alf = 1 / math.sqrt(n) if p == 0 else math.sqrt(2/n)
            q_alf = 1 / math.sqrt(m) if q == 0 else math.sqrt(2/m)

            s = 0.0
            for x in range(n):
                for y in range(m):
                    s += block[x, y] * \
                         math.cos((2 * x + 1) * p * math.pi / (2*n)) * \
                         math.cos((2 * y + 1) * q * math.pi / (2*m))
            koef[p,q] = p_alf * q_alf * s
    return koef
#3
def dcp_reverse(koef):
    koef = koef.astype(np.float64)
    n, m = koef.shape
    a = np.zeros((n, m), dtype=np.float64)

    for x in range(n):
        for y in range(m):
            s = 0.0
            for p in range(n):
                for q in range(m):
                    p_alf = 1 / math.sqrt(n) if p == 0 else math.sqrt(2 / n)
                    q_alf = 1 / math.sqrt(m) if q == 0 else math.sqrt(2 / m)

                    s += p_alf * q_alf * koef[p, q] * \
                         math.cos((2 * x + 1) * p * math.pi / (2 * n)) * \
                         math.cos((2 * y + 1) * q * math.pi / (2 * m))

            a[x, y] = s

    return a


# def test_dcp(path):
#     img = np.array(Image.open(path), dtype=np.float64)
#     H, W = img.shape[:2]
#
#     blocks = blocks8(img)
#
#     rec_blocks = []
#     for block in blocks:
#         if img.ndim == 2:
#             C = dcp(block)
#             block_rec = dcp_reverse(C)
#         else:
#             block_rec = np.zeros_like(block)
#             for c in range(3):
#                 C = dcp(block[:, :, c])
#                 block_rec[:, :, c] = dcp_reverse(C)
#
#         rec_blocks.append(block_rec)
#
#     padded_h = ((H + 7) // 8) * 8
#     padded_w = ((W + 7) // 8) * 8
#
#     rec = np.zeros((padded_h, padded_w) + img.shape[2:], dtype=np.float64)
#
#     idx = 0
#     for y in range(0, padded_h, 8):
#         for x in range(0, padded_w, 8):
#             rec[y:y+8, x:x+8] = rec_blocks[idx]
#             idx += 1
#
#     rec = rec[:H, :W] # исходный размер
#     diff = np.abs(rec - img)
#     print("Максимальная ошибка:", diff.max())
#     print("Средняя ошибка:", diff.mean())
#
#     plt.figure(figsize=(12, 6))
#
#     plt.subplot(1, 2, 1)
#     plt.title("Оригинал")
#     plt.imshow(img.astype(np.uint8))
#     plt.axis("off")
#
#     plt.subplot(1, 2, 2)
#     plt.title("После ДКП и обратного ДКП")
#     plt.imshow(np.clip(rec, 0, 255).astype(np.uint8))
#     plt.axis("off")
#
#     plt.show()
#
#     return rec
#
#
# # print(test_dcp("test_files/lena.png"))
#4
def matrix(n=8):
    a = np.zeros((n,n))
    for x in range(n):
        for y in range(n):
            alf = 1/math.sqrt(n) if x == 0 else math.sqrt(2/n)
            a[x,y] = alf * math.cos((2*y+1) * x * math.pi / (2*n))
    return a

def dct_matr(block):
    block = block.astype(np.float64)
    a = matrix(block.shape[0])
    return a @ block @ a.T

def dct_reverse_matr(k):
    a = matrix(k.shape[0])
    return a.T @ k @ a
#
#
# def test_dct_on_image(path):
#     img = np.array(Image.open(path), dtype=np.float64)
#     H, W = img.shape[:2]
#     pad_h = (8 - H % 8) % 8
#     pad_w = (8 - W % 8) % 8
#
#     padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
#     rec = np.zeros_like(padded)
#
#     for c in range(3):  # по каналам
#         for i in range(0, padded.shape[0], 8):
#             for j in range(0, padded.shape[1], 8):
#                 block = padded[i:i+8, j:j+8, c]
#                 C = dct_matr(block)
#                 block_rec = dct_reverse_matr(C)
#                 rec[i:i+8, j:j+8, c] = block_rec
#
#     rec = rec[:H, :W]
#     diff = np.abs(img.astype(np.float64) - rec.astype(np.float64))
#     print("Максимальная ошибка:", diff.max())
#     print("Средняя ошибка:", diff.mean())
#
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.title("Исходное")
#     plt.imshow(img.astype(np.uint8))
#     plt.axis("off")
#
#     plt.subplot(1, 2, 2)
#     plt.title("После DCT и IDCT через матрицы")
#     plt.imshow(np.clip(rec, 0, 255).astype(np.uint8))
#     plt.axis("off")
#
#     plt.show()
#
#
# # print(test_dct_on_image("test_files/lena.png"))



#5
def quant(c, q):
    c = np.asarray(c, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    cq = np.round(c / q)
    return cq.astype(np.int32)

#6

def normaliz_koef(cq, q):
    cq = np.asarray(cq, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    c = cq * q
    return c
