import numpy as np
from PIL import Image

from raw import save_raw

img = Image.open("lena.png")
rgb = np.array(img, dtype=np.uint8)
#
def rgb_ycbcr(rgb: np.ndarray):
    rgb = rgb.astype(np.float32)
    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]

    y = 0.299*r + 0.587*g + 0.114*b
    cb = -0.1687*r - 0.3313*g + 0.5*b + 128
    cr = 0.5*r - 0.4187*g - 0.0813*b+128

    y = np.clip(np.round(y), 0, 255)
    cb = np.clip(np.round(cb), 0, 255)
    cr = np.clip(np.round(cr), 0, 255)

    ycbcr = np.stack((y, cb, cr), axis=2)
    ycbcr = ycbcr.astype(np.uint8)
    return ycbcr


def ycbcr_rgb(ycbcr: np.ndarray):
    ycbcr = ycbcr.astype(np.float32)
    y = ycbcr[:, :, 0]
    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]

    r = y + 1.402*(cr-128)
    g = y - 0.34414*(cb-128)-0.71414*(cr-128)
    b = y + 1.772*(cb-128)

    r = np.clip(np.round(r), 0, 255)
    g = np.clip(np.round(g), 0, 255)
    b = np.clip(np.round(b), 0, 255)

    rgb = np.stack((r, g, b), axis=2)
    rgb = rgb.astype(np.uint8)
    return rgb


# ycbcr = rgb_ycbcr(rgb)
# rgb_back = ycbcr_rgb(ycbcr)
# # save_raw(ycbcr, "lena_ycbcr.raw")
# Image.fromarray(rgb_back).save("lena_new.png")


# diff = np.abs(rgb.astype(np.int16) - rgb_back.astype(np.int16))
# print("Максимальная разница пикселей", diff.max())
# print("Средняя разница", diff.mean())




# import numpy as np
#
# def rgb_ycbcr(img, inverse=False):
#     img = img.astype(np.float64)
#
#     if not inverse:
#         R = img[:,:,0]
#         G = img[:,:,1]
#         B = img[:,:,2]
#
#         Y  =  0.299*R + 0.587*G + 0.114*B
#         Cb = -0.168736*R - 0.331264*G + 0.5*B + 128
#         Cr =  0.5*R - 0.418688*G - 0.081312*B + 128
#
#         out = np.zeros_like(img)
#         out[:,:,0] = Y
#         out[:,:,1] = Cb
#         out[:,:,2] = Cr
#         return out.astype(np.uint8)
#
#     else:
#         Y  = img[:,:,0]
#         Cb = img[:,:,1] - 128
#         Cr = img[:,:,2] - 128
#
#         R = Y + 1.402 * Cr
#         G = Y - 0.344136 * Cb - 0.714136 * Cr
#         B = Y + 1.772 * Cb
#
#         out = np.zeros_like(img)
#         out[:,:,0] = np.clip(R, 0, 255)
#         out[:,:,1] = np.clip(G, 0, 255)
#         out[:,:,2] = np.clip(B, 0, 255)
#         return out.astype(np.uint8)
