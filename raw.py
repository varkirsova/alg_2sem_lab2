import struct
from PIL import Image
import os
import numpy as np

def save_raw(img: Image.Image, filename: str, colorspace: int=0):
    if isinstance(img, np.ndarray):
        h, w, c = img.shape
        data = img.astype(np.uint8).tobytes()
        img_type = 2
        colorspace = 2  # ycbcr
    else:
        mode = img.mode
        w, h = img.size
        data = img.tobytes()

        if mode == "1":
            img_type = 0
            colorspace = 0 # нет цвет пространства
        elif mode == "L":
            img_type = 1
            colorspace = 0 # нет цвет пространства
        elif mode == "RGB":
            img_type = 2
            colorspace = 1 # rgb
        else:
            raise ValueError("ошибка типа")

    with open(filename, "wb") as f:
        f.write(struct.pack("B", img_type))
        f.write(struct.pack("B", colorspace))
        f.write(struct.pack("I", w))
        f.write(struct.pack("I", h))
        f.write(data)

# img = Image.open("test_files/image_gray.png")
# save_raw(img, "test_files/image_gray.raw")
#
# def size(path):
#     return os.path.getsize(path)

# raw = "test_files/image_gray.raw"
# orig = "test_files/image_gray.png"
# raw_size = size(raw)
# orig_size = size(orig)
#
# print("RAW:", raw_size)
# print("orig:", orig_size)
# print("Коэффициент:", raw_size/orig_size)

# with open("lena_ycbcr.raw", "rb") as f:
#     header = f.read(10)
#     print(list(header))


from PIL import Image

img = Image.open("test_files/colour.png")

# 8-битный grayscale
gray = img.convert("L")
gray.save("gray.png")

# 1-битное BW (пороговое)
bw = gray.point(lambda x: 255 if x > 127 else 0).convert("1")
bw.save("bw.png")

# 1-битное BW с дизерингом
bw_dither = gray.convert("1")  # по умолчанию Floyd–Steinberg
bw_dither.save("bw_dither.png")
