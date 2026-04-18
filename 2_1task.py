import numpy as np
from PIL import Image

from task3 import bil_interpol

def resize(img: np.ndarray, new_h: int, new_w: int):
    h, w = img.shape[:2]

    if img.ndim == 2:
        out = np.zeros((new_h, new_w), dtype=img.dtype)
        channels = 1
    else:
        out = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype)
        channels = img.shape[2]

    scale_x = w / new_w
    scale_y = h / new_h

    for i in range(new_h):
        for j in range(new_w):
            x = j * scale_x
            y = i * scale_y

            x1 = int(np.floor(x))
            y1 = int(np.floor(y))
            x2 = min(x1 + 1, w - 1)
            y2 = min(y1 + 1, h - 1)

            if channels == 1:
                z11 = img[y1, x1]
                z21 = img[y1, x2]
                z12 = img[y2, x1]
                z22 = img[y2, x2]

                out[i, j] = bil_interpol(
                    x1, y1, x2, y2,
                    z11, z21, z12, z22,
                    x, y
                )

            else:
                for c in range(channels):
                    z11 = img[y1, x1, c]
                    z21 = img[y1, x2, c]
                    z12 = img[y2, x1, c]
                    z22 = img[y2, x2, c]

                    out[i, j, c] = bil_interpol(
                        x1, y1, x2, y2,
                        z11, z21, z12, z22,
                        x, y
                    )

    return out.astype(img.dtype)


#
# rgb = np.array(Image.open("test_files/color.png"))
# res = resize(rgb, 800, 600)
# Image.fromarray(res).save("color_resized.png")
