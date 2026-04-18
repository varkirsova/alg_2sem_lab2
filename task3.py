import numpy as np
from PIL import Image



def downsample(img, k=2):
    h, w = img.shape[:2]

    if h % k != 0 or w % k != 0:
        h -= h % k
        w -= w % k
        img = img[:h, :w]

    if img.ndim == 2:
        return img.reshape(h//k, k, w//k, k).mean(axis=(1,3))

    return img.reshape(h//k, k, w//k, k, -1).mean(axis=(1,3))

def upsampling(img, k=2):
    if img.ndim == 2:
        return np.kron(img, np.ones((k, k)))

    return np.kron(img, np.ones((k, k, 1)))




# img = Image.open("test_files/lena.png")
# imag = np.array(img, dtype=np.uint8)
# down = downsample(imag, 4)
# up = upsampling(down, 4)
# up_img = Image.fromarray(up)
# up_img.show()
# print(down.shape)
# print(up.shape)
# print(imag.size)
# print(down.size)
# print(up.size)
#
# down_img = Image.fromarray(down)
# down_img.save("lena_downsampe.png")
# up_img = Image.fromarray(up)
# up_img.save("lena_upsample.png")


#3
def interpol(x1, y1, x2, y2, x):
    if x1 == x2:
        raise ValueError("Иксы должны быть разные")
    return y1 + ((y2-y1)/(x2-x1))*(x-x1)


#4
def lin_spline(xi, yi, x):
    if x < xi[0] or x > xi[-1]:
        raise ValueError("ошибка диапазона")
    for i in range(len(xi)-1):
        if xi[i] <= x <= xi[i+1]:
            x1, x2 = xi[i], xi[i+1]
            y1, y2 = yi[i], yi[i+1]
            return y1 + ((y2-y1)/(x2-x1))*(x-x1)
    return yi[-1]

xs = [0, 1, 2, 3]
ys = [0, 2, 1, 3]

# print(lin_spline(xs, ys, 0.5))

#5

def bil_interpol(x1, y1, x2, y2, z11, z21, z12, z22, x, y):
    if x1 == x2:
        return interpol(y1, z11, y2, z12, y)# просто линейная интерполяция по y
    if y1 == y2:
        return interpol(x1, z11, x2, z21, x) # по x
    f1 = interpol(x1, z11, x2, z21, x)
    f2 = interpol(x1, z12, x2, z22, x)

    return interpol(y1, f1, y2, f2, y)


