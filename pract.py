import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from task2_7 import encode_image

def experiment_size_vs_quality(image_path, q_base, out_prefix="test"):
    img = np.array(Image.open(image_path).convert("L"))

    qualities = list(range(10, 99, 10))
    sizes = []

    for q_value in qualities:
        filename = f"{out_prefix}_q{q_value}.bin"

        encode_image(img, q_value, q_base, filename)

        size = os.path.getsize(filename)
        sizes.append(size)

        print(f"q={q_value} -> {size} bytes")


    plt.figure()
    plt.plot(qualities, sizes, marker='o')
    plt.xlabel("Значение quality")
    plt.ylabel("Размер сжатого в байтах")
    plt.grid(True)

    plt.show()


q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])


experiment_size_vs_quality("bw.png", q)