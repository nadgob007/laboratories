import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.io import imshow, show, imread, imsave
from skimage.metrics import peak_signal_noise_ratio     # PSNR
from scipy import fft, ifft
from scipy.signal import convolve2d

import cv2


# Возвращает цветное изображени с только с зеленым каналом и изображение в оттенках серого с значениями зеленого канала
def get_channel(img, channel_num):  # 0-red, 1-green, 2-blue
    channel = img[:, :, channel_num]
    return channel


# Данное искажение заключается в вырезании из носителя информации
# размерами 𝑁1 × 𝑁2 прямоугольной области с теми же пропорциями,
# начинающейся в точке с координатами (0,0) и составляющей долю 𝜗 от его
# площади. Оставшаяся часть заменяется значениями из исходного контейнера
def cut(Cw, C=0):
    Cw_ = []
    v_min = 0.2
    v_max = 0.9
    w = Cw.shape[0]
    h = Cw.shape[1]
    s = w*h

    for v in np.arange(v_min, v_max, 0.1):
        print(v)
        print(f'Площадь: {s*v}')
        a = int((s*v)**(1/2))
        print(a)

        Cw_ = np.zeros((h, w))
        for i in range(0, h):
            for j in range(0, w):
                if i < (a) and j < (a):
                    Cw_[i][j] = Cw[i][j]
                else:
                    Cw_[i][j] = C
    return Cw_


# В данном искажении необходимо произвести поворот изображения на
# некоторый угол 𝜑 с обрезкой полученного изображения таким образом,
# чтобы оно сохранило свой размер.
def rotation(Cw):
    (h, w) = Cw.shape[:2]
    center = (int(w / 2), int(h / 2))

    for fi in np.arange(1, 98.9, 8.9):
        rotation_matrix = cv2.getRotationMatrix2D(center, int(-fi), 1)
        Cw_ = cv2.warpAffine(Cw, rotation_matrix, (w, h))

    return Cw_


# Заключается в следующем преобразовании входного изображения:
# 𝐶𝑊̃(𝑛1,𝑛2)=𝐶𝑊(𝑛1,𝑛2)+𝐴(𝐶𝑊(𝑛1,𝑛2)−𝐶𝑠𝑚𝑜𝑜𝑡ℎ𝑊(𝑛1,𝑛2)),
# где 𝐶𝑠𝑚𝑜𝑜𝑡ℎ𝑊 – результат усреднения 𝐶𝑊 в окне размерами 𝑀×𝑀 (искаже-ние 5 текущего списка),
# а 𝐴>0 – коэффициент усиления разностного изоб-ражения.
def sharpen(Cw):
    Cw_smooth = []



    # Отображаем
    fig = plt.figure(figsize=(9, 3))
    fig.add_subplot(1, 3, 1)
    plt.title(f'Исходное:{h}x{w}')
    imshow(Cw)

    fig.add_subplot(1, 3, 2)
    plt.title(f'Повернутое на {int(-fi)}')
    imshow(Cw_)
    show()

    return Cw_


# Искажение заключается в сохранении носителя информации в фор-мате JPEG
# и последующем восстановлении его в формате без потерь.
# Параметром является показатель качества JPEG-файла 𝑄𝐹, изменяе-мый в пределах от 1 до 100.
def jpeg():
    return


if __name__ == '__main__':
    # Загружаем изображение
    image = io.imread("baboon.tif", as_gray=True)
    rotation(image)

