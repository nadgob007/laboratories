import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.io import imshow, show, imread, imsave
from skimage.metrics import peak_signal_noise_ratio     # PSNR
from scipy import fft, ifft

import cv2
from scipy.signal import convolve2d
from io import BytesIO
from PIL import Image
from lab2 import *


# Данное искажение заключается в вырезании из носителя информации
# размерами 𝑁1 × 𝑁2 прямоугольной области с теми же пропорциями,
# начинающейся в точке с координатами (0,0) и составляющей долю 𝜗 от его
# площади. Оставшаяся часть заменяется значениями из исходного контейнера
def cut(Cw, C):
    p = []
    v_min = 0.2
    v_max = 0.9
    w = Cw.shape[0]
    h = Cw.shape[1]
    s = w*h

    result = []
    j = 0
    for v in np.arange(v_min, v_max, 0.1):
        Cw_ = []
        a = int((s*v)**(1/2))
        Cw_ = np.zeros((h, w, 3))
        for i in range(0, h):
            for j in range(0, w):
                if i < a and j < a:
                    Cw_[i][j] = Cw[i][j]
                else:
                    Cw_[i][j] = C[i][j]
        p.append(calculation_p(Cw_)[:])
        result = Cw_
    return p


# В данном искажении необходимо произвести поворот изображения на
# некоторый угол 𝜑 с обрезкой полученного изображения таким образом,
# чтобы оно сохранило свой размер.
def rotation(Cw):
    p = []
    (h, w) = Cw.shape[:2]
    center = (int(w / 2), int(h / 2))

    for fi in np.arange(1, 98.9, 8.9):
        rotation_matrix = cv2.getRotationMatrix2D(center, int(-fi), 1)
        Cw_ = cv2.warpAffine(Cw, rotation_matrix, (w, h))
        p.append(calculation_p(Cw_))

    return p


# Заключается в следующем преобразовании входного изображения:
# 𝐶𝑊̃(𝑛1,𝑛2)=𝐶𝑊(𝑛1,𝑛2)+𝐴(𝐶𝑊(𝑛1,𝑛2)−𝐶𝑠𝑚𝑜𝑜𝑡ℎ𝑊(𝑛1,𝑛2)),
# где 𝐶𝑠𝑚𝑜𝑜𝑡ℎ𝑊 – результат усреднения 𝐶𝑊 в окне размерами 𝑀×𝑀 (искаже-ние 5 текущего списка),
# а 𝐴>0 – коэффициент усиления разностного изоб-ражения.
def sharpen(Cw):
    p = []
    a = 5
    for m in range(3, 15, 2):
        filter_kernel = np.ones((m, m))
        # filter_kernel *= 1/(m*m)
        for channel_num in range(3):
            Cw_ = np.zeros(Cw.shape)
            channel = get_channel(Cw, channel_num)
            Cw_smooth = convolve2d(channel, filter_kernel * 1/(m*m), mode='same', boundary='fill', fillvalue=0)
            channel = channel + a*(channel - Cw_smooth)

            for i in range(len(Cw)):
                for j in range(len(Cw)):
                    Cw_[i][channel_num] = channel[i][j]
            p.append(calculation_p(Cw_))

        # # Отображаем
        # fig = plt.figure(figsize=(9, 3))
        # fig.add_subplot(1, 3, 1)
        # plt.title(f'Исходное')
        # imshow(Cw)
        #
        # fig.add_subplot(1, 3, 2)
        # plt.title(f'Окно размером {m}')
        # imshow(Cw_, cmap='gray')
        # show()

    return p


# Искажение заключается в сохранении носителя информации в фор-мате JPEG
# и последующем восстановлении его в формате без потерь.
# Параметром является показатель качества JPEG-файла 𝑄𝐹, изменяе-мый в пределах от 1 до 100.
def jpeg(IMAGE_FILE):
    img = Image.open(IMAGE_FILE)
    p = []
    for qf in range(30, 90, 10):
        # Создаём строку буфер
        buffer = BytesIO()
        img.save(buffer, "JPEG", quality=qf)

        # Запишем, чтоб проверить что работает
        with open("./1.jpg", "wb") as handle:
            handle.write(buffer.getbuffer())

        Cw_ = io.imread("1.jpg", as_gray=False)
        p.append(calculation_p(Cw_))
    return p


def calculation_p(Cw_2, beta_mse=False):
    C = io.imread("baboon.tif", as_gray=False)
    # Генерация ЦВЗ-последовательности, распределенных по нормальному закону  (C * 3/4) * 1/2 = 98304
    size = int((C.shape[0] * C.shape[0] * 3 / 4) * 1 / 2)
    np.random.seed(1)
    W = np.random.normal(0, 1, size)
    a = 2

    # f - матрица признаков носителя
    f = []
    result = np.zeros(C.shape)
    for i in range(3):
        channel = get_channel(C, i)

        # ДПФ исходного контейнера
        C_fft = fft.fft(channel)
        f.append(C_fft)

        # Адитивное встраивание Cw = C + a*W. fw - Матрица признаков носителя информации
        Cw, map = inserting2(C_fft, W, a)

        if beta_mse:
            # Уменьшаем визуальные искажения используя  beta_mse
            beta = roll(channel, 9)
            C_ = np.abs(fft.ifft(Cw) * beta + channel * (1 - beta))
        else:
            C_ = np.abs(fft.ifft(Cw))

        result[:, :, i] = C_.astype(int)

    # Тип массива меняем на int
    result = np.int_(result)
    # Сохраняем изображение
    # io.imsave("baboon_with_watermark.png", result)

    # Cw = io.imread("baboon_with_watermark_distorted.png", as_gray=False)
    Cw = Cw_2
    ps = []
    for i in range(3):
        channel = get_channel(Cw, i)

        # ДПФ полученного контейнера. fw_ - матрица признаков принятого носителя
        fw_ = fft.fft(channel)

        # Извлекаем ЦВЗ. omega_- матрица признаков извлеченной информации
        omega_ = extraction2(fw_, f[i], a)

        # omega - матрица признаков встраиваемой информации
        omega = fft.fft(W)

        # Детектирование ЦВЗ. p - функция близости
        p = np.sum(omega * omega_) / ((np.sum(np.power(omega, 2)) ** 0.5) * (np.sum(np.power(omega_, 2)) ** 0.5))
        ps.append(np.abs(p))

    return ps


def graphic(fig, num, name, arr, red, green, blue):
    fig.add_subplot(2, 4, num+1)
    plt.title(f'{name}')
    plt.plot(arr, red, 'o-r', alpha=0.7, label="first", lw=5, mec='b', mew=2, ms=10)
    plt.plot(arr, green, 'o-g', alpha=0.7, label="first", lw=5, mec='b', mew=2, ms=10)
    plt.plot(arr, blue, 'o-b', alpha=0.7, label="first", lw=5, mec='b', mew=2, ms=10)
    plt.legend()
    plt.grid(True)


if __name__ == '__main__':
    # Загружаем изображение
    image = io.imread("baboon_with_watermark.png", as_gray=False)

    p_cut = cut(image, io.imread("baboon.tif", as_gray=False))
    print(f'Cut:{p_cut}')
    p_rotation = rotation(image)
    print(f'Rotation:{p_rotation}')
    p_sharpen = sharpen(image)
    print(f'Sharpen:{p_sharpen}')
    p_jpeg = jpeg("baboon_with_watermark.png")
    print(f'Jpeg:{p_jpeg}')
    mesurments = [p_cut, p_rotation, p_sharpen, p_jpeg]

    # Отображаем
    fig = plt.figure(figsize=(16, 8))
    names = ['cut', 'rotation', 'sharpen', 'jpeg']
    for j in range(4):
        red = []
        green = []
        blue = []
        current = mesurments[j]
        arr = [i for i in range(len(current))]
        for i in range(len(current)):
            red.append(current[i][0])
            green.append(current[i][1])
            blue.append(current[i][2])
        graphic(fig, j, names[j], arr, red, green, blue)

# Загружаем изображение для  beta:MSE
    image = io.imread("baboon_with_watermark_betamse.png", as_gray=False)

    p_cut = cut(image, io.imread("baboon.tif", as_gray=False))
    print(p_cut)
    p_rotation = rotation(image)
    print(p_rotation)
    p_sharpen = sharpen(image)
    print(p_sharpen)
    p_jpeg = jpeg("baboon_with_watermark_betamse.png")
    print(p_jpeg)
    mesurments = [p_cut, p_rotation, p_sharpen, p_jpeg]

    # Отображаем

    names = ['betaMSE\ncut', 'betaMSE\nrotation', 'betaMSE\nsharpen', 'betaMSE\njpeg']
    for j in range(4):
        red = []
        green = []
        blue = []
        current = mesurments[j]
        arr = [i for i in range(len(current))]
        for i in range(len(current)):
            red.append(current[i][0])
            green.append(current[i][1])
            blue.append(current[i][2])
        graphic(fig, j+4, names[j], arr, red, green, blue)
    show()
