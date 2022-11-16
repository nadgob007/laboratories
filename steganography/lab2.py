import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.io import imshow, show, imread, imsave
from skimage.metrics import peak_signal_noise_ratio     # PSNR
from scipy import fft, ifft
from scipy.signal import convolve2d


def psnr(omega, omega_):
    e = (np.sum((omega - omega_) ** 2)) / (len(omega) * len(omega[0]))
    p = 10 * np.log10(255 ** 2 / e)
    return p


# Возвращает цветное изображени с только с зеленым каналом и изображение в оттенках серого с значениями зеленого канала
def get_channel(img, channel_num):  # 0-red, 1-green, 2-blue
    channel = img[:, :, channel_num]
    return channel


#   Адитивное встраивание Cw = C + a*omega плюсиком
def inserting(C, omega, a):
    size = 512
    count = 0
    map = np.zeros(C.shape)
    for i in range(0, size):
        for j in range(0, size):
            if (i < (size/4) or i >= (size * 3/4)) and (j < (size/4) or j >= (size * 3/4)):
                continue
            else:
                C[i][j] = complex(C[i][j].real + a * omega[count], C[i][j].imag)
                map[i][j] = 1
                count += 1
        print(i)
    return C, map


#   Адитивное встраивание Cw = C + a*omega . половина от плюсика
def inserting2(C, omega, a):
    size = 512
    count = 0
    map = np.zeros(C.shape)
    pull1 = 16384
    pull2 = pull1*4
    for i in range(0, size):
        for j in range(0, size):
            condition1 = i < (size/4)
            condition2 = i >= (size * 3/4)
            condition3 = j < (size/4)
            condition4 = j >= (size * 3/4)

            if (i < (size/4) or i >= (size * 3/4)) and (j < (size/4) or j >= (size * 3/4)):
                continue
            else:
                if (not condition3) and (not condition4) and (condition1 or condition2):
                    if pull1 == 0:
                        continue

                    C[i][j] = complex(C[i][j].real + a * omega[count], C[i][j].imag)
                    map[i][j] = 1
                    count += 1

                    pull1 -= 1

                if (not condition1) and (not condition2):
                    if pull2 == 0:
                        continue

                    C[i][j] = complex(C[i][j].real + a * omega[count], C[i][j].imag)
                    map[i][j] = 1
                    count += 1

                    pull2 -= 1
                if count == 81919:
                    pull1 = 16384
    return C, map


#   Извлечение цифрового водяного знака
def extraction2(fw_, f, a):
    size = 512
    count = 0
    omega_ = np.zeros(shape=98304, dtype=complex)
    pull1 = 16384
    pull2 = pull1*4
    for i in range(0, size):
        for j in range(0, size):
            condition1 = i < (size/4)
            condition2 = i >= (size * 3/4)
            condition3 = j < (size/4)
            condition4 = j >= (size * 3/4)

            if (i < (size/4) or i >= (size * 3/4)) and (j < (size/4) or j >= (size * 3/4)):
                continue
            else:
                if (not condition3) and (not condition4) and (condition1 or condition2):
                    if pull1 == 0:
                        continue
                    omega_[count] = (fw_[i][j].real - f[i][j].real) / a
                    count += 1
                    pull1 -= 1

                if (not condition1) and (not condition2):
                    if pull2 == 0:
                        continue

                    omega_[count] = (fw_[i][j].real - f[i][j].real) / a
                    count += 1

                    pull2 -= 1
                if count == 81919:
                    pull1 = 16384
    return omega_


def calc_a(a, beta_mse=False):
    C = io.imread("baboon.tif", as_gray=False)
    # Генерация ЦВЗ-последовательности, распределенных по нормальному закону  (C * 3/4) * 1/2 = 98304
    size = int((C.shape[0] * C.shape[0] * 3 / 4) * 1 / 2)
    np.random.seed(1)
    omega = np.random.normal(0, 1, size)
    a = a

    # f - матрица признаков носителя
    f = []
    result = np.zeros(C.shape)
    for i in range(3):
        channel = get_channel(C, i)

        # ДПФ исходного контейнера
        C_fft = fft.fft(channel)
        f.append(C_fft)

        # Адитивное встраивание Cw = C + a*omega. fw - Матрица признаков носителя информации
        Cw, map = inserting2(C_fft, omega, a)

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
    io.imsave("baboon_with_watermark_betamse.png", result)
    Cw = io.imread("baboon_with_watermark.png", as_gray=False)
    psnr = peak_signal_noise_ratio(C, Cw)

    ps = []
    for i in range(3):
        channel = get_channel(Cw, i)

        # ДПФ полученного контейнера. fw_ - матрица признаков принятого носителя
        fw_ = fft.fft(channel)

        # Извлекаем ЦВЗ. omega_- матрица признаков извлеченной информации
        omega_ = extraction2(fw_, f[i], a)

        # omega - матрица признаков встраиваемой информации
        

        # Детектирование ЦВЗ. p - функция близости
        p = np.sum(omega*omega_)/( (np.sum(np.power(omega, 2))**0.5) * (np.sum(np.power(omega_, 2))**0.5) )
        ps.append(np.abs(p))

    return psnr, min(ps)


# Прохождение по изображению усредняющим окном 9х9. Возвращает beta
def roll(a, b):
    mat = np.zeros(a.shape)

    for i, row in enumerate(a):     # Перебираем строки в матрице
        for j, val in enumerate(row):   # Перебираем столбцы в строке

            wind = np.array(0)
            wind = np.delete(wind, 0)
            # Проверяем, какие элементы попадают в окно
            for w_i in range(b):
                for w_j in range(b):

                    if (i - int(b/2) + w_i) < 0 or (j - int(b/2) + w_j) < 0 or (i - int(b/2) + w_i) >= a.shape[0] or (j - int(b/2) + w_j) >= a.shape[0]:
                        continue
                    else:
                        current = a[i - int(b / 2) + w_i][j - int(b / 2) + w_j]
                    wind = np.append(wind, current)

            # Считаем среднеквадратическое отклонение
            std = np.std(wind)
            mat[i][j] = np.std(wind)/max(wind)
        print(f"{i}/{a.shape[0]}")
    return mat


if __name__ == '__main__':
    # Загружаем изображение
    image = io.imread("baboon.tif", as_gray=False)

    # Генерация ЦВЗ-последовательности, распределенных по нормальному закону  (C * 3/4) * 1/2 = 98304
    size = int((image.shape[0] * image.shape[0] * 3 / 4) * 1 / 2)
    np.random.seed(4)
    omega = np.random.normal(0, 1, size)
    omega2 = np.random.normal(0, 1, size)
    a = 2

    # f - матрица признаков носителя
    f = []
    result = np.zeros(image.shape)
    for i in range(3):
        channel = get_channel(image, i)

        # ДПФ исходного контейнера
        C_fft = fft.fft(channel)
        f.append(C_fft)

        # Адитивное встраивание Cw = C + a*omega
        fw, map = inserting2(C_fft, omega, a)
        Cw = np.real(fft.ifft(fw))

        result[:, :, i] = Cw#.astype(int)

    # Тип массива меняем на int
    result = np.int_(result)
    # Сохраняем изображение
    io.imsave("baboon_with_watermark.png", result)

    # Отображаем
    fig = plt.figure(figsize=(9, 3))
    fig.add_subplot(1, 3, 1)
    plt.title('Исходное')
    imshow(image)

    fig.add_subplot(1, 3, 2)
    plt.title('Карта встраивания')
    imshow(map)

    fig.add_subplot(1, 3, 3)
    plt.title('Контейнер со встроенным знаком')
    imshow(result)

    show()

    # Загружаем контейнер с водяным знаком
    img = io.imread("baboon_with_watermark.png", as_gray=False)


    channels = ["Red", "Green", "Blue"]
    for i in range(3):
        channel = get_channel(img, i)

        # Обратное ДПФ от носителя информации . fw_ - матрица признаков принятого носителя
        fw_ = fft.fft(channel)

        # Извлекаем ЦВЗ. omega_- матрица признаков извлеченной информации
        omega_ = extraction2(fw_, f[i], a)

        # omega - матрица признаков встраиваемой информации
        omega = omega

        # Детектирование ЦВЗ. p - функция близости
        p = np.sum(omega*omega_)/( (np.sum(np.power(omega, 2))**0.5) * (np.sum(np.power(omega_, 2))**0.5) )
        #p = np.sum(omega2*omega_)/( (np.sum(np.power(omega2, 2))**0.5) * (np.sum(np.power(omega_, 2))**0.5) )
        print(f"Близость[{channels[i]}]: {p}")
        print(f"Близость[{channels[i]}]: {np.abs(p)}\n")


    # max_p = 0.0
    # max_psnr = 0.0
    # for i in np.arange(1, 2, 0.05):
    #     psnr, p = calc_a(i, False)
    #     if psnr > 30:
    #         if max_p < p:
    #             max_p = p
    #             a = i
    #             max_psnr = psnr
    #     print(i)
    # print(f"Значение p: {max_p}")
    # print(f"Значение PSNR: {max_psnr}")
    # print(f"Значение a: {a}")

# 0.08257754851076042 seed(1)
# 52.846426625042184
# 3.5

# Значение p: 0.06703117574558583
# Значение PSNR: 52.84528183396575
# Значение a: 0.1

    # С beta_mse
    # max_p = 0.0
    # max_psnr = 0.0
    # for i in np.arange(0.05, 5, 0.05):
    #     psnr, p = calc_a(i, True)
    #     if psnr > 30:
    #         if max_p < p:
    #             max_p = p
    #             a = i
    #             max_psnr = psnr
    #     print(i)
    # print(f"Значение p: {max_p}")
    # print(f"Значение PSNR: {max_psnr}")
    # print(f"Значение a: {a}")
# Значение p: 0.08257754851076042
# Значение PSNR: 53.504182849999715
# Значение a: 3.5