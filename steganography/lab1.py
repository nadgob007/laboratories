import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread, imshow, show

# 11) 3.3 Green-4


# Возвращает цветное изображени с только с зеленым каналом и изображение в оттенках серого с значениями зеленого канала
def get_channel(img, channel_num):  # 0-red, 1-green, 2-blue
    result_img = img.copy()
    green_img = np.zeros(img.shape)
    green_img[:, :, channel_num] = result_img[:, :, channel_num]
    return green_img.astype(int), result_img[:, :, channel_num]


# Возвращает указанную битовую поверхность
def get_plane(img, num=4):
    sum1 = 0
    ck_img = 0
    j = 0
    for i in range(num, 0, -1):

        if i == num:
            ck_img = img % (2 ** i)
            sum1 = ck_img // (2 ** (i - 1))
        else:
            if i % 2 == 0:
                current = ck_img % (2 ** i)
                sum1 = sum1 + current // (2 ** (j+i-1))
                ck_img = current
            else:
                current = ck_img % (2 ** i)
                sum1 = sum1 - current // (2 ** (j+i-1))
                ck_img = current
        j += i
    return sum1


# Конвертируем RGB в YCbCr
def RGB2YCbCr(img):
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    return (0.18*red) + (0.81*green) + (0.01*blue)
#     return ((77/256)*red) + ((150/256)*green) + ((29/256)*blue)


# Равномерный белый шум с диапозоном от 0 до delta-1
def get_v(shape):
    return np.random.uniform(0, delta - 1, shape).astype('uint8')


# Встраивание информации в каждом пикселе
def insert_watermark(img, watermark):
    return (img/(2*delta)).astype(int) * 2*delta + watermark*delta + get_v(img.shape)


# Извлекаем водяной знак
def extract_watermark(img_with_watermark, img):
    return img_with_watermark - (img/(2*delta)).astype(int) * 2*delta - get_v(img.shape)


if __name__ == '__main__':
    bit_plane = 4
    channel = 2
    c_img = imread('baboon.tif')
    w_img = imread('ornament.tif')/255

    fig = plt.figure(figsize=(15, 5))

    # Изображение контейнер
    fig.add_subplot(2, 5, 1)
    imshow(c_img)

    # Изображение встраимоевое
    fig.add_subplot(2, 5, 2)
    imshow(w_img)

    # Зелёный канал изображения
    green, green_channel = get_channel(c_img, 1)
    fig.add_subplot(2, 5, 3)
    imshow(green)

    # 4-ая битовая плоскость
    plane = get_plane(green_channel, 4)
    fig.add_subplot(2, 5, 4)
    imshow(plane, cmap='gray')

    # Встраивание в контейнер
    green_channel_mark = green_channel - (plane * 2**3) + (w_img * 2**3)
    c_img[:, :, 1] = green_channel_mark
    fig.add_subplot(2, 5, 5)
    imshow(c_img)

    # Изъятие водяного знака
    green, water_mark = get_channel(c_img, 1)
    fig.add_subplot(2, 5, 6)
    imshow(get_plane(water_mark, 4), cmap='gray')

    # СВИ-4 ( Y (3.12) )

    delta = 4+4*11 % 3    # 4+4*11(mod3) = 6

    # Встраивание в контейнер
    img_with_watermark = c_img.copy()
    img_with_watermark[:, :, 1] = insert_watermark(RGB2YCbCr(c_img.copy()), w_img.copy())
    fig.add_subplot(2, 5, 7)
    imshow(img_with_watermark)

    # Извлечение
    extracted_w = extract_watermark(img_with_watermark[:, :, 1], RGB2YCbCr(c_img.copy()))
    fig.add_subplot(2, 5, 8)
    imshow(extracted_w, cmap='gray')

    # Без картинки

    show()
