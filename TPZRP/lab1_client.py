import socket
import os
import struct   # Server

from skimage.util import random_noise   # Noise
import numpy as np
import cv2
import matplotlib.pyplot as plt


def add_impulse_noise(img):
    noise_image = np.array(img.copy())
    noise = random_noise(np.full(noise_image.shape, -1), mode="s&p", amount=1)

    for index, _ in np.ndenumerate(noise):
        if noise[index] == 0 or noise[index] == 1:
            noise_image[index] = noise[index]

    fig = plt.figure(figsize=(10, 30))

    fig.add_subplot(1, 3, 1)
    plt.title('Source ing')
    imshow(image)

    fig.add_subplot(1, 3, 2)
    plt.title('Impulse noise')
    imshow(noise)

    fig.add_subplot(1, 3, 3)
    plt.title('Img with noise')
    imshow(noise_image)

    return noise_image


# def send_file(sck: socket.socket, filename):
#     # Получение размера файла.
#     filesize = os.path.getsize(filename)
#     # В первую очередь сообщим серверу,
#     # сколько байт будет отправлено.
#     sck.sendall(struct.pack("<Q", filesize))
#     # Отправка файла блоками по 1024 байта.
#     with open(filename, "rb") as f:
#         read_bytes = f.read(1024)
#         while read_bytes:
#             sck.sendall(read_bytes)
#             read_bytes = f.read(1024)
#
#
# with socket.create_connection(("localhost", 6190)) as conn:
#
#     original_image = cv2.imread("goldhill.tif", cv2.IMREAD_GRAYSCALE)
#     file_is_saved = cv2.imwrite("goldhill_noise.tif", original_image)
#
#     print("Подключение к серверу.")
#     print("Передача файла...")
#     add_impulse_noise(original_image)
#     send_file(conn, "goldhill_noise.tif")
#     print("Отправлено.")
# print("Соединение закрыто.")
