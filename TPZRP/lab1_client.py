import socket
import os
import struct   # Server

from skimage.util import random_noise   # Noise
import numpy as np
import cv2


def add_impulse_noise(image, param):
    noise_image = random_noise(image, mode='s&p', amount=param)
    noise = ((noise_image - image) / 2) + 0.5
    return noise_image


def send_file(sck: socket.socket, filename):
    # Получение размера файла.
    file_size = os.path.getsize(filename)
    # В первую очередь сообщим серверу,
    # сколько байт будет отправлено.
    sck.sendall(struct.pack("<Q", file_size))
    # Отправка файла блоками по 1024 байта.
    with open(filename, "rb") as f:
        read_bytes = f.read(1024)
        while read_bytes:
            sck.sendall(read_bytes)
            read_bytes = f.read(1024)


number_of_files = 10
for i in range(1, number_of_files+1):
    with socket.create_connection(("localhost", 6190)) as conn:
        original_image = cv2.imread("goldhill.tif", cv2.IMREAD_GRAYSCALE)
        print("Подключение к серверу.")
        print("Передача файла...")
        noise_img = add_impulse_noise(original_image, 0.3)

        file_is_saved = cv2.imwrite("goldhill_noise.tif", (255 * noise_img).astype(np.uint8))
        send_file(conn, "goldhill_noise.tif")
        print(f"Отправлено {i}/{number_of_files}")
print("Соединение закрыто.")
