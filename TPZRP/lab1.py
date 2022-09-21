import socket
import struct
import numpy as np
import cv2
from scipy import fft, ifft
from matplotlib import pyplot as plt
from skimage.io import imshow, show


def receive_file_size(sck: socket.socket):
    # Эта функция обеспечивает получение байтов,
    # указывающих на размер отправляемого файла,
    # который кодируется клиентом с помощью
    # struct.pack(), функции, которая генерирует
    # последовательность байтов, представляющих размер файла.
    fmt = "<Q"
    expected_bytes = struct.calcsize(fmt)
    received_bytes = 0
    stream = bytes()
    while received_bytes < expected_bytes:
        chunk = sck.recv(expected_bytes - received_bytes)
        stream += chunk
        received_bytes += len(chunk)
    filesize = struct.unpack(fmt, stream)[0]
    return filesize


def receive_file(sck: socket.socket, filename):
    # Сначала считываем из сокета количество
    # байтов, которые будут получены из файла.
    filesize = receive_file_size(sck)
    # Открываем новый файл для сохранения
    # полученных данных.
    with open(filename, "wb") as f:
        received_bytes = 0
        # Получаем данные из файла блоками по
        # 1024 байта до объема
        # общего количество байт, сообщенных клиентом.
        while received_bytes < filesize:
            chunk = sck.recv(1024)
            if chunk:
                f.write(chunk)
                received_bytes += len(chunk)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def recovery():
    orig = cv2.imread("goldhill.tif", cv2.IMREAD_GRAYSCALE)
    noise_img = cv2.imread(f"goldhill-received1.tif", cv2.IMREAD_GRAYSCALE)

    sum_all = fft.fft(noise_img.flatten())
    for i in range(2, 11):
        noise_img = cv2.imread(f"goldhill-received{i}.tif", cv2.IMREAD_GRAYSCALE)
        current = fft.fft(noise_img.flatten())
        sum_all += current

    y = fft.ifft(sum_all/10)
    y = moving_average(y, 4)
    y = np.abs(y)
    y = y/np.max(y)
    y = np.reshape(y, (512, 512))

    fig = plt.figure(figsize=(9, 3))
    fig.add_subplot(1, 3, 1)
    plt.title('Исходное')
    imshow(orig, cmap='gray')

    fig.add_subplot(1, 3, 2)
    plt.title('Зашумленное')
    imshow(noise_img, cmap='gray')

    fig.add_subplot(1, 3, 3)
    plt.title('Востановленное')
    imshow(y, cmap='gray')
    show()


number_of_files = 10
for i in range(1, number_of_files+1):
    with socket.create_server(("localhost", 6190)) as server:
        print("Ожидание клиента...")
        conn, address = server.accept()
        print(f"{address[0]}:{address[1]} подключен.")
        print("Получаем файл...")
        receive_file(conn, f"goldhill-received{i}.tif")
        print(f"Файл получен {i}/{number_of_files}")
print("Соединение закрыто.")

print("Восстановление...")
recovery()

