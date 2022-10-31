import numpy as np
import matplotlib.pyplot as plt


# Принимает - путь до файла. Возвращает - массив подготовленных данных
def get_data(path):
    with open(path) as f:
        strings = f.read().splitlines()

    current_string = [[]] * len(strings)
    for sting in strings:
        user = sting.split(':')[0]
        numbers = sting.split(':')[1]
        user_num = ''
        for i in user:
            if i.isdigit():
                user_num += user_num.join(i)
        user_num = int(user_num)

        current_string[user_num - 1] = [int(i) for i in numbers.split(';')]
    return current_string


# Принимает - строку пользователя, массив для матрицы Маркова, словарь с уникальными значениями. 
# Возвращает - Матрицу Маркова
def markov(current_string, result, index_by_value):
    values, amount = np.unique(current_string, return_counts=True)
    amount[np.where(values == current_string[-1])] -= 1

    for i, element in enumerate(current_string[:len(current_string) - 1]):
        result[index_by_value[current_string[i + 1]], index_by_value[element]] += 1
    result = np.divide(result, sum(result))

    return result


# Принимает - выборку, матрицу Маркова, словарь. Возвращает - вероятность существования выборки
def prob_of_existence(data, matrix, index_by_value):
    probability = 1
    for i, element in enumerate(data[:len(data) - 1]):
        destination = index_by_value[data[i + 1]]
        start = index_by_value[element]
        probability *= matrix[destination, start]
    return probability


# Принимает - выборку, матрицу Маркова, словарь, окно. Возвращает - наименьшую вероятность существования цепочки
def get_limit(current_string, matrix, index_by_value, window):
    result = []
    for i in range(max(len(current_string) - window, 1)):
        result.append(prob_of_existence(current_string[i: i + window], matrix, index_by_value))
    return min(result)


if __name__ == "__main__":
    # Импортируем данные и обрабатываем их
    training_sample = get_data('data.txt')
    data_true = get_data('data_true.txt')
    data_fake = get_data('data_fake.txt')

    # Собираем все элементы
    all_data = []
    for string in training_sample:
        for element in string:
            all_data.append(element)

    values, amounts = np.unique(all_data, return_counts=True)       # Выбираем уникальные элементы и их колличество
    result = np.ones((len(values), len(values)))                    # Массив для результата
    index_by_value = dict(zip(values, range(len(values))))          # Словарь: Уникальное значение - индекс

    window = 10  # Размер окна
    predict_true = []  # Результаты для массива без выбросов
    predict_fake = []  # Результаты для массива только с выбросами
    for i, current_string in enumerate(training_sample):

        # На основе тренироовочной выборки, составляем матрицу Маркова
        matrix = markov(current_string, result.copy(), index_by_value)

        # fig = plt.figure(figsize=(10, 10))
        # fig.add_subplot(1, 1, 1)
        # plt.title(f"Матрица Маркова", fontsize=12)
        # plt.matshow(matrix, 0)
        # fig.colorbar(plt.matshow(matrix, 0), orientation='vertical', fraction=0.04)
        # plt.clim(0, 1)
        #
        # plt.show()

        # Вычисляем порог для обработки значений
        limit = get_limit(current_string, matrix, index_by_value, window)

        # Используем на тестовых выборках
        predict_true.append(prob_of_existence(data_true[i], matrix, index_by_value) < limit)
        predict_fake.append(prob_of_existence(data_fake[i], matrix, index_by_value) < limit)

    print(f"Ложные %: {sum(predict_true) / len(predict_true)}")
    print(f"Пропущенные %: {(len(predict_fake) - sum(predict_fake)) / len(predict_fake)}")
