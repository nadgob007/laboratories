import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split


# Составляем биграммы из слов переданной строки. Возвращает массив биграмм строки
def get_bigramms(token_list):
    all_bigramm = []
    for i in range(len(token_list) - 1):
        bigramm = token_list[i] + ' ' + token_list[i + 1]
        all_bigramm.append(bigramm)
    return all_bigramm


# Принимаем матрицу частоты встречаемости биграм. Возвращаем матрицу частоты встречаемости биграмм TF
def do_tf(matrix):
    tf = np.zeros_like(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            a = np.sum(matrix[i])
            if a > 0:
                tf[i][j] = matrix[i][j] / a
            else:
                tf[i][j] = 0
    return tf.copy()


# Принимаем матрицу частоты встречаемости биграм. Возвращаем матирцу важности биграмм IDF
def do_idf(matrix):
    idf = np.zeros_like(matrix[0])
    D = len(matrix)
    a = np.sum(matrix, axis=0)
    for i in range(len(idf)):
        if a[i] > 0:
            idf[i] = np.log(D / a[i])
        else:
            idf[i] = 0
    return idf.copy()


# Получаем матрицы TF-IDF для тренировочной, тестовой выборки и метки классов. Возвращаем оценку
def desten(train, test, v):
    dist = np.zeros((len(train)))
    for i in range(len(train)):
        cur = 0
        for j in range(len(train[i])):
            cur += np.sqrt(abs(test[j] ** 2 - train[i][j] ** 2))
        dist[i] = cur
    ind = np.argsort(dist)
    k = v[ind][:3:]
    l, c = np.unique(k, return_counts=True)
    return 0 if l[np.argmax(c)] == 'ham' else 1


if __name__ == "__main__":
    # Добавляем стэмминг
    stemmer = PorterStemmer()

    # Считываем данные из csv файла
    data = pd.read_csv("./spam.csv", header=0, encoding="Windows-1252", usecols=[0, 1])

    # Исключаем из всех строк символы
    data.v2 = data.v2.apply(lambda x: x.lower().translate(str.maketrans('', '', "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~\r")).split(' '))

    # Разделяем выборку на тренировочную и  тестовую в соотношении 0.9 к 0.1
    train, test = train_test_split(data, train_size=0.9, random_state=42)
    test = test[:10]

    bigrams = []        # Биграммы в овсем тексте
    bigram_rows = []    # Биграммы в каждой строке
    for words in train.v2.values.tolist():
        words = list(filter(None, words))   # Фильтруем, значения None
        # Применяем стэминг, с помощью импортированного stemmer
        words_stem = []
        for word in words:
            words_stem.append(stemmer.stem(word))
        # Получаем массив биграмм для строки
        bigrs = get_bigramms(words_stem)
        bigram_rows.append(bigrs)
        bigrams += bigrs

    # Получаем отсортированный массив не повторяющихся (уникальных) биграмм
    uniq_bigr = np.unique(bigrams)

    """ Расчитываем сколько раз во всех текстах встречается каждая уникальная биграмма """
    ver = np.zeros((len(bigram_rows), len(uniq_bigr)))  # Матрица соответствия биграмм и их появлениям в текстах
    for i in range(len(bigram_rows)):
        for j in range(len(bigram_rows[i])):
            for k in range(len(uniq_bigr)):
                if uniq_bigr[k] == bigram_rows[i][j]:
                    ver[i][k] += 1

    # Рысчитываем TF
    tf_train = do_tf(ver)
    # Расчитываем IDF
    idf_train = do_idf(ver)
    # Расчитываем TF-IDF
    tf_idf_train = tf_train * idf_train

    with open('tf_idf_train.npy', 'wb') as f:
        np.save(f, [tf_idf_train, uniq_bigr])
    print('1.Обучение окончено')
    print('Начинаем тестирование')

    test_bigr = []
    for words in test.v2.values.tolist():
        words = list(filter(None, words))   # Фильтруем, значения None
        # Применяем стэминг, с помощью импортированного stemmer
        words_stem = []
        for word in words:
            words_stem.append(stemmer.stem(word))
        # Получаем массив биграмм для строки
        test_bigr.append(get_bigramms(words_stem))

    # Расчитываем сколько раз во всех текстах встречается каждая уникальная биграмма
    ver2 = np.zeros((len(test_bigr), len(uniq_bigr)))
    for i in range(len(test_bigr)):
        for j in range(len(test_bigr[i])):
            for k in range(len(uniq_bigr)):
                if uniq_bigr[k] == test_bigr[i][j]:
                    ver2[i][k] += 1

    # Рысчитываем TF
    tf_test = do_tf(ver2)
    # Расчитываем IDF
    idf_test = do_idf(ver2)
    # Расчитываем TF-IDF
    tf_idf_test = tf_test * idf_test
    print('2.Тестирование окончено')

    # Делаем вывод относится ли ли строка к спаму или нет
    det = np.zeros((len(tf_idf_test)))
    for i in range(len(tf_idf_test)):
        det[i] = desten(tf_idf_train, tf_idf_test[i], train.v1.values)
    print('3.Окончен расчёт')

    # Расчёт Точности и полноты
    TP = 0
    FP = 0
    FN = 0
    tmp = test.v1.values
    for i in range(len(det)):
        if det[i] == 0 and tmp[i] == 'ham':
            TP += 1
        if det[i] == 0 and tmp[i] == 'spam':
            FP += 1
        if det[i] == 1 and tmp[i] == 'ham':
            FN += 1
    print(f"Точность : {TP / (TP + FP)}, Полнота : {TP / (TP + FN)}")
