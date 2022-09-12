import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plab

#Вариант 9

N = 200
M1 = np.array([0, 0]).reshape(2, 1)
M2 = np.array([1, -1]).reshape(2, 1)
M3 = np.array([-2, 1]).reshape(2, 1)

#A1 = ([0.1, 0],[0.0, 0.1]) -> B1= ([0.01, 0], [0, 0.01])
#A2 = ([0.1, 0],[0.5, 0.1]) -> B2= ([0.01, 0.05], [0.05, 0.26])
#A3 = ([0.1, 0],[0.1, 0.1]) -> B3= ([0.01, 0.01], [0.01, 0.02])

B1 = np.matrix(([0.01, 0], [0, 0.01]))
B2 = np.matrix(([0.01, 0.05], [0.05, 0.26]))
B3 = np.matrix(([0.01, 0.01], [0.01, 0.02]))


#1 Алгоритмы

def CreateSample(M, B, N=200):
    sample = None
    VectorImp = None
    for i in range(N):
        n, t = B.shape
        A = TransformationMatrix(B)
        E = np.matrix(np.random.normal(0,1,n)).T
        VectorImp = A*E+M
        if sample is None:
            sample = VectorImp
        else:
            sample = np.concatenate((sample, VectorImp), axis=1) 
    return sample.T


def TransformationMatrix(B):
    n, t = B.shape
    A = np.zeros((n,n))
    for index, v in np.ndenumerate(A):
        i,j = index
        if i == j:
            sum_prev_a = 0
            for k in range(i):
                sum_prev_a += A[i,k]**2
            A[i,j] = np.sqrt(B[i,i] - sum_prev_a)
        if 1 <= i and i < j and j <= n-1:
            sum_a_elems = 0
            for k in range(i):
                sum_a_elems += A[i,k] * A[j,k]
            A[j,i] = (B[i, j] - sum_a_elems)/A[i,i]
    return A

def EstimateM(sample):
    return (sum(sample)/len(sample)).T

def EstimateB(sample):
    return (sum([i.T * i for i in sample])/len(sample)) - EstimateM(sample) * EstimateM(sample).T

def BhattacharyyaDistance(M0, B0, M1, B1):
    return (0.25 * (M1 - M0).T * np.linalg.inv((B1 + B0) / 2) * (M1 - M0) + 0.5 * np.log(np.linalg.det((B1 + B0) / 2) / np.sqrt(np.linalg.det(B1) * np.linalg.det(B0)))).item()

def MahalanobisDistance(M0, M1, B):
    return ((M1 - M0).T * np.linalg.inv(B) * (M1 - M0)).item()


#2 Два вектора с равными В

sample1 = CreateSample(M1, B1, N)
sample2 = CreateSample(M2, B1, N)
for i in sample1:
        plt.scatter(np.array(i)[0][0], np.array(i)[0][1], marker="1", color='black')
for i in sample2:
        plt.scatter(np.array(i)[0][0], np.array(i)[0][1], marker="2", color='grey')
plab.show()
with open('#2.npy', 'wb') as file:
        np.save(file, [sample1, sample2])
#print(np.load('#2.npy'))


#3 Три вектора с разными В

sample1 = CreateSample(M1, B1, N)
sample2 = CreateSample(M2, B2, N)
sample3 = CreateSample(M3, B3, N)
for i in sample1:
        plt.scatter(np.array(i)[0][0], np.array(i)[0][1], marker="1", color='black')
for i in sample2:
        plt.scatter(np.array(i)[0][0], np.array(i)[0][1], marker="2", color='grey')
for i in sample3:
        plt.scatter(np.array(i)[0][0], np.array(i)[0][1], marker="3", color='green')
plab.show()
with open('#3.npy', 'wb') as file:
        np.save(file, [sample1, sample2, sample3])
#print(np.load('#2.npy'))


#4 Оценки

# Мат. ожмдани и корреляционной матрицы

print("M1: \n", M1, "\n")
print("Оценка M1: \n", EstimateM(sample1), "\n")
print("B1: \n", B1, "\n")
print("Оценка B1: \n", EstimateB(sample1), "\n")
print("M2: \n", M2, "\n")
print("Оценка M2: \n", EstimateM(sample2), "\n")
print("B2: \n", B2, "\n")
print("Оценка B2: \n", EstimateB(sample2), "\n")
print("M3: \n", M3, "\n")
print("Оценка M3: \n", EstimateM(sample3), "\n")
print("B3: \n", B3, "\n")
print("Оценка B3: \n", EstimateB(sample3), "\n")

# Расстояний

print("Расстояние Бхатачария для векторов с равным B: ", BhattacharyyaDistance(M1, B1, M2, B1), "\n")
print("Расстояние Махаланобиса для векторов с равным B: ", MahalanobisDistance(M1, M2, B1), "\n")
print("Расстояние Бхатачария для векторов (B1,M1) и (B2,M2): ", BhattacharyyaDistance(M1, B1, M2, B2), "\n")
print("Расстояние Бхатачария для векторов (B1,M1) и (B3,M3): ", BhattacharyyaDistance(M1, B1, M3, B3), "\n")
print("Расстояние Бхатачария для векторов (B2,M2) и (B3,M3): ", BhattacharyyaDistance(M2, B2, M3, B3), "\n")