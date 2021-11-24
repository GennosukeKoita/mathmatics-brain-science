import numpy as np
import matplotlib.pyplot as plt

def task2(x0, T):
    xt = np.dot(T, x0) / np.linalg.norm(np.dot(T, x0))
    return xt

n = 100
a = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        a[i][j] = i + j + 1

x = np.arange(n)
y = np.zeros(n)
y = np.dot(a, x)
print(y)

#task2
T = np.array([[6, -3, -7], [-1, 2, 1], [5, -3, -6]])
t = 50
xt = np.zeros((t, 3))
xt[0] = np.array([4, 0, 3])

for i in range(t-1):
    xt[i+1] = np.dot(T, xt[i]) / np.linalg.norm(np.dot(T, xt[i]))
    print(xt[i+1])

plt.plot(xt)
plt.show()