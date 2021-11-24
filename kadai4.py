import numpy as np
import random
import math
import matplotlib.pyplot as plt

def triprality(x, k):
    value = 0
    value = np.sqrt(pow(x[0], 2) + pow(x[1], 2))
    return value

def dynamics(m, neuron, input, alpha, sigma):
    #2.generate x, y
    x = np.random.normal(-1, 1, (2)) #x軸
    r = [i for i in range(1, neuron+1)]
    x_value = np.zeros((neuron))
    c = np.zeros((neuron))
    c_argmin = 0
    #3.winner unit
    x_value = triprality(x, neuron)
    for i in range(neuron):
        c[i] = np.linalg.norm(x_value - m[i])
    c_argmin = np.argmin(c)
    #print(c_argmin)
    #4.Learning
    delta_m = np.zeros((neuron, 2))
    for i in range(neuron):
        delta_m[i] = alpha * (x - m[i]) * np.exp(-pow(np.linalg.norm(c - i), 2) / 2 * pow(sigma, 2))
        m[i] += delta_m[i]
    return m

NEURON = 10 #ニューロン数
SIGMA = 0.5 #標準偏差
APLHA = 0.9 #学習係数
INPUT = 10 #入力数
plot_m = []
m = np.random.normal(-0.1, 0.1, (NEURON, 2)) #参照ベクトル（重み）
for t in range(10000):
    m = dynamics(m, NEURON, INPUT, APLHA, SIGMA)
    if (t+1) % 1000 == 0:
        #print(m)
        plot_m.append(m)
print(plot_m)
#plt.plot(plot_m)
#plt.show()
