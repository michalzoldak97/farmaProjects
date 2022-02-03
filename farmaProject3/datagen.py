import numpy as np
import matplotlib.pyplot as plt
import random

# X = np.arange(-5, 5, 0.1).tolist()
# for i, num in enumerate(X):
#     X[i] = X[i] + random.random()
#
# Y = np.arange(0, 10, 0.1).tolist()

l = [[random.random()*3 for i in range(2000)] for j in range(2)]

X = [random.random()*3 for i in range(2000)]
Y = [random.random()*1 for i in range(2000)]

for i, num in enumerate(X):
    X[i] = num - Y[i] * 2.0*random.random()

plt.scatter(X, Y)
plt.show()

f = open('train_set3.txt', "w")
for i in range(len(X)):
    str_to_write = str(X[i]) + " " + str(Y[i]) + "\n"
    f.write(str_to_write)

f.close()
