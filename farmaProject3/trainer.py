import fileinput
import sys
import argparse
import matplotlib.pyplot as plt


m = 1.0
c = 1.0

lr = 0.0001

epochs = 10000


def _get_train_set(filepath):
    train_set = []
    for line in fileinput.input(files=filepath):
        train_set.append([float(i) for i in line.split()])
    return train_set


tr_set = _get_train_set('train_set.txt')

X = []
Y = []

for col in tr_set:
    X.append(col[0])
    Y.append(col[1])

n = float(len(X))

for i in range(epochs):
    Y_pred = [m*x + c for x in X]
    D_m = (-2/n) * sum([x * (Y[i] - Y_pred[i]) for i, x in enumerate(X)])
    D_c = (-2/n) * sum([y - Y_pred[i] for i, y in enumerate(Y)])
    m = m - lr * D_m
    c = c - lr * D_c

print(m, c)

Y_pred = [m*x + c for x in X]

plt.scatter(X, Y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')
plt.show()
