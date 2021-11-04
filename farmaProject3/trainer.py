import fileinput
import sys
import argparse
# import matplotlib.pyplot as plt


def _get_file(filepath):
    file = []
    for line in fileinput.input(files=filepath):
        file.append([float(i) for i in line.split()])
    return file

def _get_description():
    description = []
    for in_col in sys.stdin:
        description.append([float(i) for i in in_col.split()])
    return description

def _get_function(description):
    a = []
    for col in description[1:]:
        for num in col:
            if num != 0:
                

print(_get_description())

a = [1.0]
c = 1.0

lr = 0.0001

epochs = 10000

tr_set = _get_file('train_set2.txt')

X = []
Y = []

for col in tr_set:
    X.append(col[:-1])
    Y.append(col[-1])

X = list(zip(*X))

n = float(len(X[0]))

for i in range(epochs):
    for j, m in enumerate(a):
        Y_pred = [m*x + c for x in X[j]]
        D_m = (-2/n) * sum([x * (Y[i] - Y_pred[i]) for i, x in enumerate(X[j])])
        a[j] = m - lr * D_m
    D_c = (-2/n) * sum([y - Y_pred[i] for i, y in enumerate(Y)])
    c = c - lr * D_c

print(a, c)
# print("x")
# for i in X[0]:
#     print(i)
# print("x2")
# for i in X[1]:
#     print(i)
# print("y")
# for i in Y:
#     print(i)

# Y_pred = [m*x + c for x in X]

# plt.scatter(X, Y)
# plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')
# plt.show()
