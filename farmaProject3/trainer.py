import fileinput
import sys
import argparse
import matplotlib.pyplot as plt


def _get_filepaths():
    args = argparse.ArgumentParser()
    args.add_argument("-t", "--train_set")
    args.add_argument("-i", "--data_in")
    args.add_argument("-o", "--data_out")
    return args.parse_args().train_set, args.parse_args().data_in, args.parse_args().data_out


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


def _get_iter(filepath):
    itr = 0
    for line in fileinput.input(files=filepath):
        itr = int(line.split("=")[1])
    return itr


def _get_function(description):
    a = [0 for x in range(int(description[0][0])+1)]
    c = 0
    for col in description[1:]:
        for i, num in enumerate(col):
            if num != 0 and i != len(col) - 1:
                a[int(num)] = col[-1]
        if all(int(x) == 0 for x in col[:-1]):
            c = col[-1]
    return a[1:], c


tr_set_path, dt_in_path, dt_out_path = _get_filepaths()

a, c = _get_function(_get_description())

epochs = _get_iter(dt_in_path)

tr_set = _get_file(tr_set_path)

lr = 0.0001

iter_count = 0
loss = [[] for x in a]

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
        loss.append(D_m)
    D_c = (-2/n) * sum([y - Y_pred[i] for i, y in enumerate(Y)])
    c = c - lr * D_c
    iter_count += 1


print(a, c, iter_count)
print(loss)
# print("x")
# for i in X[0]:
#     print(i)
# print("x2")
# for i in X[1]:
#     print(i)
# print("y")
# for i in Y:
#     print(i)
#
# Y_pred = [a[0]*x + c for x in X[0]]
#
# plt.scatter(X[0], Y)
# plt.plot([min(X[0]), max(X[0])], [min(Y_pred), max(Y_pred)], color='red')
# plt.show()
