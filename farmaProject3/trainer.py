import fileinput
import sys
import argparse


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
    descr = []
    for in_col in sys.stdin:
        descr.append([float(i) for i in in_col.split()])
    return descr


def _get_iter(filepath):
    itr = 0
    for line in fileinput.input(files=filepath):
        itr = int(line.split("=")[1])
    return itr


def _get_function(descr):
    a = [0 for x in range(int(descr[0][0])+1)]
    c = 0
    for col in descr[1:]:
        for i, num in enumerate(col):
            if num != 0 and i != len(col) - 1:
                a[int(num)] = col[-1]
        if all(int(x) == 0 for x in col[:-1]):
            c = col[-1]
    return a[1:], c


def _write_data_out(filepath, iter_num):
    f = open(filepath, "w")
    str_to_write = "iterations=" + str(iter_num) + "\n"
    f.write(str_to_write)
    f.close()


tr_set_path, dt_in_path, dt_out_path = _get_filepaths()

description = _get_description()

a, c = _get_function(description)

epochs = _get_iter(dt_in_path)

tr_set = _get_file(tr_set_path)

lr = 0.01

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
        loss[j].append(D_m)
    D_c = (-2/n) * sum([y - Y_pred[i] for i, y in enumerate(Y)])
    c = c - lr * D_c
    iter_count += 1
    if lr > 0.0001:
        lr = lr - (1 / iter_count) * 0.001
    if iter_count > 100000000 or all(0.0000001 > l[-1] > -0.0000001 for l in loss):
        break

_write_data_out(dt_out_path, iter_count)

for i, el in enumerate(description[0]):
    sys.stdout.write(str(int(el)))
    if i < len(description[0]) - 1:
        sys.stdout.write(" ")
    else:
        sys.stdout.write("\n")

for i, m in enumerate(a):
    for n in range(len(a)-1):
        sys.stdout.write("0 ")
    sys.stdout.write(str(i + 1) + " ")
    sys.stdout.write(str(m) + "\n")

for n in range(len(a)):
    sys.stdout.write("0 ")
sys.stdout.write(str(c) + "\n")
