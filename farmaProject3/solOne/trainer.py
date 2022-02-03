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
    try:
        arr = [1.0 for nr in range(len(descr[1:]))]
        con = 0.0
        for colmn in descr[1:]:
            if all(int(x) == 0 for x in colmn[:-1]):
                con = colmn[-1]
            else:
                for nm in colmn:
                    if int(nm) != 0 and nm < len(arr):
                        arr[int(nm)] = colmn[-1]

        return arr[1:], con
    except:
        if len(descr) - 2 > 0:
            arr = [1.0 for x in range(len(descr) - 2)]
        else:
            arr = [1.0]
        return arr, 1.0


def _calc_val(w, b, X_i):
    return sum([w[i]*x for i, x in enumerate(X_i)]) + b


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

lr = 0.005

iter_count = 0
loss = [1.0 for x in a]

X = []
Y = []

for col in tr_set:
    X.append(col[:-1])
    Y.append(col[-1])
X_pre = X
X = list(zip(*X))
n = float(len(a))

pre_d = -1/n

for itr in range(epochs):
    for j, m in enumerate(a):
        Y_pred = [m*x for x in X[j]]
        D_m = pre_d * sum([x * (Y[i] - Y_pred[i]) for i, x in enumerate(X[j])])
        a[j] = m - lr * D_m
        loss[j] = D_m
    D_c = pre_d * sum([y - _calc_val(a, c, X_pre[i]) for i, y in enumerate(Y)])
    c = c - lr * D_c
    iter_count += 1
    if iter_count > 9999 or all(0.00000001 > l_val > -0.00000001 for l_val in loss):
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
