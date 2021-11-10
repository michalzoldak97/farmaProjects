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


def _calc_funct_val(args, c, x_all):
    res = 0.0
    for i, x in enumerate(x_all):
        res += x*args[i]
    return res + c


def _get_function(descr):
    elements = []
    for col in descr[1:]:
        for num in col[:-1]:
            elements.append(num)
    arr = [0.0 for x in range(int(max(elements)))]
    con = 0.0
    for col in descr[1:]:
        if all(int(x) == 0 for x in col[:-1]):
            con += col[-1]
        else:
            for num in col[:-1]:
                if int(num) > 0 and int(num) <= (len(arr)):
                    arr[int(num) - 1] += col[-1]
    return arr, con


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

lr = 0.1
iter_count = 0

loss = [1.0 for x in a]
loss.append(1.0)
X = []
Y = []

for col in tr_set:
    X.append(col[:-1])
    Y.append(col[-1])

n = float(len(Y))

pre_d = -2.0/n

for epoch in range(epochs):
# 1. create list of function values
    y_pred = [_calc_funct_val(a, c, col) for col in X]
# 2. calculate partial derivative foreach arg
    y_diff = [Y[i] - pred for i, pred in enumerate(y_pred)]
    
    for i, arg in enumerate(a):
        d_m  = pre_d * sum( [X[j][i] * df for j, df in enumerate(y_diff)])
        loss[i] = d_m
        a[i] = arg - lr * d_m
# 3 calculate partial derivateve for c
    d_c = pre_d * sum([df for df in y_diff])
    loss[-1] = d_c
# 4 change c 
    c = c - lr * d_c
    iter_count += 1
    if all(0.00001 > l_val > -0.00001 for l_val in loss):
        break

_write_data_out(dt_out_path, iter_count)

for i, el in enumerate(description[0]):
    sys.stdout.write(str(int(el)))
    if i < len(description[0]) - 1:
        sys.stdout.write(" ")
    else:
        sys.stdout.write("\n")

for i, m in enumerate(a):
    sys.stdout.write(str(i + 1) + " ")
    sys.stdout.write(str(m) + "\n")


sys.stdout.write("0 " + str(c) + "\n")
