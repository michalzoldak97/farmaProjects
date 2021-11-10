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

def is_nan(num):
    return num!=num

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

lr = 0.01

iter_count = 0
loss = [1.0 for x in a]

X = []
Y = []

for col in tr_set:
    X.append(col[:-1])
    Y.append(col[-1])

X = list(zip(*X))
n = float(len(Y))

pre_d = 1.0/n

for itr in range(epochs):
    for j, m in enumerate(a):
        Y_pred = [m*x + c for x in X[j]]
        D_m = pre_d * sum([x * (Y_pred[i] - Y[i]) for i, x in enumerate(X[j])])
        a[j] =  m - lr * D_m
        loss[j] = D_m
        D_c = pre_d * sum([(Y_pred[i] - Y[i]) for i, x in enumerate(X[j])])
        if len(a) > 9:
            lr = lr / ((iter_count + 1)*0.5)
        c = c - lr * D_c
    
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
