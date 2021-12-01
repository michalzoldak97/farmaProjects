import fileinput
import argparse
import sys
import math
import random


def get_filepath():
    args = argparse.ArgumentParser()
    args.add_argument("-t", "--set")

    return args.parse_args().set


def get_data(filepath):
    file = [[float(i) for i in line.split()] for line in fileinput.input(files=filepath)]
    x, y = [], []
    for col in file:
        x.append(col[:-1])
        y.append(col[-1])
        
    return x, y


def read_sys_file():
    return [[float(i) for i in in_col.split()] for in_col in sys.stdin]


def train_test_split(x, y, train_val):
    all_num = len(x)
    train_num = math.floor(all_num * train_val)
    train_idx = random.sample(range(all_num), train_num)
    test_idx = list(filter(lambda x: x not in train_idx, range(all_num)))
    x_train, y_train, x_test, y_test = [[] for _ in range(4)]
    for i in train_idx:
        x_train.append(x[i])
        y_train.append(y[i])
    for i in test_idx:
        x_test.append(x[i])
        y_test.append(y[i])
    
    return x_train, y_train, x_test, y_test