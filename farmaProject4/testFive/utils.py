import fileinput
import argparse
import sys


def get_filepath():
    args = argparse.ArgumentParser()
    args.add_argument("-t", "--set")
    return args.parse_args().set


def get_filepath_res():
    args = argparse.ArgumentParser()
    args.add_argument("-t", "--set")
    args.add_argument("-r", "--res")
    return args.parse_args().set, args.parse_args().res


def get_data_res():
    args = argparse.ArgumentParser()
    args.add_argument("-t", "--set")
    args.add_argument("-r", "--res")
    file = []
    for line in fileinput.input(files=args.parse_args().set):
        file.append([float(i) for i in line.split()])
    res_file = []
    for line in fileinput.input(files=args.parse_args().res):
        file.append([float(i) for i in line.split()])
    x = []
    y = []
    for col in file:
        x.append(col[:-1])
        y.append(col[-1])

    res = []
    for col in res_file:
        res.append(col[-1])

    return x, y, res


def get_data(filepath):
    file = []
    for line in fileinput.input(files=filepath):
        file.append([float(i) for i in line.split()])
    x = []
    y = []
    for col in file:
        x.append(col[:-1])
        y.append(col[-1])

    return x, y


def read_sys_file():
    return [[float(i) for i in in_col.split()] for in_col in sys.stdin]
