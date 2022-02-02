import fileinput
import sys
import argparse


def _get_description():
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--description")
    description = []
    for line in fileinput.input(files=args.parse_args().description):
        description.append([float(i) for i in line.split()])
    return description


def _determine_val(description, test_var):
    test_var[:0] = [1]
    res = 0.0
    for stat in description[1:]:
        comp = 1.0
        for coeff in stat[:(len(stat) - 1)]:
            if int(coeff) == 0:
                continue
            comp = comp * test_var[int(coeff)]
        comp = comp * stat[-1]
        res = res + comp
    return res


def calculate_res():
    params = []
    description = _get_description()
    for in_col in sys.stdin:
        params.append([float(i) for i in in_col.split()])
    for param in params:
        print(_determine_val(description, param))


calculate_res()
