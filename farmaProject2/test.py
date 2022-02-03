import sys


def _get_description(infile):
    description = []
    for i, char in enumerate(infile):
        description.append(char.replace('\n', '').split(' '))
        for j, num in enumerate(description[i]):
            description[i][j] = float(num)
    return description


def _determine_val(description, test_var):
    test_var[:0] = [1]
    res = 0
    for stat in description[1:]:
        comp = 1
        for coeff in stat[:(len(stat)-1)]:
            if coeff == 0:
                continue
            comp = comp*test_var[int(coeff)]
        comp = comp*stat[-1]
        res = res + comp
    return res


def calculate_res():
    description = _get_description(open("description.txt", "r"))
    params = []
    for i, in_col in enumerate(sys.stdin):
        params.append(in_col.replace('\n', '').split(' '))
        for j, num in enumerate(params[i]):
            params[i][j] = float(num)
    for param in params:
        print(_determine_val(description, param))


calculate_res()
