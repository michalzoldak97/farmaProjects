import sys
import fileinput


def _determine_val(description, test_var):
    test_var[:0] = [1]
    res = 0.0
    for stat in description[1:]:
        comp = 1.0
        if len(stat) > 1:
            for coeff in stat[:(len(stat) - 1)]:
                if int(coeff) == 0:
                    continue
                comp = comp * test_var[int(coeff)]
        comp = comp * stat[-1]
        res = res + comp
    return res


def calculate_res():
    description = []
    try:
        for f in (open('description1.txt', 'r')):
            line_parts = []
            for part in f.split(" "):
                line_parts.append(float(part.rstrip()))
            description.append(line_parts)
    except:
        description = 5.0
    params_str = []
    for in_col in sys.stdin:
        params_str.append(in_col.replace("\n", "").split(" "))

    params = []
    for par_col in params_str:
        new_col = []
        for par in par_col:
            new_col.append(float(par))
        params.append(new_col)

    for param in params:
        print(_determine_val(description, param))


calculate_res()
