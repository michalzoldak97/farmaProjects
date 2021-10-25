import fileinput as fin


def _get_description(infile):
    description = []
    for i, char in enumerate(infile):
        description.append(char.replace('\n', '').split(' '))
        for j, num in enumerate(description[i]):
            description[i][j] = float(num)
    return description


test_var = [2, -3]
test_var[:0] = [1]
res = 0


def _determine_val(description):
    global res
    for stat in description[1:]:
        comp = 1
        for coeff in stat[:(len(stat)-1)]:
            if coeff == 0:
                continue
            comp = comp*test_var[int(coeff)]
        comp = comp*stat[-1]
        res = res + comp


_determine_val(_get_description(fin.input(files='./description.txt')))
print(res)
