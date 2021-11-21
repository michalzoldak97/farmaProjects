
def inverse(x):
    x_inv = [[] for i in x[0]]
    for col in x:
        for i, el in enumerate(col):
            x_inv[i].append(el)
    return x_inv
