import matchoperations as mat


def normalize(x):
    x_rev = mat.inverse(x)
    for col in x_rev:
        x_max = max(col)
        x_min = min(col)
        for i, x_pr in enumerate(col):
            col[i] = (x_pr - x_min) / (x_max - x_min)

    return mat.inverse(x_rev)


def normalize_universal(x, ab):
    x_rev = mat.inverse(x)
    for col in x_rev:
        x_max = max(col)
        x_min = min(col)
        for i, x_pr in enumerate(col):
            try:
                col[i] = ab[0] + ((x_pr - x_min)*(ab[1] - ab[0])) / (x_max - x_min)
            except ZeroDivisionError:
                continue

    return mat.inverse(x_rev)


def ones(l):
    return [1.0 for x in range(l)]
