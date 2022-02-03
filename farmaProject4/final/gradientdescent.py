from mathoperations import ones


def _calc_funct_val(args, c, x_all):
    res = 0.
    for i, x in enumerate(x_all):
        res += x * args[i]

    return res + c


def _gradient_descent_reg(start, x, y, lmb, lr_, epochs):
    vec = start
    c = 1.
    pre = 1. / float(len(y))
    lmb = lmb / float(len(y))
    loss = ones(len(vec) + 1)
    for _ in range(epochs):
        y_pred = [_calc_funct_val(vec, c, col) for col in x]
        y_diff = [pred - y[i] for i, pred in enumerate(y_pred)]

        for i, arg in enumerate(vec):
            d_m = pre * sum([df * x[j][i] for j, df in enumerate(y_diff)])
            loss[i] = d_m
            d_m = d_m + (lmb * arg)
            vec[i] = arg - lr_ * d_m
        
        d_c = pre * sum(y_diff)
        loss[-1] = d_c
        # d_c = d_c + (lmb * c)
        c = c - lr_ * d_c

        if all(0.0001 > l_val > -0.0001 for l_val in loss):
            break

        if not all(param < 9999 for param in vec):
            vec = [1.0 for _ in vec]
            break

    return vec, c


def minimize(start, x, y, reg, lr_=.1, epochs=2500):
    return _gradient_descent_reg(start, x, y, reg, lr_, epochs)
