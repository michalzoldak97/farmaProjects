from mathoperations import ones


def _calc_funct_val(args, c, x_all):
    res = 0.0
    for i, x in enumerate(x_all):
        res += x * args[i]

    return res + c


def _gradient_descent_reg(start, x, y, lmb, lr, epochs):
    vec = start
    c = 1.0
    pre_d = -2.0 / (float(len(y)))
    lmb = lmb / (float(len(y)))
    loss = ones(len(vec) + 1)
    for _ in range(epochs):
        y_pred = [_calc_funct_val(vec, c, col) for col in x]
        y_diff = [y[i] - pred for i, pred in enumerate(y_pred)]

        for i, arg in enumerate(vec):
            d_m = pre_d * sum([x[j][i] * df for j, df in enumerate(y_diff)])
            loss[i] = d_m
            d_m = d_m + (lmb * arg)
            vec[i] = arg - lr * d_m
        
        d_c = pre_d * sum(y_diff)
        loss[-1] = d_c
        d_c = d_c + (lmb * c)
        c = c - lr * d_c

        if all(0.001 > l_val > -0.001 for l_val in loss):
            break

        if not all(param < 999 for param in vec):
            vec = [1.0 for el in vec]
            break

    return vec, c


def minimize(start, x, y, reg, lr=.1, epochs=2500):
    return _gradient_descent_reg(start, x, y, reg, lr, epochs)
