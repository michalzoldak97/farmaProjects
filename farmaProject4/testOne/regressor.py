import fileinput
import sys
import argparse


def get_filepath():
    args = argparse.ArgumentParser()
    args.add_argument("-t", "--set")
    return args.parse_args().set


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


def column(a, i):
    """
    Returns a specific column
    of a multidimensional list
    """
    return [row[i] for row in a]
def transpose(a):
    """
    Returns the transpose of a
    mutidimensional list
    """
    trans_a     =   []
    for i in range(len(a[0])):
        trans_a.append(column(a,i))
    return trans_a
def polynomial(M , i):
    """
    Helps to create the matrix A
    """
    row = [1]
    for exp in range(1, M+1):
        row.append(i ** exp)
    return row
def matrix_A(x, M):
    """
    Create matrix A to fit data
    to a polynomial of order M
    """
    N           = len(x)
    matrix_A    = []
    for i in x:
        matrix_A.append(polynomial(M,i))
    return matrix_A
def matrix_mult(a,b):
    """
    Matrix multiplication
    """
    zip_b   =   zip(*b)
    zip_b   =   list(zip_b)
    matrix  =   [[sum(a_i * b_i for a_i ,
                      b_i in zip(row_a, col_b))
                  for col_b in zip_b] for row_a in a]
    return matrix
def Identity(n):
    """
    Create a identity matrix
    """
    result  = d_2_List(n,n)
    for i in range(n):
        result[i][i]    =   1
    return result
def d_2_List(rows, cols):
    """
    Auxiliar function
    to calculate inverse
    """
    a=[]
    for row in range(rows):
        a += [[0]*cols]
    return a
def S_Matrix(m, row, k):
    """
    Square matrix
    """
    n                   =   len(m)
    r_Oper              =   Identity(n)
    r_Oper[row][row]    =   k
    return matrix_mult(r_Oper, m)
def add_S_Matrix(m, s_row, k, t_row):
    """
    Add rows of
    square matrix
    """
    n           =   len(m)
    r_oper      =   Identity(n)
    r_oper[t_row][s_row]    =   k
    return matrix_mult(r_oper, m)
def inverse(m):
    """
    Inverse of matrix m
    """
    n           =   len(m)
    assert(len(m) == len(m[0]))
    inverse     =   Identity(n)
    for col in range(n):
        d_row   =   col
        assert(m[d_row][col] != 0)
        k       =   1 / m[d_row][col]
        m       =   S_Matrix(m, d_row, k)
        inverse =   S_Matrix(inverse, d_row, k)
        s_row   =   d_row
        """
        Gauss Jordan Elimination
        """
        for t_row in range(n):
            if (s_row != t_row):
                k       =   -m[t_row][col]
                m       =   add_S_Matrix(m, s_row,
                                         k, t_row)
                inverse =   add_S_Matrix(inverse,
                                         s_row,k,
                                         t_row)
    return inverse


def function(x,pol_coeff):
    """
    Function to evaluate the polynomial from fitting process
    """
    res = 0
    for exp, coeff in enumerate(pol_coeff):
        res = res + coeff[0] * x ** exp
    return res


def fitting(M):
    """
    Main function:
    File   : file name containing data
    char_  : character separator
    M      : order of polynomial to fit data
    plot   : If True ==> Plot , if False ==> pass
    """
    x , Y       =   get_data(get_filepath())
    print(x)
    #y = [Y[i : (i+1)] for i in range(len(Y))]
    print(Y)
    A_transpose =   transpose(matrix_A(x,M))
    Matrix_S    =   matrix_mult(A_transpose,matrix_A(x,M))
    vector_Z    =   matrix_mult(A_transpose, Y)
    S_inverse   =   inverse(Matrix_S)
    pol_coeff   =   matrix_mult(S_inverse,vector_Z)
    y_calc      =   []
    for value in x:
        y_calc.append(function(value , pol_coeff))

    y_mean      =   sum(Y) / len(Y)
    sum_upper   =   0
    sum_lower   =   0
    for i in range(len(x)):
        sum_upper   =   sum_upper + (y_calc[i] - y_mean) ** 2
        sum_lower   =   sum_lower + (Y[i] - y_mean) ** 2
    R_2     =   sum_upper / sum_lower
    print(y_calc)
    return {'Coefficients': pol_coeff , 'R2' : R_2}


fitting(2)
