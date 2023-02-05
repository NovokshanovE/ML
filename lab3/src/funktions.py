import numpy as np
import matplotlib.pyplot as plt



def lu_perm(A: list, permute: bool):
    n = len(A)
    C = np.array(A.copy())
    P = np.array([np.array([0**(abs(i-j)) for j in range(n)]) for i in range(n)], dtype=np.float64)
    for i in range(n):
        max_abs = 0
        max_row = -1
        for j in range(i, n):
            if(abs(C[j][i]) > max_abs):
                max_abs = abs(C[j][i])
                max_row = j
        if(max_abs!=0):
            if(permute):
                P[[max_row, i]] = P[[i, max_row]]
                C[[max_row, i]] = C[[i, max_row]]
            for j in range(i+1, n):
               C[j][i] /= C[i][i]
               for k in range(i+1, n):
                   C[j][k] -= C[j][i] * C[i][k]
    U = np.triu(C)
    L = np.tril(C, -1)
    return P, L, U

def lu(A):
    n = len(A)
    C = np.array(A.copy())
    for i in range(n):
        for j in range(i+1, n):
            C[j][i] /= C[i][i]
            for k in range(i+1, n):
                C[j][k] -= C[j][i] * C[i][k]
    U = np.triu(C)
    L = np.tril(C, -1)
    return L, U

def solve(L, U, P, b):
    n = len(b)
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    b = np.matmul(P, b)
    for i in range(n):
        y[i] = b[i] - sum([L[i][k] * y[k] for k in range(i)])
    for i in range(n):
        x[n - i - 1] = (y[n - i - 1] - sum([U[n - i - 1][n - k - 1] * x[n - k - 1] for k in range(i)])) / \
                       U[n - i - 1][n - i - 1]
    return x

def sol(L, U, b):
    n = len(b)
    X = np.zeros(n, dtype=np.float64)
    Y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        Y[i] = b[i] - sum([L[i][k] * Y[k] for k in range(i)])
    for i in range(n):
        X[n - i - 1] = (Y[n - i - 1] - sum([U[n - i - 1][n - k - 1] * X[n - k - 1] for k in range(i)])) / \
                       U[n - i - 1][n - i - 1]
    return X

def norm(A):
    k = len(A)
    summary = 0
    for i in range(k):
        summary += A[i]**2
    return(summary**(1/2))