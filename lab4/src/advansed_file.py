import numpy as np
import os
from base_file import *

b = np.eye(2)
def norm(x):
    return (x[0]**2+x[1]**2)**0.5
def lu(A: list, permute: bool):
    n = len(A)
    p = []
    C = np.array(A.copy())
    P = np.array([np.array([0.**(abs(i-j)) for j in range(n)]) for i in range(n)])
    for i in range(n):
        pivotValue = 0
        pivot = -1
        for row in range(i, n):
            if( abs(C[ row ][ i ]) > pivotValue ):
                pivotValue = abs(C[ row ][ i ])
                pivot = row
        if( pivotValue != 0 ):
            if(permute):
                P[[pivot, i]] = P[[i, pivot]]
                C[[pivot, i]] = C[[i, pivot]]
            for j in range(i+1, n):
               C[ j ][ i ] /= C[ i ][ i ]
               for k in range(i+1, n):
                   C[ j ][ k ] -= C[ j ][ i ] * C[ i ][ k ]
    L = np.array([np.array([0. for j in range(n)]) for i in range(n)])
    U = np.array([np.array([0. for j in range(n)]) for i in range(n)])
    for i in range(n):
        for j in range(n):
            if(i>j):
                L[i][j] = C[i][j]
            if(i==j):
                L[i][j] = 1
                U[i][j] = C[i][j]
            if(i<j):
                U[i][j] = C[i][j]
    return L, U, P

def solve(L, U,P, b):
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
def obrat(J):
    b = np.array([[1,0],[0,1]])
    L, U, P = lu(J, 1)
    J_res = np.empty((2,2))
    for i in range(len(b)):
        b_i = b[:,i]
        J_o = solve(L, U, P, b_i)
        for j in range(len(J_res)):
            for k in range(len(J_res)):
                if(k == i):
                    J_res[j][k] = J_o[j]
    return J_res
def newtonV2(x_0, f, J):
    E = 1e-8
    x = []

    x.append(x_0)

    L, U, P = lu(J(x_0), True)
    Y = solve(L,U,P,f(x_0))

    x_next = x_0 - Y
    x.append(x_next)
    i = 1
    while(abs(np.linalg.norm(x[i]-x[i-1], ord=np.inf)) > E):
        L, U, P = lu(J(x[i]), True)
        Y = solve(L, U, P, f(x[i]))

        x_next = x[i] - Y
        #x_next = x[i] - f(x[i]) / (np.array([J(x[i])[0][0], J(x[i])[1][1]]))
        x.append(x_next)
        i += 1
    #print(x)
    return len(x)-1, x[-1]

# def newton(x_0, f, J):
#     E = 1e-8
#     x = []
#
#     x.append(x_0)
#     count = 0
#     F = f(x_0)
#     J_o = obrat(J(x_0))
#     x_next = x_0 - J_o.dot(F)
#     x.append(x_next)
#     count += 1
#     i = 1
#     while(abs(np.linalg.norm(x[i]-x[i-1], ord=np.inf)) > E):
#
#         J_o = obrat(J(x[i]))
#         x_next = x[i] - J_o.dot(f(x[i]))
#         #x_next = x[i] - f(x[i]) / (np.array([J(x[i])[0][0], J(x[i])[1][1]]))
#         x.append(x_next)
#         count += 1
#         i += 1
#     #print(x)
#     return count, x[-1],  J(x[-1])

# def h(t, x, z):
#     f = lambda x: np.array([alpha * x[0] - betta * x[0] * x[1], delta*x[0]*x[1] - gamma*x[1]])
#     return f(x - t*z/np.linalg.norm(z, ord=2 )).dot(f(x - t*z/np.linalg.norm(z, ord=2 )))
def tSearch(x_k, z_k):
    def f(x):
        return np.array([alpha * x[0] - betta * x[0] * x[1], delta * x[0] * x[1] - gamma * x[1]])
    def h(t):
        return f(x_k - t * z_k / (np.linalg.norm(z_k, ord=2))).dot(f(x_k - t * z_k / (np.linalg.norm(z_k, ord=2))))
    t_1 = 0.
    t_3 = 1.
    iter = 0
    while h(t_1) <= h(t_3):
        iter += 1
        t_3_d = t_3/2.
        t_3_u = t_3*2.
        if(np.linalg.norm([h(t_1)-h(t_3_d)], ord = np.inf) > np.linalg.norm([h(t_1)-h(t_3_u)], ord = np.inf)):
            t_3 = t_3_u
        else:
            t_3 = t_3_d
        if(iter>1000):
            break
    t_2 = t_3 / 2.
    return t_1, t_2, t_3
def resSearch(t_1, t_2, t_3, x_k, z_k):
    def f(x):
        return np.array([alpha * x[0] - betta * x[0] * x[1], delta * x[0] * x[1] - gamma * x[1]])

    # f = lambda x:
    def h(t):
        return f(x_k - t * z_k / (np.linalg.norm(z_k, ord=2))).dot(f(x_k - t * z_k / (np.linalg.norm(z_k, ord=2))))
    a = h(t_1) / ((t_1 - t_2) * (t_1 - t_3))
    b = h(t_2) / ((t_2 - t_1) * (t_2 - t_3))
    c = h(t_3) / ((t_3 - t_1) * (t_3 - t_2))
    t_k = (a * (t_2 + t_3) + b * (t_1 + t_3) + c * (t_1 + t_2)) / (2 * (a + b + c))
    #print(t_k)
    x_k = x_k - t_k * z_k / (np.linalg.norm(z_k, ord=2))
    return x_k

def gradient_descent_M(x_0, f, J):
    eps = 1E-8
    x_1 = x_0
    J_t = np.transpose(J(x_1))
    z_k = J_t.dot(f(x_1))
    t_1, t_2, t_3 = tSearch(x_1, z_k)
    x_k = resSearch(t_1, t_2, t_3, x_1, z_k)
    count = 1
    while np.linalg.norm(x_k-x_1, ord=np.inf)> eps:
        count += 1
        x_1 = x_k
        J_t = np.transpose(J(x_1))
        z_k = J_t.dot(f(x_1))
        t_1, t_2, t_3 = tSearch(x_1, z_k)
        x_k = resSearch(t_1, t_2, t_3, x_1, z_k)
    return count, x_k
def trans(J):
    n = len(J)
    res = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            res[j][i] = J[i][j]
    return res
def advanced():
    def J(x):
        res = np.array([[0.0,0.0],[0.0,0.0]])

        res[0][0] = alpha - betta*x[1]
        res[0][1] = -betta*x[0]
        res[1][0] = delta*x[1]
        res[1][1] = delta*x[0] - gamma
        return res
    def f(x):
        return np.array([alpha * x[0] - betta * x[0] * x[1], delta*x[0]*x[1] - gamma*x[1]])

    start_data = np.array([15 * i for i in range(0, 201)])
    count = 0
    n = len(start_data)
    #print(newtonV2(np.array([200, 200]), f, J))
    res = np.eye(n, n)
    res2 = np.eye(n, n)
    fig, ax = plt.subplots()
    sum_grad = []
    sum_newton = []
    for x_0 in range(n):

        for y_0 in range(n):

            count += 1
            os.system('CLS')

            print(f"Осталось: {100 - count / (n ** 2) * 100 :.{1}f}%")
            #print(count, f')x = {start_data[x_0]}y ={start_data[y_0]}---------')

            c, x = newtonV2(np.array([start_data[x_0], start_data[y_0]]), f, J)
            res2[x_0][y_0] = np.linalg.norm(x, ord=np.inf)
            sum_newton.append(c)
            #print('newton', c, np.linalg.norm(x, ord=np.inf)) #
            c, x = gradient_descent_M(np.array([start_data[x_0], start_data[y_0]]), f, J)
            res[x_0][y_0] = np.linalg.norm(x, ord=np.inf)
            sum_grad.append(c)

            #print('grad', c, np.linalg.norm(x, ord=np.inf))
    M_newton = sum(sum_newton) / count
    M_grad = sum(sum_grad) / count
    print("Мат. ожидание для ньютона:", M_newton, "Мат. ожидание для градиента:", M_grad)
    Sr_newton = 0
    Sr_grad = 0
    for i in range(len(sum_newton)):
        Sr_grad += (sum_grad[i] - M_grad) ** 2 / (len(sum_newton) * (len(sum_newton) - 1))
        Sr_newton += (sum_newton[i] - M_newton) ** 2 / (len(sum_newton) * (len(sum_newton) - 1))
    print("Среднеквадратичное отклонение для ньютона:", Sr_newton ** 0.5,
          "Среднеквадратичное отклонение для градиента:", Sr_grad ** 0.5)
    x_range = np.linspace(0, 3000, n)
    y_range = np.linspace(0, 3000, n)
    cs = ax.contourf(x_range, y_range, res, cmap="PuBu_r")
    # cs = ax.contourf(res, cmap ="Wistia")
    cbar = plt.colorbar(cs)
    #ax.clabel(cs)
    plt.savefig('grad_res_2.png', dpi=500)
    # print('res')
    # for i in res:
    #     for j in i:
    #         print(j, end=' ')
    #     print('\n')
    fig, ax = plt.subplots()
    cs  = ax.contourf(x_range, y_range, res, cmap="autumn")
    cbar = plt.colorbar(cs)
    # print('res2')
    # for i in res2:
    #     for j in i:
    #         print(j, end=' ')
    #     print('\n')
    #ax.clabel(cs)
    plt.savefig('grad_res_1.png', dpi=500)


if __name__ == "__main__":
    #base()
    advanced()